"""
Unit tests for LLM provider classes in vibe-coder.py.

Tests:
  1. SSE parse (OpenAICompatClient._iter_sse)
  2. tool_calls parse (OpenAICompatClient.chat_sync)
  3. OpenRouterClient (_model_supports_tools, _inject_tools_as_prompt, _format_tools_as_xml)
  4. VertexAIClient (ADC file loading, token cache)
  5. create_client() factory

All tests use unittest.mock — no external API calls.
Run with: pytest tests/test_providers.py -v
"""

import importlib.util
import io
import json
import os
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import vibe-coder.py (hyphenated filename requires importlib)
# ---------------------------------------------------------------------------
_VIBE_CODER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vibe-coder.py",
)
spec = importlib.util.spec_from_file_location("vibe_coder", _VIBE_CODER_PATH)
vc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vc)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_sse_bytes(*lines):
    """Build raw SSE byte payload from text lines.

    Each line is terminated by ``\\n``, and SSE events are separated by an
    extra ``\\n`` (i.e. blank line between events).
    """
    return "\n".join(lines).encode("utf-8")


def _make_openai_compat_client(**overrides):
    """Create a minimal OpenAICompatClient for testing."""
    defaults = dict(
        base_url="http://localhost:11434/v1",
        api_key="test-key",
        default_model="test-model",
    )
    defaults.update(overrides)
    return vc.OpenAICompatClient(**defaults)


SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from disk",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "encoding": {"type": "string", "description": "Encoding"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SSE Parse Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSSEParsing:
    """Tests for OpenAICompatClient._iter_sse()."""

    def test_normal_sse_data(self):
        """Normal SSE events should be yielded as parsed dicts."""
        chunk1 = {"id": "1", "choices": [{"delta": {"content": "Hello"}}]}
        chunk2 = {"id": "2", "choices": [{"delta": {"content": " World"}}]}
        raw = _make_sse_bytes(
            f"data: {json.dumps(chunk1)}",
            "",
            f"data: {json.dumps(chunk2)}",
            "",
            "data: [DONE]",
            "",
        )
        resp = io.BytesIO(raw)
        client = _make_openai_compat_client()
        results = list(client._iter_sse(resp))
        assert len(results) == 2
        assert results[0] == chunk1
        assert results[1] == chunk2

    def test_empty_lines_ignored(self):
        """Empty lines between events should not produce output."""
        chunk = {"id": "1", "choices": [{"delta": {"content": "Hi"}}]}
        raw = _make_sse_bytes(
            "",
            "",
            f"data: {json.dumps(chunk)}",
            "",
            "",
            "",
            "data: [DONE]",
            "",
        )
        resp = io.BytesIO(raw)
        client = _make_openai_compat_client()
        results = list(client._iter_sse(resp))
        assert len(results) == 1
        assert results[0] == chunk

    def test_comment_lines_ignored(self):
        """SSE comment lines (starting with ':') should be skipped."""
        chunk = {"id": "1", "choices": [{"delta": {"content": "OK"}}]}
        raw = _make_sse_bytes(
            ": OPENROUTER PROCESSING",
            f"data: {json.dumps(chunk)}",
            "",
            ": keep-alive",
            "data: [DONE]",
            "",
        )
        resp = io.BytesIO(raw)
        client = _make_openai_compat_client()
        results = list(client._iter_sse(resp))
        assert len(results) == 1
        assert results[0]["id"] == "1"

    def test_done_terminates_stream(self):
        """'data: [DONE]' should terminate the generator immediately."""
        chunk1 = {"id": "1", "choices": [{"delta": {"content": "A"}}]}
        chunk2 = {"id": "2", "choices": [{"delta": {"content": "B"}}]}
        raw = _make_sse_bytes(
            f"data: {json.dumps(chunk1)}",
            "",
            "data: [DONE]",
            "",
            f"data: {json.dumps(chunk2)}",
            "",
        )
        resp = io.BytesIO(raw)
        client = _make_openai_compat_client()
        results = list(client._iter_sse(resp))
        # chunk2 should NOT be yielded because [DONE] comes first
        assert len(results) == 1
        assert results[0]["id"] == "1"

    def test_invalid_json_skipped(self):
        """Malformed JSON in data: lines should be silently skipped."""
        chunk = {"id": "1", "choices": [{"delta": {"content": "OK"}}]}
        raw = _make_sse_bytes(
            "data: {invalid json!!!}",
            "",
            f"data: {json.dumps(chunk)}",
            "",
            "data: [DONE]",
            "",
        )
        resp = io.BytesIO(raw)
        client = _make_openai_compat_client()
        results = list(client._iter_sse(resp))
        assert len(results) == 1
        assert results[0] == chunk

    def test_empty_stream(self):
        """An empty stream (EOF immediately) should yield nothing."""
        resp = io.BytesIO(b"")
        client = _make_openai_compat_client()
        results = list(client._iter_sse(resp))
        assert results == []


# ═══════════════════════════════════════════════════════════════════════════════
# 2. tool_calls Parse Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestToolCallsParsing:
    """Tests for OpenAICompatClient.chat_sync() tool_calls extraction."""

    def _mock_chat_response(self, message_body):
        """Return a mock non-streaming response with the given message."""
        return {
            "choices": [{"message": message_body}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }

    def test_single_tool_call(self):
        """A single tool_call should be normalized correctly."""
        msg = {
            "content": "",
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "/tmp/test.txt"}',
                    },
                }
            ],
        }
        client = _make_openai_compat_client()
        with patch.object(client, "chat", return_value=self._mock_chat_response(msg)):
            result = client.chat_sync(model="test", messages=[])

        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0]
        assert tc["id"] == "call_abc123"
        assert tc["name"] == "read_file"
        assert tc["arguments"] == {"path": "/tmp/test.txt"}

    def test_multiple_tool_calls(self):
        """Multiple tool_calls in one response should all be parsed."""
        msg = {
            "content": "I'll read both files.",
            "tool_calls": [
                {
                    "id": "call_001",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "/a.txt"}',
                    },
                },
                {
                    "id": "call_002",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": '{"path": "/b.txt", "content": "hello"}',
                    },
                },
            ],
        }
        client = _make_openai_compat_client()
        with patch.object(client, "chat", return_value=self._mock_chat_response(msg)):
            result = client.chat_sync(model="test", messages=[])

        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0]["name"] == "read_file"
        assert result["tool_calls"][1]["name"] == "write_file"
        assert result["tool_calls"][1]["arguments"]["content"] == "hello"
        # <think> stripping on content
        assert result["content"] == "I'll read both files."

    def test_no_tool_calls(self):
        """Response without tool_calls should return empty list."""
        msg = {"content": "Just text, no tools.", "role": "assistant"}
        client = _make_openai_compat_client()
        with patch.object(client, "chat", return_value=self._mock_chat_response(msg)):
            result = client.chat_sync(model="test", messages=[])

        assert result["tool_calls"] == []
        assert result["content"] == "Just text, no tools."

    def test_think_tags_stripped(self):
        """<think>...</think> blocks should be removed from content."""
        msg = {
            "content": "<think>Let me reason...</think>Here is the answer.",
        }
        client = _make_openai_compat_client()
        with patch.object(client, "chat", return_value=self._mock_chat_response(msg)):
            result = client.chat_sync(model="test", messages=[])

        assert result["content"] == "Here is the answer."

    def test_invalid_json_arguments_fallback(self):
        """Invalid JSON in arguments should fall back to raw string."""
        msg = {
            "content": "",
            "tool_calls": [
                {
                    "id": "call_bad",
                    "type": "function",
                    "function": {
                        "name": "some_tool",
                        "arguments": "not valid json {{{",
                    },
                }
            ],
        }
        client = _make_openai_compat_client()
        with patch.object(client, "chat", return_value=self._mock_chat_response(msg)):
            result = client.chat_sync(model="test", messages=[])

        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["arguments"] == {"raw": "not valid json {{{"}

    def test_tool_call_without_id_gets_generated(self):
        """tool_call without an id should get a generated one."""
        msg = {
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "/x"}',
                    },
                }
            ],
        }
        client = _make_openai_compat_client()
        with patch.object(client, "chat", return_value=self._mock_chat_response(msg)):
            result = client.chat_sync(model="test", messages=[])

        tc = result["tool_calls"][0]
        assert tc["id"].startswith("call_")
        assert len(tc["id"]) > 5


# ═══════════════════════════════════════════════════════════════════════════════
# 3. OpenRouterClient Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestOpenRouterClient:
    """Tests for OpenRouterClient-specific logic."""

    def _make_client(self, model="qwen/qwen-2.5-coder-32b-instruct"):
        return vc.OpenRouterClient(api_key="or-test-key", model=model)

    # --- _model_supports_tools ---

    def test_model_supports_tools_claude(self):
        client = self._make_client()
        assert client._model_supports_tools("anthropic/claude-3.5-sonnet") is True

    def test_model_supports_tools_openai(self):
        client = self._make_client()
        assert client._model_supports_tools("openai/gpt-4o") is True

    def test_model_supports_tools_google(self):
        client = self._make_client()
        assert client._model_supports_tools("google/gemini-2.0-flash") is True

    def test_model_supports_tools_qwen3(self):
        client = self._make_client()
        assert client._model_supports_tools("qwen/qwen3-235b") is True

    def test_model_supports_tools_llama33(self):
        client = self._make_client()
        assert client._model_supports_tools("meta-llama/llama-3.3-70b") is True

    def test_model_not_supports_tools_qwen25(self):
        """qwen-2.5 series is NOT in TOOL_CAPABLE_PREFIXES."""
        client = self._make_client()
        assert client._model_supports_tools("qwen/qwen-2.5-coder-32b-instruct") is False

    def test_model_not_supports_tools_deepseek(self):
        client = self._make_client()
        assert client._model_supports_tools("deepseek/deepseek-coder") is False

    def test_model_not_supports_tools_unknown(self):
        client = self._make_client()
        assert client._model_supports_tools("some/random-model") is False

    # --- _format_tools_as_xml ---

    def test_format_tools_as_xml_single(self):
        """Single tool should produce a single formatted line."""
        tools = [SAMPLE_TOOLS[0]]
        result = vc.OpenRouterClient._format_tools_as_xml(tools)
        assert "read_file" in result
        assert "path: string(required)" in result
        assert "encoding: string" in result
        assert "Read a file from disk" in result

    def test_format_tools_as_xml_multiple(self):
        """Multiple tools should each have their own line."""
        result = vc.OpenRouterClient._format_tools_as_xml(SAMPLE_TOOLS)
        lines = result.strip().split("\n")
        assert len(lines) == 2
        assert lines[0].startswith("- read_file(")
        assert lines[1].startswith("- write_file(")

    def test_format_tools_as_xml_required_annotation(self):
        """Required params should be annotated with '(required)'."""
        result = vc.OpenRouterClient._format_tools_as_xml(SAMPLE_TOOLS)
        # read_file: path is required, encoding is not
        assert "path: string(required)" in result
        # "encoding" should NOT have (required)
        # Extract the encoding part from the read_file line
        for line in result.split("\n"):
            if "read_file" in line:
                assert "encoding: string," in line or "encoding: string)" in line
                assert "encoding: string(required)" not in line
                break

    def test_format_tools_as_xml_unwrapped_format(self):
        """Unwrapped tool format (no 'function' wrapper) should also work."""
        tools = [
            {
                "name": "list_dir",
                "description": "List directory",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ]
        result = vc.OpenRouterClient._format_tools_as_xml(tools)
        assert "list_dir" in result
        assert "List directory" in result

    # --- _inject_tools_as_prompt ---

    def test_inject_tools_with_existing_system_message(self):
        """Tools should be appended to the existing system message."""
        client = self._make_client()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        result = client._inject_tools_as_prompt(messages, SAMPLE_TOOLS)

        # Original messages should not be mutated
        assert messages[0]["content"] == "You are a helpful assistant."

        # The returned system message should contain original + tools
        sys_msg = result[0]
        assert sys_msg["role"] == "system"
        assert "You are a helpful assistant." in sys_msg["content"]
        assert "## Available Tools" in sys_msg["content"]
        assert "read_file" in sys_msg["content"]
        assert "write_file" in sys_msg["content"]
        assert '<function=TOOL_NAME>' in sys_msg["content"]

    def test_inject_tools_without_system_message(self):
        """When no system message exists, one should be inserted at index 0."""
        client = self._make_client()
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        result = client._inject_tools_as_prompt(messages, SAMPLE_TOOLS)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "## Available Tools" in result[0]["content"]
        assert result[1]["role"] == "user"

    def test_inject_tools_does_not_mutate_original(self):
        """The original messages list should not be modified."""
        client = self._make_client()
        messages = [
            {"role": "system", "content": "Original"},
            {"role": "user", "content": "Hello"},
        ]
        original_len = len(messages)
        original_content = messages[0]["content"]
        client._inject_tools_as_prompt(messages, SAMPLE_TOOLS)

        assert len(messages) == original_len
        assert messages[0]["content"] == original_content

    # --- MODEL_ALIASES ---

    def test_alias_qwen3_resolves(self):
        """Short alias 'qwen3' should resolve to full model ID."""
        client = self._make_client(model="qwen3")
        assert client.default_model == "qwen/qwen3-coder"

    def test_alias_qwen3_coder_resolves(self):
        client = self._make_client(model="qwen3-coder")
        assert client.default_model == "qwen/qwen3-coder"

    def test_alias_qwen25_resolves(self):
        """Short alias 'qwen2.5' should resolve to full model ID."""
        client = self._make_client(model="qwen2.5")
        assert client.default_model == "qwen/qwen-2.5-coder-32b-instruct"

    def test_alias_qwen25_coder_resolves(self):
        client = self._make_client(model="qwen2.5-coder")
        assert client.default_model == "qwen/qwen-2.5-coder-32b-instruct"

    def test_full_model_id_passes_through(self):
        """Full model IDs should not be altered."""
        client = self._make_client(model="qwen/qwen3-coder")
        assert client.default_model == "qwen/qwen3-coder"

    def test_unknown_model_passes_through(self):
        """Non-alias model names should pass through unchanged."""
        client = self._make_client(model="anthropic/claude-3.5-sonnet")
        assert client.default_model == "anthropic/claude-3.5-sonnet"

    def test_alias_resolved_in_chat(self):
        """Aliases passed to chat() should be resolved."""
        client = self._make_client(model="qwen/qwen3-coder")
        with patch.object(
            vc.OpenAICompatClient, "chat", return_value=iter([])
        ) as mock_chat:
            list(client.chat(
                model="qwen2.5",
                messages=[{"role": "user", "content": "hi"}],
                tools=SAMPLE_TOOLS,
            ))
            mock_chat.assert_called_once()
            call_args = mock_chat.call_args
            # Should be resolved to full ID and routed as non-tool model
            assert call_args[1].get("tools") is None or call_args[0][2] is None

    # --- chat() routing ---

    def test_chat_routes_tools_for_capable_model(self):
        """Tool-capable models should pass tools to the parent chat()."""
        client = self._make_client(model="anthropic/claude-3.5-sonnet")
        with patch.object(
            vc.OpenAICompatClient, "chat", return_value=iter([])
        ) as mock_chat:
            list(client.chat(
                model="anthropic/claude-3.5-sonnet",
                messages=[{"role": "user", "content": "hi"}],
                tools=SAMPLE_TOOLS,
            ))
            mock_chat.assert_called_once()
            _, kwargs = mock_chat.call_args
            # tools should be passed through
            assert kwargs.get("tools") == SAMPLE_TOOLS or mock_chat.call_args[0][2] == SAMPLE_TOOLS

    def test_chat_injects_xml_for_non_capable_model(self):
        """Non-tool models should get tools=None and XML in system prompt."""
        client = self._make_client(model="qwen/qwen-2.5-coder-32b-instruct")
        with patch.object(
            vc.OpenAICompatClient, "chat", return_value=iter([])
        ) as mock_chat:
            list(client.chat(
                model="qwen/qwen-2.5-coder-32b-instruct",
                messages=[{"role": "user", "content": "hi"}],
                tools=SAMPLE_TOOLS,
            ))
            mock_chat.assert_called_once()
            call_args = mock_chat.call_args
            # tools param should be None
            if call_args.kwargs:
                assert call_args.kwargs.get("tools") is None
            else:
                # positional: model, messages, tools, stream
                assert call_args[0][2] is None


# ═══════════════════════════════════════════════════════════════════════════════
# 4. VertexAIClient Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestVertexAIClient:
    """Tests for VertexAIClient ADC loading and token caching."""

    def _make_client(self, **kwargs):
        defaults = dict(
            project_id="test-project",
            location="us-central1",
            model="gemini-2.0-flash",
        )
        defaults.update(kwargs)
        return vc.VertexAIClient(**defaults)

    # --- MODEL_ALIASES ---

    def test_alias_qwen3_coder_resolves(self):
        client = self._make_client(model="qwen3-coder")
        assert client.default_model == "qwen/qwen3-coder-480b-a35b-instruct-maas"

    def test_alias_qwen3_resolves_to_lightweight(self):
        """Short alias 'qwen3' should resolve to the lightweight 235b model."""
        client = self._make_client(model="qwen3")
        assert client.default_model == "qwen/qwen3-235b-a22b-instruct-2507-maas"

    def test_alias_qwen3_235b_resolves(self):
        client = self._make_client(model="qwen3-235b")
        assert client.default_model == "qwen/qwen3-235b-a22b-instruct-2507-maas"

    def test_full_model_id_passes_through(self):
        client = self._make_client(model="qwen/qwen3-coder-480b-a35b-instruct-maas")
        assert client.default_model == "qwen/qwen3-coder-480b-a35b-instruct-maas"

    def test_alias_resolved_in_chat(self):
        """Aliases passed to chat() should be resolved."""
        client = self._make_client(model="qwen/qwen3-coder-480b-a35b-instruct-maas")
        client._token_cache = "test-token"
        client._token_expiry = time.time() + 3600
        with patch.object(
            vc.OpenAICompatClient, "chat", return_value=iter([])
        ) as mock_chat:
            list(client.chat(
                model="qwen3",
                messages=[{"role": "user", "content": "hi"}],
            ))
            mock_chat.assert_called_once()
            call_model = mock_chat.call_args[1].get("model") or mock_chat.call_args[0][0]
            assert call_model == "qwen/qwen3-235b-a22b-instruct-2507-maas"

    # --- ADC file loading ---

    def test_load_adc_from_env_var(self):
        """GOOGLE_APPLICATION_CREDENTIALS should be tried first."""
        creds = {
            "type": "authorized_user",
            "client_id": "cid",
            "client_secret": "csecret",
            "refresh_token": "rtoken",
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(creds, f)
            f.flush()
            tmp_path = f.name

        try:
            client = self._make_client()
            with patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": tmp_path}):
                loaded = client._load_adc_credentials()
                assert loaded["type"] == "authorized_user"
                assert loaded["client_id"] == "cid"
                assert loaded["refresh_token"] == "rtoken"
        finally:
            os.unlink(tmp_path)

    def test_load_adc_from_default_path(self):
        """When env var is not set, ADC_PATHS should be searched."""
        creds = {
            "type": "service_account",
            "project_id": "my-project",
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(creds, f)
            f.flush()
            tmp_path = f.name

        try:
            client = self._make_client()
            # Clear env var and patch ADC_PATHS
            with patch.dict(
                os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": ""}, clear=False
            ):
                with patch.object(
                    vc.VertexAIClient, "ADC_PATHS", [tmp_path]
                ):
                    loaded = client._load_adc_credentials()
                    assert loaded["type"] == "service_account"
        finally:
            os.unlink(tmp_path)

    def test_load_adc_not_found_raises(self):
        """When no ADC file is found, RuntimeError should be raised."""
        client = self._make_client()
        with patch.dict(
            os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": ""}, clear=False
        ):
            with patch.object(
                vc.VertexAIClient, "ADC_PATHS", ["/nonexistent/path.json"]
            ):
                with pytest.raises(RuntimeError, match="GCP credentials not found"):
                    client._load_adc_credentials()

    def test_load_adc_caches_result(self):
        """Second call to _load_adc_credentials should return cached value."""
        creds = {"type": "authorized_user", "client_id": "x",
                 "client_secret": "y", "refresh_token": "z"}
        client = self._make_client()
        client._adc_creds = creds  # pre-populate cache

        # Should return cached value without touching filesystem
        loaded = client._load_adc_credentials()
        assert loaded is creds

    # --- Token cache ---

    def test_token_cache_returns_cached_when_valid(self):
        """Cached token should be returned if not expired."""
        client = self._make_client()
        client._token_cache = "cached-token-abc"
        client._token_expiry = time.time() + 3600  # 1 hour from now

        with patch.object(client, "_load_adc_credentials") as mock_load:
            token = client._get_access_token()
            assert token == "cached-token-abc"
            # Should NOT call _load_adc_credentials (cache hit)
            mock_load.assert_not_called()

    def test_token_cache_refreshes_when_expired(self):
        """Expired token should trigger a refresh via OAuth2."""
        client = self._make_client()
        client._token_cache = "old-token"
        client._token_expiry = 0  # already expired

        creds = {
            "type": "authorized_user",
            "client_id": "cid",
            "client_secret": "csecret",
            "refresh_token": "rtoken",
        }
        client._adc_creds = creds

        # Mock the HTTP call to token endpoint
        token_response = json.dumps({
            "access_token": "new-fresh-token",
            "expires_in": 3600,
            "token_type": "Bearer",
        }).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = token_response
        mock_resp.decode = lambda: token_response.decode()

        with patch("urllib.request.urlopen", return_value=mock_resp):
            token = client._get_access_token()

        assert token == "new-fresh-token"
        assert client._token_cache == "new-fresh-token"
        # Expiry should be set to now + 3600 - 300 (= now + 3300)
        assert client._token_expiry > time.time() + 3200

    def test_token_cache_with_mocked_time(self):
        """Token cache expiry logic tested with mocked time.time()."""
        client = self._make_client()

        creds = {
            "type": "authorized_user",
            "client_id": "cid",
            "client_secret": "csecret",
            "refresh_token": "rtoken",
        }
        client._adc_creds = creds

        token_response = json.dumps({
            "access_token": "token-v1",
            "expires_in": 3600,
        }).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = token_response

        base_time = 1000000.0

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with patch("time.time", return_value=base_time):
                token = client._get_access_token()
                assert token == "token-v1"
                # expiry = base_time + 3600 - 300 = base_time + 3300
                assert client._token_expiry == base_time + 3300

        # Still within cache validity (base_time + 1000 < base_time + 3300)
        with patch("time.time", return_value=base_time + 1000):
            token = client._get_access_token()
            assert token == "token-v1"  # cached

        # Token expired (base_time + 4000 > base_time + 3300)
        token_response_v2 = json.dumps({
            "access_token": "token-v2",
            "expires_in": 3600,
        }).encode()
        mock_resp_v2 = MagicMock()
        mock_resp_v2.read.return_value = token_response_v2

        with patch("urllib.request.urlopen", return_value=mock_resp_v2):
            with patch("time.time", return_value=base_time + 4000):
                token = client._get_access_token()
                assert token == "token-v2"  # refreshed

    def test_build_headers_includes_bearer_token(self):
        """_build_headers should include Authorization: Bearer <token>."""
        client = self._make_client()
        client._token_cache = "my-token"
        client._token_expiry = time.time() + 3600

        headers = client._build_headers()
        assert headers["Authorization"] == "Bearer my-token"
        assert "application/json" in headers["Content-Type"]

    def test_check_connection_success(self):
        """check_connection should return (True, [model]) on success."""
        client = self._make_client()
        client._token_cache = "valid-token"
        client._token_expiry = time.time() + 3600

        ok, models = client.check_connection()
        assert ok is True
        assert "gemini-2.0-flash" in models

    def test_check_connection_failure(self):
        """check_connection should return (False, [error]) on failure."""
        client = self._make_client()
        with patch.object(
            client, "_get_access_token",
            side_effect=RuntimeError("Auth failed"),
        ):
            ok, models = client.check_connection()
            assert ok is False
            assert any("Auth failed" in m for m in models)

    def test_unknown_credential_type_raises(self):
        """Unknown ADC type should raise RuntimeError."""
        client = self._make_client()
        client._adc_creds = {"type": "weird_type"}
        client._token_expiry = 0

        with pytest.raises(RuntimeError, match="Unknown ADC credential type"):
            client._get_access_token()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. create_client() Factory Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCreateClient:
    """Tests for the create_client() factory function."""

    def _make_config(self, **overrides):
        """Build a minimal config namespace."""
        defaults = dict(
            provider="openai-compat",
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:8080/v1",
            max_tokens=4096,
            temperature=0.7,
            context_window=32768,
            timeout=300,
            debug=False,
            ollama_host="http://localhost:11434",
            vertexai_project="test-project",
            vertexai_location="us-central1",
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_create_openai_compat_client(self):
        config = self._make_config(provider="openai-compat")
        client = vc.create_client(config)
        assert isinstance(client, vc.OpenAICompatClient)
        assert client.base_url == "http://localhost:8080/v1"

    def test_create_openrouter_client(self):
        config = self._make_config(provider="openrouter")
        client = vc.create_client(config)
        assert isinstance(client, vc.OpenRouterClient)
        assert "openrouter.ai" in client.base_url

    def test_create_openrouter_without_api_key_raises(self):
        config = self._make_config(provider="openrouter", api_key="")
        with pytest.raises(ValueError, match="OpenRouter requires an API key"):
            vc.create_client(config)

    def test_create_vertexai_client(self):
        config = self._make_config(provider="vertexai")
        client = vc.create_client(config)
        assert isinstance(client, vc.VertexAIClient)
        assert "aiplatform.googleapis.com" in client.base_url
        assert client.project_id == "test-project"

    def test_create_vertexai_without_project_raises(self):
        config = self._make_config(provider="vertexai", vertexai_project="")
        with pytest.raises(ValueError, match="VertexAI requires a project ID"):
            vc.create_client(config)

    def test_create_ollama_client(self):
        config = self._make_config(provider="ollama")
        client = vc.create_client(config)
        assert isinstance(client, vc.OllamaClient)

    def test_unknown_provider_raises(self):
        config = self._make_config(provider="nonexistent")
        with pytest.raises(ValueError, match="Unknown provider: nonexistent"):
            vc.create_client(config)

    def test_missing_provider_defaults_to_ollama(self):
        """When config has no 'provider' attribute, should default to 'ollama'."""
        config = self._make_config()
        del config.provider  # remove the attribute
        client = vc.create_client(config)
        assert isinstance(client, vc.OllamaClient)

    def test_openai_compat_without_base_url_raises(self):
        config = self._make_config(provider="openai-compat", base_url="")
        with pytest.raises(ValueError, match="openai-compat requires a base URL"):
            vc.create_client(config)


# ═══════════════════════════════════════════════════════════════════════════════
# HookRunner tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestHookRunner:
    """Tests for the HookRunner class."""

    def test_no_hooks_file_allows(self, tmp_path):
        """When hooks.json does not exist, all tools should be allowed."""
        runner = vc.HookRunner(str(tmp_path / "nonexistent.json"))
        allowed, reason = runner.run_pre_tool_use("Bash", {"command": "rm -rf /"})
        assert allowed is True
        assert reason is None

    def test_empty_hooks_allows(self, tmp_path):
        """When hooks.json has no PreToolUse entries, all tools allowed."""
        hooks_file = tmp_path / "hooks.json"
        hooks_file.write_text(json.dumps({"hooks": {}}))
        runner = vc.HookRunner(str(hooks_file))
        allowed, reason = runner.run_pre_tool_use("Bash", {"command": "ls"})
        assert allowed is True

    def test_matcher_filters_tool(self, tmp_path):
        """Hook with matcher='Bash' should not run for 'Read' tool."""
        script = tmp_path / "block.sh"
        script.write_text("#!/bin/bash\nexit 2")
        script.chmod(0o755)
        hooks_file = tmp_path / "hooks.json"
        hooks_file.write_text(json.dumps({
            "hooks": {
                "PreToolUse": [{
                    "matcher": "Bash",
                    "hooks": [{"type": "command", "command": str(script)}]
                }]
            }
        }))
        runner = vc.HookRunner(str(hooks_file))
        allowed, reason = runner.run_pre_tool_use("Read", {"file_path": "/etc/passwd"})
        assert allowed is True

    def test_script_exit_0_allows(self, tmp_path):
        """Script returning exit 0 should allow execution."""
        script = tmp_path / "allow.sh"
        script.write_text("#!/bin/bash\nexit 0")
        script.chmod(0o755)
        hooks_file = tmp_path / "hooks.json"
        hooks_file.write_text(json.dumps({
            "hooks": {
                "PreToolUse": [{
                    "matcher": "Bash",
                    "hooks": [{"type": "command", "command": str(script)}]
                }]
            }
        }))
        runner = vc.HookRunner(str(hooks_file))
        allowed, reason = runner.run_pre_tool_use("Bash", {"command": "ls -la"})
        assert allowed is True
        assert reason is None

    def test_script_exit_2_blocks(self, tmp_path):
        """Script returning exit 2 should block execution."""
        script = tmp_path / "deny.sh"
        script.write_text('#!/bin/bash\necho "Dangerous command" >&2\nexit 2')
        script.chmod(0o755)
        hooks_file = tmp_path / "hooks.json"
        hooks_file.write_text(json.dumps({
            "hooks": {
                "PreToolUse": [{
                    "matcher": "Bash",
                    "hooks": [{"type": "command", "command": str(script)}]
                }]
            }
        }))
        runner = vc.HookRunner(str(hooks_file))
        allowed, reason = runner.run_pre_tool_use("Bash", {"command": "rm -rf /"})
        assert allowed is False
        assert "Dangerous command" in reason

    def test_script_receives_json_payload(self, tmp_path):
        """Script should receive correct JSON on stdin."""
        script = tmp_path / "check.sh"
        output_file = tmp_path / "received.json"
        script.write_text(f'#!/bin/bash\ncat > {output_file}\nexit 0')
        script.chmod(0o755)
        hooks_file = tmp_path / "hooks.json"
        hooks_file.write_text(json.dumps({
            "hooks": {
                "PreToolUse": [{
                    "matcher": "Bash",
                    "hooks": [{"type": "command", "command": str(script)}]
                }]
            }
        }))
        runner = vc.HookRunner(str(hooks_file))
        runner.run_pre_tool_use("Bash", {"command": "echo hello"})
        received = json.loads(output_file.read_text())
        assert received["tool_name"] == "Bash"
        assert received["tool_input"]["command"] == "echo hello"

    def test_missing_script_fails_open(self, tmp_path):
        """When script does not exist, should fail open (allow)."""
        hooks_file = tmp_path / "hooks.json"
        hooks_file.write_text(json.dumps({
            "hooks": {
                "PreToolUse": [{
                    "matcher": "Bash",
                    "hooks": [{"type": "command", "command": "/nonexistent/script.sh"}]
                }]
            }
        }))
        runner = vc.HookRunner(str(hooks_file))
        allowed, reason = runner.run_pre_tool_use("Bash", {"command": "ls"})
        assert allowed is True

    def test_symlink_hooks_file_rejected(self, tmp_path):
        """Symlinked hooks.json should be rejected for security."""
        real_file = tmp_path / "real_hooks.json"
        real_file.write_text(json.dumps({
            "hooks": {
                "PreToolUse": [{
                    "matcher": "Bash",
                    "hooks": [{"type": "command", "command": "exit 2"}]
                }]
            }
        }))
        link = tmp_path / "hooks.json"
        link.symlink_to(real_file)
        runner = vc.HookRunner(str(link))
        # Symlink rejected — no hooks loaded, so allow
        allowed, reason = runner.run_pre_tool_use("Bash", {"command": "rm -rf /"})
        assert allowed is True

    def test_empty_matcher_matches_all_tools(self, tmp_path):
        """Hook with empty matcher should run for all tools."""
        script = tmp_path / "block_all.sh"
        script.write_text('#!/bin/bash\necho "blocked" >&2\nexit 2')
        script.chmod(0o755)
        hooks_file = tmp_path / "hooks.json"
        hooks_file.write_text(json.dumps({
            "hooks": {
                "PreToolUse": [{
                    "matcher": "",
                    "hooks": [{"type": "command", "command": str(script)}]
                }]
            }
        }))
        runner = vc.HookRunner(str(hooks_file))
        allowed, _ = runner.run_pre_tool_use("Write", {"file_path": "/etc/passwd"})
        assert allowed is False
