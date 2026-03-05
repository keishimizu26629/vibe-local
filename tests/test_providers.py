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


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Symlink Parent Directory Validation (H-S02)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSymlinkParentDir:
    """Test _resolve_safe_write_path detects symlinks in parent directories."""

    def test_normal_path_allowed(self, tmp_path):
        """Normal (non-symlink) path should be allowed."""
        f = tmp_path / "test.txt"
        f.write_text("hello")
        resolved, err = vc._resolve_safe_write_path(str(f))
        assert err is None
        assert resolved is not None

    def test_file_symlink_rejected(self, tmp_path):
        """Direct file symlink should be rejected."""
        real = tmp_path / "real.txt"
        real.write_text("hello")
        link = tmp_path / "link.txt"
        link.symlink_to(real)
        resolved, err = vc._resolve_safe_write_path(str(link))
        assert resolved is None
        assert "symlink" in err

    def test_parent_dir_symlink_rejected(self, tmp_path):
        """Symlinked parent directory should be rejected."""
        real_dir = tmp_path / "real_dir"
        real_dir.mkdir()
        (real_dir / "file.txt").write_text("hello")
        link_dir = tmp_path / "link_dir"
        link_dir.symlink_to(real_dir)
        resolved, err = vc._resolve_safe_write_path(str(link_dir / "file.txt"))
        assert resolved is None
        assert "symlinked directory" in err

    def test_new_file_in_normal_dir_allowed(self, tmp_path):
        """New file in a normal (non-symlink) directory should be allowed."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        resolved, err = vc._resolve_safe_write_path(str(subdir / "new.txt"))
        assert err is None
        assert resolved is not None

    def test_nonexistent_path_allowed(self, tmp_path):
        """Non-existent path with valid parent should be allowed."""
        resolved, err = vc._resolve_safe_write_path(str(tmp_path / "nonexistent.txt"))
        assert err is None


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Session ID Strength (M-S07)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSessionIDStrength:
    """Test that session IDs have sufficient entropy."""

    def test_session_id_has_sufficient_length(self):
        """Auto-generated session ID should have >20 chars of randomness."""
        cfg = mock.MagicMock()
        cfg.session_id = None
        cfg.sessions_dir = tempfile.mkdtemp()
        cfg.history_file = os.path.join(cfg.sessions_dir, "history")
        session = vc.Session(cfg, "test prompt")
        # Format: YYYYMMDD_HHMMSS_<random>
        parts = session.session_id.split("_", 2)
        assert len(parts) >= 3, f"Expected at least 3 parts, got: {session.session_id}"
        random_part = parts[2] if len(parts) > 2 else ""
        assert len(random_part) >= 20, (
            f"Random part too short ({len(random_part)} chars): {session.session_id}"
        )

    def test_session_id_uniqueness(self):
        """Two session IDs should be different."""
        cfg = mock.MagicMock()
        cfg.session_id = None
        cfg.sessions_dir = tempfile.mkdtemp()
        cfg.history_file = os.path.join(cfg.sessions_dir, "history")
        s1 = vc.Session(cfg, "test")
        s2 = vc.Session(cfg, "test")
        assert s1.session_id != s2.session_id

    def test_custom_session_id_preserved(self):
        """User-provided session ID should be used as-is."""
        cfg = mock.MagicMock()
        cfg.session_id = "my-custom-id"
        cfg.sessions_dir = tempfile.mkdtemp()
        cfg.history_file = os.path.join(cfg.sessions_dir, "history")
        session = vc.Session(cfg, "test")
        assert session.session_id == "my-custom-id"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: TeamManager
# ═══════════════════════════════════════════════════════════════════════════════


class TestTeamManager:
    """Test TeamManager CRUD operations and file persistence."""

    def test_create_team(self, tmp_path):
        tm = vc.TeamManager(str(tmp_path))
        config = tm.create_team("test-team", "Test description")
        assert config["team_name"] == "test-team"
        assert config["description"] == "Test description"
        assert config["members"] == []
        # Verify file was created
        assert os.path.exists(tm._config_path("test-team"))
        assert os.path.exists(tm._tasks_path("test-team"))

    def test_create_duplicate_team_raises(self, tmp_path):
        tm = vc.TeamManager(str(tmp_path))
        tm.create_team("test-team")
        with pytest.raises(ValueError, match="already exists"):
            tm.create_team("test-team")

    def test_get_team(self, tmp_path):
        tm = vc.TeamManager(str(tmp_path))
        tm.create_team("my-team", "desc")
        team = tm.get_team("my-team")
        assert team is not None
        assert team["team_name"] == "my-team"

    def test_get_nonexistent_team(self, tmp_path):
        tm = vc.TeamManager(str(tmp_path))
        assert tm.get_team("nope") is None

    def test_list_teams(self, tmp_path):
        tm = vc.TeamManager(str(tmp_path))
        assert tm.list_teams() == []
        tm.create_team("alpha")
        tm.create_team("beta")
        teams = sorted(tm.list_teams())
        assert teams == ["alpha", "beta"]

    def test_add_and_remove_member(self, tmp_path):
        tm = vc.TeamManager(str(tmp_path))
        tm.create_team("team")
        tm.add_member("team", "coder", "agent-1", "full")
        team = tm.get_team("team")
        assert len(team["members"]) == 1
        assert team["members"][0]["name"] == "coder"
        tm.remove_member("team", "coder")
        team = tm.get_team("team")
        assert len(team["members"]) == 0

    def test_add_duplicate_member_raises(self, tmp_path):
        tm = vc.TeamManager(str(tmp_path))
        tm.create_team("team")
        tm.add_member("team", "coder", "agent-1", "full")
        with pytest.raises(ValueError, match="already exists"):
            tm.add_member("team", "coder", "agent-2", "read-only")

    def test_delete_team(self, tmp_path):
        tm = vc.TeamManager(str(tmp_path))
        tm.create_team("doomed")
        tm.delete_team("doomed")
        assert tm.get_team("doomed") is None

    def test_task_persistence(self, tmp_path):
        tm = vc.TeamManager(str(tmp_path))
        tm.create_team("team")
        tasks = tm.load_tasks("team")
        assert tasks == {"next_id": 1, "tasks": {}}
        tasks["tasks"]["1"] = {"id": "1", "subject": "Test", "status": "pending"}
        tasks["next_id"] = 2
        tm.save_tasks("team", tasks)
        loaded = tm.load_tasks("team")
        assert loaded["tasks"]["1"]["subject"] == "Test"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: MessageBus
# ═══════════════════════════════════════════════════════════════════════════════


class TestMessageBus:
    """Test MessageBus DM, broadcast, and shutdown protocol."""

    def test_send_and_receive_dm(self):
        bus = vc.MessageBus()
        bus.register("alice")
        bus.register("bob")
        bus.send("alice", "bob", "Hello Bob")
        msg = bus.receive("bob", block=False)
        assert msg is not None
        assert msg.sender == "alice"
        assert msg.content == "Hello Bob"
        assert msg.msg_type == "message"

    def test_receive_empty_returns_none(self):
        bus = vc.MessageBus()
        bus.register("alice")
        msg = bus.receive("alice", block=False)
        assert msg is None

    def test_broadcast(self):
        bus = vc.MessageBus()
        bus.register("lead")
        bus.register("worker1")
        bus.register("worker2")
        bus.broadcast("lead", "All hands meeting")
        m1 = bus.receive("worker1", block=False)
        m2 = bus.receive("worker2", block=False)
        assert m1 is not None and m1.content == "All hands meeting"
        assert m2 is not None and m2.content == "All hands meeting"
        # Sender should NOT receive their own broadcast
        assert bus.receive("lead", block=False) is None

    def test_shutdown_protocol(self):
        bus = vc.MessageBus()
        bus.register("lead")
        bus.register("worker")
        req_id = bus.shutdown_request("lead", "worker")
        msg = bus.receive("worker", block=False)
        assert msg is not None
        assert msg.msg_type == "shutdown_request"
        assert msg.request_id == req_id
        bus.shutdown_response("worker", req_id, True, "OK")
        resp = bus.receive("lead", block=False)
        assert resp is not None
        assert resp.msg_type == "shutdown_response"
        assert "approved" in resp.content

    def test_send_to_unknown_recipient_raises(self):
        bus = vc.MessageBus()
        bus.register("alice")
        with pytest.raises(ValueError, match="Unknown recipient"):
            bus.send("alice", "nobody", "Hello?")

    def test_pending_count(self):
        bus = vc.MessageBus()
        bus.register("alice")
        assert bus.pending_count("alice") == 0
        bus.send("system", "alice", "msg1")
        bus.send("system", "alice", "msg2")
        assert bus.pending_count("alice") == 2

    def test_unregister(self):
        bus = vc.MessageBus()
        bus.register("alice")
        bus.unregister("alice")
        assert bus.pending_count("alice") == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Team Tools
# ═══════════════════════════════════════════════════════════════════════════════


class TestTeamTools:
    """Test TeamCreateTool, TeamDeleteTool, SendMessageTool."""

    def test_team_create_tool(self, tmp_path):
        tm = vc.TeamManager(str(tmp_path))
        tool = vc.TeamCreateTool(tm)
        result = tool.execute({"team_name": "my-team", "description": "Test"})
        assert "Created team" in result
        assert tm.get_team("my-team") is not None

    def test_team_create_invalid_name(self, tmp_path):
        tm = vc.TeamManager(str(tmp_path))
        tool = vc.TeamCreateTool(tm)
        result = tool.execute({"team_name": "../evil"})
        assert "Error" in result

    def test_team_delete_tool(self, tmp_path):
        tm = vc.TeamManager(str(tmp_path))
        tm.create_team("temp")
        tool = vc.TeamDeleteTool(tm)
        result = tool.execute({"team_name": "temp"})
        assert "Deleted" in result

    def test_team_delete_with_members_blocked(self, tmp_path):
        tm = vc.TeamManager(str(tmp_path))
        tm.create_team("busy")
        tm.add_member("busy", "worker", "id-1", "full")
        tool = vc.TeamDeleteTool(tm)
        result = tool.execute({"team_name": "busy"})
        assert "still has" in result

    def test_send_message_tool_dm(self):
        bus = vc.MessageBus()
        bus.register("main")
        bus.register("worker")
        tool = vc.SendMessageTool(bus, sender_name="main")
        result = tool.execute({"type": "message", "recipient": "worker", "content": "Do task"})
        assert "Message sent" in result
        msg = bus.receive("worker", block=False)
        assert msg.content == "Do task"

    def test_send_message_tool_broadcast(self):
        bus = vc.MessageBus()
        bus.register("main")
        bus.register("w1")
        bus.register("w2")
        tool = vc.SendMessageTool(bus, sender_name="main")
        result = tool.execute({"type": "broadcast", "content": "Attention"})
        assert "Broadcast sent" in result

    def test_send_message_tool_shutdown(self):
        bus = vc.MessageBus()
        bus.register("main")
        bus.register("worker")
        tool = vc.SendMessageTool(bus, sender_name="main")
        result = tool.execute({"type": "shutdown_request", "recipient": "worker"})
        assert "Shutdown request sent" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Task Store Persistence
# ═══════════════════════════════════════════════════════════════════════════════


class TestTaskStorePersistence:
    """Test that task tools use _get_task_store / _save_task_store correctly."""

    def test_task_create_in_memory(self):
        """TaskCreateTool works with default in-memory store."""
        # Reset store
        vc._task_store = {"next_id": 1, "tasks": {}}
        vc._active_team_name = None
        vc._active_team_manager = None
        tool = vc.TaskCreateTool()
        result = tool.execute({"subject": "Test", "description": "Desc"})
        assert "Created task #1" in result

    def test_task_create_with_team(self, tmp_path):
        """TaskCreateTool persists to team file when team is active."""
        tm = vc.TeamManager(str(tmp_path))
        tm.create_team("test-team")
        # Activate team context
        vc._active_team_name = "test-team"
        vc._active_team_manager = tm
        try:
            tool = vc.TaskCreateTool()
            tool.execute({"subject": "Team Task", "description": "Desc"})
            # Verify persisted to file
            tasks = tm.load_tasks("test-team")
            assert "1" in tasks["tasks"]
            assert tasks["tasks"]["1"]["subject"] == "Team Task"
        finally:
            vc._active_team_name = None
            vc._active_team_manager = None
