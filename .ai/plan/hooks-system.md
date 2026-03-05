# Hooks システム実装

## 概要
vibe-coderにPre-tool-useフックを実装し、ツール実行前に外部スクリプトでコマンドを検証・ブロックする機能を追加する。Claude Codeのhooksと互換性のあるJSON入力形式を採用し、既存の`deny-check.sh`をそのまま流用可能にする。

## 現状の問題
- vibe-coderにはフック機構がなく、危険なコマンドの検出は `PermissionMgr._ALWAYS_CONFIRM_PATTERNS`（4パターンのみ）と `BashTool.execute()` 内の基本チェックに限定されている
- ユーザーが独自の拒否パターンやカスタムスクリプトで柔軟に制御する手段がない

## やること

### 1. HookRunner クラス追加（~60行）
- `~/.config/vibe-local/hooks.json` から設定を読み込む
- PreToolUse フックをサポート
- 外部スクリプトにJSON（`{"tool_name": "Bash", "tool_input": {"command": "..."}}`）をstdin経由で渡す
- Exit code判定: 0=許可, 2=ブロック（stderr をエラーメッセージとして表示）

### 2. hooks.json フォーマット（Claude Code互換）
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "~/.claude/scripts/deny-check.sh"
          }
        ]
      }
    ]
  }
}
```

### 3. PermissionMgr 統合（~15行）
- `PermissionMgr.__init__()` で `HookRunner` をインスタンス化
- `PermissionMgr.check()` の最初（session deny チェック直後）で `HookRunner.run_pre_tool_use()` を呼び出す
- フックがブロックした場合は即座に `False` を返す

### 4. Config 更新（~5行）
- `hooks_file` プロパティを追加（`~/.config/vibe-local/hooks.json`）

### 5. テスト追加（~50行）
- HookRunner の単体テスト
  - hooks.json が存在しない場合 → 何もせず許可
  - matcher がマッチしない場合 → 許可
  - スクリプトが exit 0 → 許可
  - スクリプトが exit 2 → ブロック
  - スクリプトが存在しない場合 → 許可（フェイルオープン）

## 実行方式

| 項目 | 値 |
|------|-----|
| 規模 | 小規模 |
| モード | Code |
| スキル | workflow-feature-implementation |
| エージェント | メイン |
| 並列実行 | なし |

## 変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `vibe-coder.py` | HookRunner クラス追加（PermissionMgr の直前）、PermissionMgr への統合、Config にhooks_fileパス追加 |
| `tests/test_providers.py` | HookRunner 単体テスト追加 |

## 変更しないもの
- BashTool.execute() 内の既存セキュリティチェック（3層防御はそのまま維持）
- PermissionMgr._ALWAYS_CONFIRM_PATTERNS（既存の危険パターン検出は変更なし）
- Claude Code側のhooks設定ファイル（.claude/settings.json, deny-check.sh）

## 挿入ポイント詳細

```
PermissionMgr.check() フロー:

  session deny チェック
  ↓
  ★ HookRunner.run_pre_tool_use(tool_name, params) ← ここに挿入
  ↓
  yes_mode + 危険パターンチェック
  ↓
  SAFE_TOOLS チェック
  ↓
  persistent rules チェック
  ↓
  session allows チェック
  ↓
  TUI 確認
```

## テスト・検証
1. `python -m pytest tests/test_providers.py -v` で全テスト通過
2. `~/.config/vibe-local/hooks.json` にClaude Codeと同じhooks設定を記述し、deny-check.shを流用して動作確認

---
作成日: 2026-03-05
