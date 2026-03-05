# コードレビュー修正 + Teams システム実装

## 概要
code-review.md の Phase 1 即時対応（symlink親ディレクトリ検証、except Exception具体化、セッションID強化）と teams-system.md の全5フェーズを、Claude Code の Team 機能で並列実行する。

## 現状の問題
1. **セキュリティ**: WriteTool/EditTool で親ディレクトリの symlink 未検証（H-S02）、セッションID のランダム部分が 6文字のみ（M-S07）
2. **コード品質**: `except Exception` が 93箇所に散在し、エラー隠蔽のリスク（C-E01）
3. **機能不足**: SubAgent は一方通行、タスク共有・メッセージングによる協調作業ができない

## やること

### ワークストリーム A: コードレビュー Phase 1 修正

#### A-1. symlink 親ディレクトリ検証（H-S02）
- 共通ヘルパー `_resolve_safe_path(file_path)` を追加
  - `os.path.realpath()` で全パスコンポーネント解決
  - WriteTool（行3385）、EditTool（行3488）で使用
  - ReadTool（行3127）は既に `os.path.realpath()` 使用済み → 変更不要
- テスト: 親ディレクトリが symlink の場合に拒否されること

#### A-2. セッションID 強化（M-S07）
- Session.__init__（行5923）で `secrets.token_urlsafe(24)` に変更
- タイムスタンプ + ランダム部分の組み合わせは維持
- テスト: セッションID が十分なエントロピーを持つこと

#### A-3. except Exception 具体化（C-E01）— スコープ限定
- 93箇所全てではなく、以下の重要領域に絞る（約25箇所）:
  - PermissionMgr / HookRunner（セキュリティ関連）
  - Agent.run()（コアループ）
  - ファイル操作ツール（Read/Write/Edit）
  - セッション管理（Session）
- 各箇所で適切な具体的例外型（OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired 等）に置換

### ワークストリーム B: Teams システム実装

既存計画 `.ai/plan/teams-system.md` に従い、5フェーズで実装。

#### B-1. TeamManager（~150行）
- `~/.config/vibe-local/teams/{team-name}/config.json` の CRUD
- メンバー登録・削除・ライフサイクル管理

#### B-2. MessageBus（~200行）
- `queue.Queue` ベースのスレッドセーフメッセージング
- DM / broadcast / shutdown プロトコル

#### B-3. TeamAgent（~200行）
- SubAgentTool 拡張、idle/wake-up ループ
- メッセージ受信割り込み、タスク自動取得

#### B-4. ツール群（~250行）
- TeamCreateTool, TeamDeleteTool, SendMessageTool

#### B-5. 統合（~100行）
- main() への TeamManager 初期化
- _task_store のファイル永続化
- メインループへのメッセージ配信チェック統合

## 実行方式

| 項目 | 値 |
|------|-----|
| 規模 | 大規模 |
| モード | Code |
| スキル | workflow-feature-implementation, test-driven-development |
| エージェント | Team: security-fixer, teams-builder, test-runner |
| 並列実行 | あり（ワークストリーム A と B-1/B-2 を並列） |

## 並列実行戦略

単一ファイル（vibe-coder.py）への同時編集の競合を回避するため:

```
Phase 1（並列）:
  security-fixer: A-1 + A-2 + A-3（行3353-3605, 5564-5929 付近）
  teams-builder:  B-1 + B-2（PermissionMgr 直前 = 行5563 付近に新規クラス追加）
  ※ 担当行範囲が重ならないため並列可能

Phase 2（直列、Phase 1 完了後）:
  teams-builder:  B-3 + B-4（Phase 1 の TeamManager/MessageBus に依存）

Phase 3（直列、Phase 2 完了後）:
  メイン:  B-5 統合（main() 修正、_task_store 永続化）

Phase 4（並列）:
  test-runner: 全テスト実行 + 回帰確認
  code-reviewer: 最終レビュー
```

## 変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `vibe-coder.py` | _resolve_safe_path ヘルパー、symlink 検証強化、セッションID 強化、except Exception 具体化（~25箇所）、TeamManager, MessageBus, TeamAgent, TeamCreateTool, TeamDeleteTool, SendMessageTool 追加、_task_store 永続化、main() 統合 |
| `tests/test_providers.py` | symlink 親ディレクトリテスト、セッションID テスト、except 具体化のリグレッションテスト、TeamManager/MessageBus/TeamAgent/SendMessage テスト |

## 変更しないもの
- 既存の SubAgentTool / ParallelAgentTool のインターフェース
- 既存の TaskCreate/List/Get/Update ツールのインターフェース（内部を永続化対応に変更するが外部 API は維持）
- LLMクライアント層（OpenRouter/VertexAI/Ollama）
- BashTool のセキュリティモデル（shell=True は設計妥当）
- code-review.md の Phase 2-4 の項目（今回スコープ外）

## 難所・リスク

| リスク | 対策 |
|--------|------|
| 単一ファイルへの並列編集による競合 | 担当行範囲を明確に分離、Phase 分割で直列化 |
| except Exception 変更による予期せぬ動作変更 | 変更箇所ごとにテスト、段階的に適用 |
| Teams の複数エージェント同時 LLM API 呼び出し | セマフォで同時リクエスト数制限（max 3） |
| _task_store 永続化の競合 | fcntl.flock によるファイルロック |
| メッセージ配信タイミング | idle時 block=True、ターン中 non-blocking ポーリング |

## テスト・検証
1. `python3 -m pytest tests/test_providers.py -v` で全テスト通過
2. symlink 親ディレクトリテスト: `/tmp/symlink_dir/file.txt` パターンで拒否確認
3. セッションID: エントロピー検証（32文字以上のランダム部分）
4. TeamManager: CRUD + ファイル永続化
5. MessageBus: DM/broadcast/shutdown プロトコル
6. TeamAgent: spawn → idle → メッセージ復帰 → shutdown
7. 統合: 2エージェントチームでの簡易タスク分担
8. 既存70テストの回帰なし確認

---
作成日: 2026-03-05
