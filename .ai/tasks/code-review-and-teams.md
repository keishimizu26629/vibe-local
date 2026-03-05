# コードレビュー修正 + Teams システム実装 - タスク分解

計画: `.ai/plan/code-review-and-teams.md`

## タスク一覧

| # | タスク | 担当 | 依存 | 並列可 | 受け入れ条件 |
|---|--------|------|------|--------|-------------|
| 1 | ブランチ作成 (`feature/code-review-and-teams`) | メイン | - | - | main から新ブランチ作成 |
| 2 | symlink 親ディレクトリ検証 + セッションID 強化 | security-fixer | #1 | Yes (#3と) | WriteTool/EditTool で親 symlink 拒否、セッションID が secrets.token_urlsafe 使用 |
| 3 | TeamManager + MessageBus 実装 | teams-builder | #1 | Yes (#2と) | チーム CRUD + DM/broadcast/shutdown 動作、テスト通過 |
| 4 | except Exception 具体化（重要領域 ~25箇所） | security-fixer | #2 | Yes (#5と) | PermissionMgr/HookRunner/Agent.run()/ファイルツール/Session の except 具体化、テスト通過 |
| 5 | TeamAgent + ツール群実装 | teams-builder | #3 | Yes (#4と) | TeamAgent ライフサイクル動作、TeamCreate/Delete/SendMessage ツール動作、テスト通過 |
| 6 | 統合（main() + _task_store 永続化） | メイン | #4, #5 | No | main() にTeamManager初期化、タスクファイル永続化、メッセージ配信チェック統合 |
| 7 | 全テスト実行 + 回帰確認 | test-runner | #6 | Yes (#8と) | 全テスト通過、既存70テスト回帰なし |
| 8 | 最終コードレビュー | code-reviewer | #6 | Yes (#7と) | セキュリティ・品質の確認 |
| 9 | コミット + PR作成 | メイン | #7, #8 | No | PR作成、レビューポイント記載 |

## 実行順序

```
#1 (直列: ブランチ作成)
  → #2 + #3 (並列: セキュリティ修正 + Teams コア)
  → #4 + #5 (並列: except 具体化 + Teams エージェント/ツール)
  → #6 (直列: 統合)
  → #7 + #8 (並列: テスト + レビュー)
  → #9 (直列: コミット + PR)
```

## 委譲マッピング

| タスク | エージェント | subagent_type | 理由 |
|--------|------------|---------------|------|
| #2 | security-fixer | general-purpose | symlink/セッションID のセキュリティ修正 + テスト作成 |
| #3 | teams-builder | general-purpose | 新規クラス実装（TeamManager, MessageBus）+ テスト |
| #4 | security-fixer | general-purpose | except Exception 具体化（広範囲の変更） |
| #5 | teams-builder | general-purpose | TeamAgent/ツール実装（B-1/B-2 に依存） |
| #7 | test-runner | test-runner | テスト実行・結果分析の専門 |
| #8 | code-reviewer | code-reviewer | コード品質・セキュリティの専門レビュー |

---
作成日: 2026-03-05
