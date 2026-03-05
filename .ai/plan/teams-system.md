# Teams システム実装

## 概要
vibe-coderにTeams機能を追加し、複数エージェントがタスクリストを共有しながらメッセージングで協調作業する仕組みを実現する。既存のSubAgent/ParallelAgents/Taskツール基盤を拡張する。

## 現状の問題
- SubAgent は結果を返すだけの一方通行で、実行中の双方向通信ができない
- ParallelAgents は並列実行できるが、タスク割り当てや協調の概念がない
- タスクはメモリ内のみ（`_task_store`）で、チーム横断の共有ができない

## 既存の土台

| コンポーネント | 状態 | Teams での活用 |
|---------------|------|---------------|
| `SubAgentTool` (行4610) | 実装済み | TeamAgent の基盤（独立エージェントループ） |
| `ParallelAgentTool` (行5438) | 実装済み | スレッド管理パターンの参考 |
| `_task_store` + Taskツール群 (行4317) | 実装済み（メモリ内） | チーム共有タスクリストに拡張 |
| `_print_lock` / `_task_store_lock` | 実装済み | スレッド安全性の基盤 |
| `Agent.run()` (行7431) | 実装済み | メッセージ割り込みポイント |

## やること

### Phase 1: TeamManager（チーム管理）~150行

```
~/.config/vibe-local/teams/{team-name}/config.json
```

```json
{
  "team_name": "my-project",
  "description": "Working on feature X",
  "members": [
    {"name": "researcher", "agentId": "abc-123", "agentType": "read-only"},
    {"name": "coder", "agentId": "def-456", "agentType": "full"}
  ]
}
```

- TeamManager クラス: チーム作成・メンバー登録・設定ファイル読み書き
- チームごとのタスクリスト保存: `~/.config/vibe-local/teams/{team-name}/tasks.json`
- メンバーのライフサイクル管理（spawn/idle/shutdown）

### Phase 2: MessageBus（メッセージング）~200行

```python
class MessageBus:
    """スレッドセーフなエージェント間メッセージキュー"""
    _queues: dict[str, queue.Queue]  # agent_name -> message queue

    def send(self, sender, recipient, content, msg_type="message")
    def broadcast(self, sender, content)
    def receive(self, agent_name, block=False, timeout=None) -> Message | None
    def shutdown_request(self, sender, recipient)
    def shutdown_response(self, agent_name, request_id, approve)
```

メッセージ型:
- `message`: DM（1対1）
- `broadcast`: 全員向け
- `shutdown_request` / `shutdown_response`: シャットダウンプロトコル

### Phase 3: TeamAgent（チームメンバーエージェント）~200行

SubAgentTool を拡張し、以下を追加:
- **idle/wake-up ループ**: タスク完了後にidle状態へ遷移、メッセージ受信で復帰
- **メッセージ受信割り込み**: Agent.run() のイテレーション冒頭でMessageBusをポーリング
- **タスク自動取得**: idle時にTaskListからunblocked/unownedタスクを自動claim
- **シャットダウンハンドリング**: shutdown_request 受信時の応答

```
TeamAgent ライフサイクル:
  spawn → [タスク実行 → idle → メッセージ受信 → タスク実行]* → shutdown
```

### Phase 4: ツール群 ~250行

| ツール | 説明 |
|--------|------|
| `TeamCreateTool` | チーム作成（config.json + tasks.json 生成） |
| `TeamDeleteTool` | チーム削除（全メンバーshutdown確認後） |
| `SendMessageTool` | DM/broadcast/shutdown_request/response |

### Phase 5: 統合 ~100行

- `main()` でのTeamManager初期化
- メインループでのメッセージ配信チェック（tui.get_multiline_input 待機中に割り込み）
- TaskツールのTeam対応（`_task_store` をファイル永続化に切替、チーム横断共有）

## アーキテクチャ

```
┌─────────────────────────────────────────────────┐
│                   Main Agent                     │
│  (Agent.run() + TUI + メインループ 8304-8800)     │
│                                                  │
│  TeamManager ─── config.json (チーム定義)          │
│  MessageBus  ─── queue.Queue (スレッド間通信)      │
│  TaskStore   ─── tasks.json (永続化)              │
└──────┬──────────────┬──────────────┬─────────────┘
       │              │              │
       ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│TeamAgent │  │TeamAgent │  │TeamAgent │
│"researcher"│ │"coder"   │ │"tester"  │
│(Thread)  │  │(Thread)  │  │(Thread)  │
│          │  │          │  │          │
│ read-only│  │ full     │  │ read-only│
│ tools    │  │ tools    │  │ + Bash   │
└──────────┘  └──────────┘  └──────────┘
       ↕              ↕              ↕
       └──── MessageBus (queue.Queue) ────┘
```

## 実行方式

| 項目 | 値 |
|------|-----|
| 規模 | 大規模 |
| モード | Code |
| スキル | workflow-feature-implementation, test-driven-development |
| エージェント | メイン + sub-agent（テスト並列） |
| 並列実行 | あり（Phase 1-2 は直列、Phase 3-4 は並列可） |

## 変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `vibe-coder.py` | TeamManager, MessageBus, TeamAgent, ツール群追加。_task_store の永続化対応。main() への統合 |
| `tests/test_providers.py` | TeamManager, MessageBus, SendMessage ツールの単体テスト |

## 変更しないもの
- 既存の SubAgentTool / ParallelAgentTool（後方互換維持）
- 既存の TaskCreate/List/Get/Update ツールのインターフェース
- LLMクライアント層（OpenRouter/VertexAI/Ollama）

## 難所・リスク

| リスク | 対策 |
|--------|------|
| 複数エージェントの同時LLM API呼び出しによるレート制限 | セマフォで同時リクエスト数を制限（max 3） |
| _task_store のファイル永続化時の競合 | fcntl.flock によるファイルロック |
| メッセージ配信タイミング（ターン中 vs idle中） | idle時はqueue.get(block=True)、ターン中はイテレーション冒頭でnon-blockingポーリング |
| エージェントの暴走（無限ループ） | TeamAgent にも max_turns ハード上限（HARD_MAX_TURNS=50） |

## テスト・検証
1. `python3 -m pytest tests/test_providers.py -v` で全テスト通過
2. TeamManager: チーム作成・メンバー追加・削除のCRUD
3. MessageBus: DM送受信、broadcast、shutdown プロトコル
4. TeamAgent: spawn → タスク実行 → idle → メッセージ復帰 → shutdown
5. 統合テスト: 2エージェントのチームで簡単なタスク分担実行

## 実行順序

```
Phase 1: TeamManager（直列）
  → Phase 2: MessageBus（直列）
  → Phase 3 + Phase 4（並列可）
  → Phase 5: 統合（直列）
```

---
作成日: 2026-03-05
