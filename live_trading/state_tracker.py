"""
live_trading/state_tracker.py
─────────────────────────────────────────────────────────────────────────────
포지션·사이클·주문 상태를 sqlite에 영속화.
봇이 재시작돼도 상태 복원 가능.
"""

from __future__ import annotations
import sqlite3
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path


DB_PATH = Path(__file__).parent / "state.db"


@dataclass
class BotState:
    # 자본
    cash: float = 0.0
    holdings_btc: float = 0.0
    initial_cash: float = 0.0

    # 사이클
    in_cycle: bool = False
    cycle_start_cash: float = 0.0
    avg_price: float = 0.0
    cycle_budget_remaining: float = 0.0

    # 통계
    completed_cycles: int = 0
    total_pnl: float = 0.0
    peak_equity: float = 0.0

    # 열린 주문 ID 목록 (json 직렬화)
    open_order_ids: list = field(default_factory=list)

    updated_at: str = ""

    def equity(self, price: float) -> float:
        return self.cash + self.holdings_btc * price

    def max_drawdown(self, price: float) -> float:
        eq = self.equity(price)
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - eq) / self.peak_equity


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bot_state (
            id      INTEGER PRIMARY KEY CHECK (id = 1),
            payload TEXT NOT NULL
        )
    """)
    conn.commit()


def load_state(initial_cash: float) -> BotState:
    with sqlite3.connect(DB_PATH) as conn:
        _init_db(conn)
        row = conn.execute("SELECT payload FROM bot_state WHERE id=1").fetchone()
        if row:
            d = json.loads(row[0])
            return BotState(**d)
        # 최초 실행 — 초기 상태 생성
        state = BotState(
            cash=initial_cash,
            initial_cash=initial_cash,
            peak_equity=initial_cash,
        )
        save_state(state)
        return state


def save_state(state: BotState) -> None:
    state.updated_at = datetime.now(timezone.utc).isoformat()
    payload = json.dumps(asdict(state))
    with sqlite3.connect(DB_PATH) as conn:
        _init_db(conn)
        conn.execute(
            "INSERT OR REPLACE INTO bot_state (id, payload) VALUES (1, ?)",
            (payload,),
        )
        conn.commit()


def reset_state(initial_cash: float) -> BotState:
    """DB 초기화 후 새 상태 반환 (긴급 리셋용)."""
    if DB_PATH.exists():
        DB_PATH.unlink()
    return load_state(initial_cash)
