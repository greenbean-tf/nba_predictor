"""
live_scraper.py
---------------
利用 ESPN 隱藏 API 即時抓取比賽比分，
每 60 秒 poll 一次，偵測節次結束後觸發模型預測。

使用方式：
  python src/scraper/live_scraper.py
  python src/scraper/live_scraper.py --interval 30
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import requests
import yaml

# ── 設定 logging ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── 載入設定 ──────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

ESPN_URL      = CFG["live"]["espn_scoreboard_url"]
POLL_INTERVAL = CFG["live"]["poll_interval_sec"]
TRIGGER_QTR   = set(CFG["live"]["trigger_quarters"])   # {1, 2}


# ── ESPN API 解析 ─────────────────────────────────────────────

def fetch_scoreboard() -> list[dict]:
    """
    抓取今天所有 NBA 比賽的即時狀態。

    回傳 list，每個元素代表一場比賽：
    {
      "game_id":     "401585589",
      "status":      "in",           # pre / in / post
      "status_text": "End of 2nd",
      "quarter":     2,              # 當前節次
      "clock":       "0:00",        # 剩餘時間
      "home_team":   "LAL",
      "away_team":   "GSW",
      "home_q1": 32, "home_q2": 28, ...
      "away_q1": 27, "away_q2": 31, ...
      "home_total":  60,
      "away_total":  58,
      "q1_total":    59,
      "h1_total":    118,
    }
    """
    try:
        resp = requests.get(ESPN_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.error(f"ESPN API error: {e}")
        return []

    games = []
    for event in data.get("events", []):
        competition = event["competitions"][0]
        status      = event["status"]

        # 比賽狀態
        state        = status["type"]["state"]         # pre / in / post
        status_text  = status["type"]["description"]   # e.g. "End of 2nd Quarter"
        quarter      = status.get("period", 0)
        clock        = status["displayClock"]           # e.g. "0:00"

        # 主客場資訊
        home = next(c for c in competition["competitors"] if c["homeAway"] == "home")
        away = next(c for c in competition["competitors"] if c["homeAway"] == "away")

        home_abbr = home["team"]["abbreviation"]
        away_abbr = away["team"]["abbreviation"]

        # 各節比分（ESPN 提供 linescores）
        home_qs = _parse_linescores(home.get("linescores", []))
        away_qs = _parse_linescores(away.get("linescores", []))

        home_total = int(home.get("score", 0) or 0)
        away_total = int(away.get("score", 0) or 0)

        q1_total = home_qs.get(1, 0) + away_qs.get(1, 0)
        h1_total = (home_qs.get(1, 0) + home_qs.get(2, 0) +
                    away_qs.get(1, 0) + away_qs.get(2, 0))

        games.append({
            "game_id":     event["id"],
            "status":      state,
            "status_text": status_text,
            "quarter":     quarter,
            "clock":       clock,
            "home_team":   home_abbr,
            "away_team":   away_abbr,
            # 各節
            "home_q1": home_qs.get(1, 0), "home_q2": home_qs.get(2, 0),
            "home_q3": home_qs.get(3, 0), "home_q4": home_qs.get(4, 0),
            "away_q1": away_qs.get(1, 0), "away_q2": away_qs.get(2, 0),
            "away_q3": away_qs.get(3, 0), "away_q4": away_qs.get(4, 0),
            # 合計
            "home_total": home_total,
            "away_total": away_total,
            "q1_total":   q1_total,
            "h1_total":   h1_total,
        })

    return games


def _parse_linescores(linescores: list) -> dict[int, int]:
    """
    ESPN linescores → {1: 32, 2: 28, 3: 30, 4: 29}
    """
    result = {}
    for i, ls in enumerate(linescores, start=1):
        try:
            result[i] = int(ls.get("value", 0) or 0)
        except (ValueError, TypeError):
            result[i] = 0
    return result


# ── 節次結束偵測 ──────────────────────────────────────────────

class QuarterTracker:
    """
    追蹤每場比賽的節次狀態，偵測「節次結束」事件。

    邏輯：
      - clock == "0:00" 且 quarter 在 TRIGGER_QTR 內
      - 同一場比賽同一節只觸發一次（避免重複預測）
    """

    def __init__(self):
        # {game_id: set of quarters already triggered}
        self._triggered: dict[str, set] = {}

    def check(self, game: dict) -> int | None:
        """
        回傳應該觸發預測的節次（1 或 2），否則回傳 None。
        """
        gid     = game["game_id"]
        quarter = game["quarter"]
        clock   = game["clock"]
        status  = game["status"]

        if status != "in":
            return None
        if quarter not in TRIGGER_QTR:
            return None

        # 節次結束條件：
        #   clock 為 0:00 / 0:00.0 / 0.0
        #   或 status_text 含 "end" / "halftime"
        status_lower = game["status_text"].lower()
        is_end = (
            clock in ("0:00", "0:00.0", "0.0") or
            "end" in status_lower or
            "halftime" in status_lower or
            "half time" in status_lower
        )
        if not is_end:
            return None

        # 避免重複觸發
        triggered = self._triggered.setdefault(gid, set())
        if quarter in triggered:
            return None

        triggered.add(quarter)
        return quarter

    def reset_game(self, game_id: str):
        """比賽結束後清除紀錄。"""
        self._triggered.pop(game_id, None)


# ── 主要 polling 迴圈 ─────────────────────────────────────────

def poll(interval: int = POLL_INTERVAL, on_trigger=None):
    """
    每隔 interval 秒抓一次比分，偵測到節次結束時呼叫 on_trigger。

    Args:
        interval   : polling 間隔（秒）
        on_trigger : callback(game, quarter)，由 predictor.py 注入
                     若為 None 則只印 log（debug 用）
    """
    tracker = QuarterTracker()
    log.info(f"Live polling started (interval={interval}s, "
             f"trigger quarters={sorted(TRIGGER_QTR)})")

    while True:
        now = datetime.now().strftime("%H:%M:%S")
        games = fetch_scoreboard()

        if not games:
            log.info(f"[{now}] No games found or API error, retrying...")
        else:
            live = [g for g in games if g["status"] == "in"]
            done = [g for g in games if g["status"] == "post"]
            pre  = [g for g in games if g["status"] == "pre"]

            log.info(f"[{now}] Games: {len(pre)} upcoming / "
                     f"{len(live)} live / {len(done)} finished")

            for game in live:
                _log_game(game)
                triggered_quarter = tracker.check(game)
                if triggered_quarter:
                    log.info(
                        f"  ★ TRIGGER Q{triggered_quarter} END: "
                        f"{game['away_team']} @ {game['home_team']}  "
                        f"score={game['away_total']}-{game['home_total']}"
                    )
                    if on_trigger:
                        on_trigger(game, triggered_quarter)
                    else:
                        _debug_trigger(game, triggered_quarter)

            # 清理已結束比賽的追蹤紀錄
            for game in done:
                tracker.reset_game(game["game_id"])

        time.sleep(interval)


def _log_game(game: dict):
    log.info(
        f"  {game['away_team']:>3} @ {game['home_team']:<3}  "
        f"Q{game['quarter']} {game['clock']:<6}  "
        f"{game['away_total']}-{game['home_total']}  "
        f"({game['status_text']})"
    )


def _debug_trigger(game: dict, quarter: int):
    """on_trigger 未注入時的預設行為（印出觸發資訊）。"""
    label = "Q1" if quarter == 1 else "HALFTIME"
    total_so_far = (game["q1_total"] if quarter == 1 else game["h1_total"])
    log.info(
        f"  → [{label}] q1_total={game['q1_total']}  "
        f"h1_total={game['h1_total']}  "
        f"(would run prediction here)"
    )


# ── 進入點 ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NBA live score poller")
    parser.add_argument("--interval", type=int, default=POLL_INTERVAL,
                        help=f"Polling interval in seconds (default: {POLL_INTERVAL})")
    args = parser.parse_args()

    poll(interval=args.interval)


if __name__ == "__main__":
    main()
