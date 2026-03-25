"""
predictor.py
------------
即時預測主程式，把所有模組串接起來：

  live_scraper  → 偵測節次結束
       ↓
  manual_features → 建立即時特徵
       ↓
  ensemble → 三模型加權預測
       ↓
  output → 14 個 threshold over/under 信號

使用方式：
  python src/live/predictor.py
  python src/live/predictor.py --interval 30
  python src/live/predictor.py --demo       # 用歷史資料測試，不需要真實比賽
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

FEAT_DIR      = Path(__file__).parent.parent.parent / CFG["data"]["features_dir"]
LOG_DIR       = Path(__file__).parent.parent.parent / "logs"
POLL_INTERVAL = CFG["live"]["poll_interval_sec"]
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ── 歷史資料快取 ──────────────────────────────────────────────

class HistoryCache:
    """
    把 features.csv 載入記憶體，提供即時查詢。
    只在啟動時載入一次，節省 I/O。
    """

    def __init__(self):
        path = FEAT_DIR / "features.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run manual_features.py first."
            )
        self._df = pd.read_csv(path, parse_dates=["game_date"])
        log.info(f"History cache loaded: {len(self._df)} games")

    def get_team_history(self, home: str, away: str,
                         n: int = 40) -> pd.DataFrame:
        """取兩隊都相關的最近 n 場歷史。"""
        mask = (
            (self._df["home_team"] == home) | (self._df["away_team"] == home) |
            (self._df["home_team"] == away) | (self._df["away_team"] == away)
        )
        return self._df[mask].tail(n).copy()

    def get_season_avg_total(self) -> float:
        """用近 500 場平均總得分當 fallback 莊家線。"""
        return float(self._df.tail(500)["total_score"].mean())


# ── 莊家線取得 ────────────────────────────────────────────────

def fetch_odds_line(home: str, away: str) -> float | None:
    """
    從 The Odds API 取得莊家開盤 total line。
    需要在環境變數設定 ODDS_API_KEY。
    若無 key 則回傳 None，由外部 fallback。
    """
    import os
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        return None

    try:
        import requests
        resp = requests.get(
            CFG["live"]["odds_api_url"],
            params={
                "apiKey": api_key,
                "regions": "us",
                "markets": "totals",
                "oddsFormat": "american",
            },
            timeout=8,
        )
        for game in resp.json():
            teams = [game.get("home_team", ""), game.get("away_team", "")]
            home_match = any(home in t or t in home for t in teams)
            away_match = any(away in t or t in away for t in teams)
            if home_match and away_match:
                for bookmaker in game.get("bookmakers", []):
                    for market in bookmaker.get("markets", []):
                        if market["key"] == "totals":
                            over = next(
                                (o for o in market["outcomes"] if o["name"] == "Over"),
                                None,
                            )
                            if over:
                                return float(over["point"])
    except Exception as e:
        log.warning(f"Odds API error: {e}")

    return None


# ── 單場完整預測流程 ──────────────────────────────────────────

def run_prediction(game: dict, quarter: int,
                   history: HistoryCache) -> dict:
    """
    Q1 或半場結束後執行完整預測。

    Args:
        game    : live_scraper 回傳的比賽 dict
        quarter : 1 (Q1 結束) 或 2 (半場結束)
        history : HistoryCache 實例

    Returns:
        預測結果 dict
    """
    from src.models.ensemble import predict_live
    from src.models.output   import pred_to_signals, format_output

    home = game["home_team"]
    away = game["away_team"]

    log.info(f"Running prediction: {away} @ {home}  "
             f"({'Q1' if quarter == 1 else 'HALFTIME'})")

    team_history = history.get_team_history(home, away)

    # 莊家線
    line = fetch_odds_line(home, away)
    if line is None:
        line = history.get_season_avg_total()
        log.info(f"  Odds API unavailable, using avg: {line:.1f}")
    else:
        log.info(f"  Odds line: {line}")

    # Ensemble 預測
    result = predict_live(game, team_history, implied_total=line)

    # 轉成 14 個 threshold 信號
    signals = pred_to_signals(
        pred_total=result["pred_total"],
        pred_std=result["pred_std"],
    )

    # 印出報告
    score_str = (
        f"H1: {game['away_total']}-{game['home_total']}"
        if quarter == 2
        else f"Q1: {game['away_q1']}-{game['home_q1']}"
    )
    print(format_output(
        pred_total=result["pred_total"],
        pred_std=result["pred_std"],
        game_info={
            "home":    home,
            "away":    away,
            "quarter": "HALFTIME" if quarter == 2 else "Q1",
            "score":   score_str,
        },
    ))

    payload = {
        "game_id":   game["game_id"],
        "home":      home,
        "away":      away,
        "quarter":   quarter,
        "line":      line,
        "q1_total":  game["q1_total"],
        "h1_total":  game["h1_total"],
        "result":    result,
        "n_high_conf_bets": int(signals["recommend"].sum()),
        "timestamp": datetime.now().isoformat(),
    }

    _log_prediction(payload)
    return payload


def _log_prediction(payload: dict):
    """追加寫入 JSONL 預測日誌。"""
    with open(LOG_DIR / "live_predictions.jsonl", "a") as f:
        f.write(json.dumps(payload, default=str) + "\n")


# ── Demo 模式 ─────────────────────────────────────────────────

def run_demo(history: HistoryCache):
    """
    用 features.csv 的最後一場歷史比賽當作「即時」輸入，
    不需要真實比賽，適合驗證整個 pipeline。
    """
    log.info("Demo mode: using last historical game as live input")

    df   = pd.read_csv(FEAT_DIR / "features.csv", parse_dates=["game_date"])
    last = df.iloc[-1]

    fake_game = {
        "game_id":    str(last["game_id"]),
        "status":     "in",
        "status_text":"End of 2nd Quarter",
        "quarter":    2,
        "clock":      "0:00",
        "home_team":  last["home_team"],
        "away_team":  last["away_team"],
        "home_q1":    int(last["home_q1"]),
        "home_q2":    int(last["home_q2"]),
        "home_q3":    0, "home_q4": 0,
        "away_q1":    int(last["away_q1"]),
        "away_q2":    int(last["away_q2"]),
        "away_q3":    0, "away_q4": 0,
        "home_total": int(last["home_q1"]) + int(last["home_q2"]),
        "away_total": int(last["away_q1"]) + int(last["away_q2"]),
        "q1_total":   int(last["q1_total"]),
        "h1_total":   int(last["h1_total"]),
    }

    log.info(f"Game:          {fake_game['away_team']} @ {fake_game['home_team']}")
    log.info(f"Halftime:      {fake_game['away_total']}-{fake_game['home_total']}")
    log.info(f"Actual total:  {last['total_score']}")

    run_prediction(fake_game, quarter=2, history=history)


# ── 進入點 ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NBA live total predictor")
    parser.add_argument("--interval", type=int, default=POLL_INTERVAL,
                        help="Polling interval in seconds")
    parser.add_argument("--demo", action="store_true",
                        help="Test pipeline with last historical game")
    args = parser.parse_args()

    history = HistoryCache()

    if args.demo:
        run_demo(history)
        return

    # 正式模式：注入 callback 到 live_scraper
    from src.scraper.live_scraper import poll

    def on_trigger(game: dict, quarter: int):
        try:
            run_prediction(game, quarter, history)
        except Exception as e:
            log.error(f"Prediction error [{game['game_id']}]: {e}", exc_info=True)

    log.info("Live predictor is running. Waiting for games to start...")
    poll(interval=args.interval, on_trigger=on_trigger)


if __name__ == "__main__":
    main()
