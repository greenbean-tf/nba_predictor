"""
portfolio.py
------------
半場結束後：半場模型預測 + Polymarket 賠率 + 動態 Kelly 下注建議。

使用方式：
  # 手動測試（輸入半場比分）
  python src/live/portfolio.py --game "Pistons @ Wizards" --h1 63-55

  # 自動模式（live_scraper 偵測到半場結束自動觸發）
  python src/live/portfolio.py --live

  # 查看今日資金狀況
  python src/live/portfolio.py --status --capital 10000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse, logging, pickle
import pandas as pd
import numpy as np
from scipy.stats import norm
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

FEAT_DIR  = Path(__file__).parent.parent.parent / CFG["data"]["features_dir"]
MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "saved"

PRED_STD = 9.0      # 半場後預測標準差
MIN_EV   = 0.02
MIN_EDGE = 0.03

TEAM_NAME_MAP = {
    "Cavaliers": "CLE", "Bulls": "CHI", "Pistons": "DET", "Wizards": "WAS",
    "Clippers": "LAC",  "Pelicans": "NOP", "Lakers": "LAL", "Heat": "MIA",
    "Bucks": "MIL",     "Jazz": "UTA",  "Magic": "ORL",  "Hornets": "CHA",
    "76ers": "PHI",     "Kings": "SAC", "Suns": "PHX",   "Spurs": "SAS",
    "Warriors": "GSW",  "Nets": "BKN",  "Celtics": "BOS","Grizzlies": "MEM",
    "Hawks": "ATL",     "Mavericks": "DAL", "Nuggets": "DEN", "Thunder": "OKC",
    "Raptors": "TOR",   "Timberwolves": "MIN", "Trail Blazers": "POR",
    "Rockets": "HOU",   "Knicks": "NYK", "Pacers": "IND",
}
ABBR_TO_FULL = {v: k for k, v in TEAM_NAME_MAP.items()}


# ── 模型 ─────────────────────────────────────────────────────

def load_models():
    xgb = pickle.load(open(MODEL_DIR / "xgb.pkl", "rb"))
    lgb = pickle.load(open(MODEL_DIR / "lgb.pkl", "rb"))
    return xgb, lgb


# ── 特徵建構 ──────────────────────────────────────────────────

def build_features(home_abbr: str, away_abbr: str,
                   history: pd.DataFrame,
                   home_q1: int, away_q1: int,
                   home_q2: int, away_q2: int,
                   feat_cols: list) -> pd.DataFrame:

    feat = {}

    def rolling(team, W, col):
        mask = (history["home_team"] == team) | (history["away_team"] == team)
        sub  = history[mask].tail(W)
        if col == "scored":
            return sub.apply(lambda r: r["home_total"] if r["home_team"] == team
                              else r["away_total"], axis=1)
        elif col == "allowed":
            return sub.apply(lambda r: r["away_total"] if r["home_team"] == team
                              else r["home_total"], axis=1)
        return sub["total_score"]

    for side, team in [("home", home_abbr), ("away", away_abbr)]:
        for W in [5, 10, 20]:
            s = rolling(team, W, "scored")
            a = rolling(team, W, "allowed")
            t = rolling(team, W, "total")
            feat[f"{side}_avg_scored_{W}"]  = s.mean() if len(s) else 110
            feat[f"{side}_avg_allowed_{W}"] = a.mean() if len(a) else 110
            feat[f"{side}_avg_total_{W}"]   = t.mean() if len(t) else 220
        s10 = rolling(team, 10, "scored")
        feat[f"{side}_std_scored_10"] = s10.std() if len(s10) > 1 else 10

    feat["implied_total_10"] = (
        feat["home_avg_scored_10"] + feat["away_avg_scored_10"] +
        feat["home_avg_allowed_10"] + feat["away_avg_allowed_10"]
    ) / 2

    for side, team in [("home", home_abbr), ("away", away_abbr)]:
        for stat in ["pace", "off_rtg", "def_rtg"]:
            col = f"{side}_{stat}"
            if col in history.columns:
                vals = history[history[f"{side}_team"] == team].tail(20)[col].dropna()
                feat[col] = float(vals.mean()) if len(vals) else 100.0
            else:
                feat[col] = 100.0
    feat["pace_matchup"] = (feat["home_pace"] + feat["away_pace"]) / 2

    from datetime import date
    today = pd.Timestamp(date.today())
    for side, team in [("home", home_abbr), ("away", away_abbr)]:
        mask = (history["home_team"] == team) | (history["away_team"] == team)
        last = history[mask]["game_date"].max()
        days = (today - pd.Timestamp(last)).days if pd.notna(last) else 3
        feat[f"{side}_rest_days"] = min(int(days), 14)
        feat[f"{side}_b2b"]       = 1 if days == 1 else 0

    # 盤中特徵（半場）
    q1_total = home_q1 + away_q1
    h1_total = home_q1 + away_q1 + home_q2 + away_q2
    implied  = feat["implied_total_10"]

    feat["home_q1"]       = home_q1
    feat["home_q2"]       = home_q2
    feat["away_q1"]       = away_q1
    feat["away_q2"]       = away_q2
    feat["q1_total"]      = q1_total
    feat["h1_total"]      = h1_total
    feat["q2_total"]      = home_q2 + away_q2
    feat["q1_pace_ratio"] = q1_total / (implied / 4) if implied > 0 else 1
    feat["h1_pace_ratio"] = h1_total / (implied / 2) if implied > 0 else 1
    feat["q1_home_lead"]  = home_q1 - away_q1
    feat["h1_home_lead"]  = (home_q1 + home_q2) - (away_q1 + away_q2)
    feat["q1_vs_avg"]     = q1_total - (implied / 4)

    row = pd.Series(feat)
    for c in feat_cols:
        if c not in row.index:
            row[c] = 0
    return row[feat_cols].to_frame().T.fillna(0)


# ── 預測 ─────────────────────────────────────────────────────

def predict(xgb_model, lgb_model, feat_df: pd.DataFrame) -> float:
    p_xgb = float(xgb_model.predict(feat_df)[0])
    p_lgb = float(lgb_model.predict(feat_df)[0])
    return round((p_xgb + p_lgb) / 2, 1)


# ── 單場計算 ──────────────────────────────────────────────────

def run_halftime(away_name: str, home_name: str,
                 home_q1: int, away_q1: int,
                 home_q2: int, away_q2: int,
                 tracker=None) -> pd.DataFrame:
    """
    半場結束後，對單場比賽計算完整的 EV + Kelly 建議。
    若傳入 tracker，以動態剩餘資金計算 Kelly。
    """
    xgb_model, lgb_model = load_models()
    history   = pd.read_csv(FEAT_DIR / "features.csv", parse_dates=["game_date"])
    feat_cols = list(xgb_model.feature_names_in_)

    home_abbr = TEAM_NAME_MAP.get(home_name, home_name[:3].upper())
    away_abbr = TEAM_NAME_MAP.get(away_name, away_name[:3].upper())

    feat_df = build_features(
        home_abbr, away_abbr, history,
        home_q1, away_q1, home_q2, away_q2, feat_cols
    )
    pred = predict(xgb_model, lgb_model, feat_df)

    h1_score = f"{away_q1+away_q2}-{home_q1+home_q2}"
    log.info(f"[HALFTIME] {away_name} @ {home_name}  H1: {h1_score}")
    log.info(f"  Predicted total: {pred}  (±{PRED_STD})")

    # 抓 Polymarket 賠率
    from src.scraper.polymarket_scraper import fetch_nba_game_events, parse_total_markets
    events  = fetch_nba_game_events()
    markets = parse_total_markets(events)

    h_lower = home_name.lower()
    a_lower = away_name.lower()
    game_markets = [
        m for m in markets
        if (h_lower in m["home_team"].lower() or h_lower in m["away_team"].lower())
        and (a_lower in m["away_team"].lower() or a_lower in m["home_team"].lower())
    ]

    if not game_markets:
        log.warning(f"No Polymarket markets for {away_name} @ {home_name}")
        return pd.DataFrame()

    # 決定 Kelly 基礎資金
    capital = tracker.available if tracker else None

    rows = []
    for m in game_markets:
        line        = m["line"]
        mkt_over    = m["over_prob"]
        mkt_under   = m["under_prob"]
        model_over  = float(1 - norm.cdf(line, loc=pred, scale=PRED_STD))
        model_under = 1 - model_over
        edge_over   = model_over  - mkt_over
        edge_under  = model_under - mkt_under
        ev_over     = (model_over  / mkt_over)  - 1 if mkt_over  > 0 else -1
        ev_under    = (model_under / mkt_under) - 1 if mkt_under > 0 else -1

        # Kelly 金額
        def kelly_amt(model_p, market_p):
            if tracker:
                return tracker.kelly_amount(model_p, market_p)
            # 無 tracker 時回傳比例
            if market_p <= 0 or market_p >= 1:
                return 0
            f = max(0, (model_p - market_p) / (1 - market_p) * 0.25)
            return round(f * 100, 2)   # 回傳 % 數

        rows.append({
            "game":          f"{away_name} @ {home_name}",
            "line":          line,
            "pred":          pred,
            "diff":          round(pred - line, 1),
            "model_over":    round(model_over,  3),
            "mkt_over":      round(mkt_over,    3),
            "edge_over":     round(edge_over,   3),
            "ev_over":       round(ev_over,     4),
            "kelly_over":    kelly_amt(model_over,  mkt_over),
            "model_under":   round(model_under, 3),
            "mkt_under":     round(mkt_under,   3),
            "edge_under":    round(edge_under,  3),
            "ev_under":      round(ev_under,    4),
            "kelly_under":   kelly_amt(model_under, mkt_under),
        })

    df = pd.DataFrame(rows).drop_duplicates(subset=["line"])
    print_result(df, away_name, home_name, pred, capital=capital)
    return df


# ── 印出結果 ──────────────────────────────────────────────────

def print_result(df: pd.DataFrame, away: str, home: str,
                 pred: float, capital: float = None):

    use_dollar = capital is not None
    kelly_label = "金額($)" if use_dollar else "Kelly%"

    print(f"\n{'═'*72}")
    print(f"  {away} @ {home}  |  預測總分：{pred}")
    if capital:
        print(f"  當前可用資金：${capital:,.0f}")
    print(f"{'═'*72}")
    print(f"  {'線':>6}  {'方向':<6}  {'模型':>7}  {'市場':>7}  {'Edge':>7}  "
          f"{'EV':>7}  {kelly_label:>10}")
    print(f"  {'─'*66}")

    bets = []
    for _, row in df.iterrows():
        for side, col in [("OVER","over"), ("UNDER","under")]:
            model_p = row[f"model_{col}"]
            mkt_p   = row[f"mkt_{col}"]
            edge    = row[f"edge_{col}"]
            ev      = row[f"ev_{col}"]
            kelly   = row[f"kelly_{col}"]
            has_ev  = ev >= MIN_EV and edge >= MIN_EDGE
            flag    = " ◀" if has_ev else ""
            k_str   = f"${kelly:,.0f}" if use_dollar else f"{kelly:.1f}%"
            print(f"  {row['line']:>6.1f}  {side:<6}  {model_p:>6.1%}  {mkt_p:>6.1%}  "
                  f"{edge:>+6.1%}  {ev:>+6.1%}  {k_str:>10}{flag}")
            if has_ev:
                bets.append((row["line"], side, edge, ev, kelly))

    print(f"{'═'*72}")
    if bets:
        print(f"\n  建議下注：")
        for line, side, edge, ev, k in sorted(bets, key=lambda x: -x[3]):
            k_str = f"${k:,.0f}" if use_dollar else f"{k:.1f}%"
            print(f"    {line} {side:<6}  edge={edge:+.1%}  EV={ev:+.1%}  → {k_str}")
    else:
        print(f"\n  今場無符合條件的下注機會")
    print()


# ── 進入點 ────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game",    type=str, help='"Pistons @ Wizards"')
    parser.add_argument("--h1",      type=str, help='半場比分 "away-home" e.g. "63-55"')
    parser.add_argument("--q1",      type=str, help='Q1 比分 "away-home" e.g. "30-28"')
    parser.add_argument("--capital", type=float, default=None,
                        help="可用資金（啟用動態 Kelly）")
    parser.add_argument("--status",  action="store_true")
    parser.add_argument("--live",    action="store_true")
    args = parser.parse_args()

    # 動態 Kelly tracker（可選）
    tracker = None
    if args.capital:
        from src.live.capital_tracker import CapitalTracker
        tracker = CapitalTracker(total_capital=args.capital)

    if args.status:
        if tracker:
            tracker.print_status()
        else:
            print("請加上 --capital 金額")

    elif args.live:
        from src.scraper.live_scraper import poll
        from src.live.capital_tracker import evaluate_and_bet, auto_settle_finished_games
        import threading, time as _time

        # 每天 00:00 後自動結算（在背景執行）
        def auto_settle_loop():
            from datetime import datetime
            last_settle_date = None
            while True:
                now = datetime.now()
                # 凌晨 00:30 執行結算
                if now.hour == 0 and now.minute >= 30:
                    today = now.date()
                    if last_settle_date != today:
                        log.info("Auto-settling finished games...")
                        auto_settle_finished_games(tracker)
                        last_settle_date = today
                _time.sleep(300)  # 每 5 分鐘檢查一次

        if tracker:
            t = threading.Thread(target=auto_settle_loop, daemon=True)
            t.start()
            log.info("Auto-settle thread started (runs at 00:30)")

        def on_trigger(game: dict, quarter: int):
            if quarter != 2:   # 只在半場觸發
                log.info(f"Q1 ended for {game['away_team']} @ {game['home_team']}, skipping (halftime only)")
                return
            away = ABBR_TO_FULL.get(game["away_team"], game["away_team"])
            home = ABBR_TO_FULL.get(game["home_team"], game["home_team"])

            if tracker:
                tracker.print_status()

            df = run_halftime(
                away, home,
                game["home_q1"], game["away_q1"],
                game["home_q2"], game["away_q2"],
                tracker=tracker
            )
            if not df.empty and tracker:
                placed = evaluate_and_bet(tracker, df, quarter=2, auto_bet=True)
                if placed:
                    log.info(f"  Placed {len(placed)} bets automatically")
                    tracker.print_status()

        log.info("Live mode：等待半場結束（自動下注 + 自動結算）...")
        poll(on_trigger=on_trigger)

    elif args.game:
        parts = args.game.split(" @ ")
        away_name = parts[0].strip()
        home_name = parts[1].strip() if len(parts) == 2 else ""

        if args.h1:
            sc = [int(x) for x in args.h1.split("-")]
            # h1 格式: away_h1-home_h1，各節平均分配
            away_h1, home_h1 = sc[0], sc[1]
            # 近似各節（若只給半場合計）
            run_halftime(
                away_name, home_name,
                home_q1=home_h1//2, away_q1=away_h1//2,
                home_q2=home_h1-home_h1//2, away_q2=away_h1-away_h1//2,
                tracker=tracker
            )
        elif args.q1:
            sc = [int(x) for x in args.q1.split("-")]
            away_q1, home_q1 = sc[0], sc[1]
            run_halftime(
                away_name, home_name,
                home_q1=home_q1, away_q1=away_q1,
                home_q2=0, away_q2=0,
                tracker=tracker
            )
    else:
        parser.print_help()
