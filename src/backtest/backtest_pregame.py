"""
backtest_pregame.py
-------------------
用賽前模型 + 真實莊家線做有意義的回測。

流程：
  1. 載入 pregame_test_predictions.csv
  2. 載入 The Odds API 的賽前賠率
  3. 配對（日期 + 隊名）
  4. 計算期望值、Kelly、真實 ROI

若賠率資料不存在，先抓即時賠率（免費方案）
用來驗證框架是否正確。

使用方式：
  python src/backtest/backtest_pregame.py
  python src/backtest/backtest_pregame.py --collect-odds   # 先抓今天的賠率
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import logging
import pandas as pd
import numpy as np
import yaml
from scipy.stats import norm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

RAW_DIR   = Path(__file__).parent.parent.parent / CFG["data"]["raw_dir"]
MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "saved"
LOG_DIR   = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

PRED_STD   = 13.0   # 賽前模型較大的不確定性
KELLY_FRAC = 0.25   # 使用 1/4 Kelly


# ── 機率與賠率工具 ────────────────────────────────────────────

def model_prob_over(pred: float, line: float) -> float:
    return float(1 - norm.cdf(line, loc=pred, scale=PRED_STD))


def implied_prob(american_odds: int) -> float:
    if american_odds > 0:
        return 100 / (american_odds + 100)
    return abs(american_odds) / (abs(american_odds) + 100)


def remove_vig(p_over: float, p_under: float) -> tuple[float, float]:
    total = p_over + p_under
    return p_over / total, p_under / total


def american_to_decimal(odds: int) -> float:
    if odds > 0:
        return (odds / 100) + 1
    return (100 / abs(odds)) + 1


def expected_value(model_p: float, decimal_odds: float) -> float:
    return model_p * (decimal_odds - 1) - (1 - model_p)


def kelly(model_p: float, decimal_odds: float) -> float:
    b = decimal_odds - 1
    k = (model_p * b - (1 - model_p)) / b
    return max(0.0, k * KELLY_FRAC)


# ── 配對賽前賠率 ──────────────────────────────────────────────

def load_pregame_predictions() -> pd.DataFrame:
    path = MODEL_DIR / "pregame_test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python src/models/pregame_model.py"
        )
    df = pd.read_csv(path, parse_dates=["game_date"])
    log.info(f"Loaded {len(df)} pregame predictions")
    return df


def find_odds_file() -> Path | None:
    """找任何可用的賠率 CSV。"""
    candidates = sorted(RAW_DIR.glob("odds_*.csv"))
    if candidates:
        log.info(f"Found odds file: {candidates[-1].name}")
        return candidates[-1]
    return None


def match_with_odds(preds: pd.DataFrame,
                    odds: pd.DataFrame) -> pd.DataFrame:
    """把預測配對對應的莊家線（日期 ±1 天，隊名匹配）。"""

    ABBR_TO_FULL = {
        "ATL": "Atlanta Hawks",       "BOS": "Boston Celtics",
        "BKN": "Brooklyn Nets",       "CHA": "Charlotte Hornets",
        "CHI": "Chicago Bulls",       "CLE": "Cleveland Cavaliers",
        "DAL": "Dallas Mavericks",    "DEN": "Denver Nuggets",
        "DET": "Detroit Pistons",     "GSW": "Golden State Warriors",
        "HOU": "Houston Rockets",     "IND": "Indiana Pacers",
        "LAC": "LA Clippers",         "LAL": "Los Angeles Lakers",
        "MEM": "Memphis Grizzlies",   "MIA": "Miami Heat",
        "MIL": "Milwaukee Bucks",     "MIN": "Minnesota Timberwolves",
        "NOP": "New Orleans Pelicans","NYK": "New York Knicks",
        "OKC": "Oklahoma City Thunder","ORL": "Orlando Magic",
        "PHI": "Philadelphia 76ers",  "PHX": "Phoenix Suns",
        "POR": "Portland Trail Blazers","SAC": "Sacramento Kings",
        "SAS": "San Antonio Spurs",   "TOR": "Toronto Raptors",
        "UTA": "Utah Jazz",           "WAS": "Washington Wizards",
    }
    FULL_TO_ABBR = {v: k for k, v in ABBR_TO_FULL.items()}

    def to_abbr(name: str) -> str:
        if name in FULL_TO_ABBR:
            return FULL_TO_ABBR[name]
        for full, abbr in FULL_TO_ABBR.items():
            words = full.split()
            if any(w in name for w in words[-2:]):
                return abbr
        return name

    odds = odds.copy()
    odds["home_abbr"] = odds["home_team"].apply(to_abbr)
    odds["away_abbr"] = odds["away_team"].apply(to_abbr)
    odds["date"] = pd.to_datetime(odds["commence_time"]).dt.date

    # 每場取最佳 bookmaker
    PREFERRED = ["draftkings", "fanduel", "betmgm", "caesars", "bovada"]

    preds = preds.copy()
    preds["game_date_dt"] = pd.to_datetime(preds["game_date"]).dt.date

    lines, o_prices, u_prices = [], [], []
    for _, row in preds.iterrows():
        gdate = row["game_date_dt"]
        home  = row["home_team"]
        away  = row["away_team"]

        cands = odds[
            (odds["date"].apply(lambda d: abs((d - gdate).days) <= 1)) &
            (odds["home_abbr"] == home) &
            (odds["away_abbr"] == away)
        ]
        if cands.empty:
            lines.append(None)
            o_prices.append(None)
            u_prices.append(None)
        else:
            best = None
            for bk in PREFERRED:
                r = cands[cands["bookmaker"] == bk]
                if not r.empty:
                    best = r.iloc[0]
                    break
            if best is None:
                best = cands.iloc[0]
            lines.append(best["total_line"])
            o_prices.append(best["over_price"])
            u_prices.append(best["under_price"])

    preds["odds_line"]        = lines
    preds["odds_over_price"]  = o_prices
    preds["odds_under_price"] = u_prices

    matched = preds["odds_line"].notna().sum()
    log.info(f"Matched {matched} / {len(preds)} games")
    return preds


# ── 核心回測 ──────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["odds_line"].notna()].copy()
    log.info(f"Running backtest on {len(df)} matched games")

    rows = []
    for _, r in df.iterrows():
        pred   = r["pred_pregame"]
        line   = r["odds_line"]
        actual = r["total_score"]
        op     = int(r["odds_over_price"])
        up     = int(r["odds_under_price"])

        # 模型機率
        mp_over  = model_prob_over(pred, line)
        mp_under = 1 - mp_over

        # 市場機率（去 vig）
        mkt_o, mkt_u = remove_vig(implied_prob(op), implied_prob(up))

        dec_o = american_to_decimal(op)
        dec_u = american_to_decimal(up)

        ev_o = expected_value(mp_over,  dec_o)
        ev_u = expected_value(mp_under, dec_u)

        k_o = kelly(mp_over,  dec_o)
        k_u = kelly(mp_under, dec_u)

        # 選邊
        if ev_o >= ev_u and ev_o > 0:
            side, ev, kf, dec, won = "OVER",  ev_o, k_o, dec_o, actual > line
        elif ev_u > 0:
            side, ev, kf, dec, won = "UNDER", ev_u, k_u, dec_u, actual <= line
        else:
            side, ev, kf, dec, won = "SKIP",  max(ev_o, ev_u), 0, 1, None

        pnl = (kf * (dec - 1) if won else -kf) if side != "SKIP" else 0

        rows.append({
            "game_date":    r["game_date"],
            "home_team":    r["home_team"],
            "away_team":    r["away_team"],
            "pred":         round(pred, 1),
            "line":         line,
            "actual":       actual,
            "error":        round(abs(pred - actual), 1),
            "mp_over":      round(mp_over, 3),
            "mkt_over":     round(mkt_o, 3),
            "edge":         round(mp_over - mkt_o, 3),
            "ev_over":      round(ev_o, 4),
            "ev_under":     round(ev_u, 4),
            "bet_side":     side,
            "bet_ev":       round(ev, 4),
            "bet_kelly":    round(kf, 4),
            "bet_won":      won,
            "pnl":          round(pnl, 4),
        })

    return pd.DataFrame(rows)


# ── 報告 ──────────────────────────────────────────────────────

def print_report(res: pd.DataFrame):
    bets  = res[res["bet_side"] != "SKIP"]
    skip  = res[res["bet_side"] == "SKIP"]
    mae   = res["error"].mean()

    total_pnl  = bets["pnl"].sum()
    total_risk = bets["bet_kelly"].sum()
    roi        = total_pnl / total_risk if total_risk > 0 else 0
    win_rate   = bets["bet_won"].mean() if len(bets) > 0 else 0

    print(f"\n{'═'*58}")
    print(f"{'  Pregame Backtest (Real Odds)':^58}")
    print(f"{'═'*58}")
    print(f"  Total games:      {len(res)}")
    print(f"  Model MAE:        {mae:.2f} pts")
    print(f"  Bets placed:      {len(bets)}  ({len(bets)/len(res)*100:.0f}%)")
    print(f"  Skipped (no EV):  {len(skip)}")
    print(f"  Win rate:         {win_rate:.1%}")
    print(f"  Total P&L:        {total_pnl:+.3f} units")
    print(f"  ROI:              {roi:+.1%}")
    print(f"{'─'*58}")

    if len(bets) > 0:
        over_b  = bets[bets["bet_side"] == "OVER"]
        under_b = bets[bets["bet_side"] == "UNDER"]
        print(f"  OVER  bets: {len(over_b):4}  "
              f"win: {over_b['bet_won'].mean():.1%}  "
              f"P&L: {over_b['pnl'].sum():+.3f}")
        print(f"  UNDER bets: {len(under_b):4}  "
              f"win: {under_b['bet_won'].mean():.1%}  "
              f"P&L: {under_b['pnl'].sum():+.3f}")
        print(f"{'─'*58}")

        # Edge 分組
        print(f"\n  {'Edge':<12}  {'場次':>5}  {'勝率':>7}  {'ROI':>8}")
        print(f"  {'─'*38}")
        bets2 = bets.copy()
        bets2["edge_bin"] = pd.cut(
            bets2["edge"].abs(),
            bins=[0, 0.03, 0.06, 0.10, 0.20, 1.0],
            labels=["0-3%", "3-6%", "6-10%", "10-20%", ">20%"]
        )
        for label, grp in bets2.groupby("edge_bin", observed=True):
            if len(grp) == 0:
                continue
            grp_roi = grp["pnl"].sum() / grp["bet_kelly"].sum()
            print(f"  {str(label):<12}  {len(grp):>5}  "
                  f"{grp['bet_won'].mean():>6.1%}  {grp_roi:>+7.1%}")
    print(f"{'═'*58}\n")


# ── 進入點 ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect-odds", action="store_true",
                        help="Collect today's live odds before backtesting")
    args = parser.parse_args()

    if args.collect_odds:
        from src.scraper.odds_scraper import fetch_live_odds
        log.info("Collecting live odds...")
        df_odds = fetch_live_odds()
        if not df_odds.empty:
            path = RAW_DIR / "odds_live.csv"
            df_odds.to_csv(path, index=False)
            log.info(f"Saved {len(df_odds)} odds → {path}")

    preds = load_pregame_predictions()
    odds_path = find_odds_file()

    if odds_path is None:
        log.warning(
            "No odds file found in data/raw/\n"
            "Options:\n"
            "  1. python src/backtest/backtest_pregame.py --collect-odds\n"
            "  2. python src/scraper/odds_scraper.py --mode historical (paid plan)\n"
            "  3. Upgrade The Odds API at https://the-odds-api.com/"
        )
        return

    odds = pd.read_csv(odds_path)
    matched = match_with_odds(preds, odds)
    results = run_backtest(matched)
    print_report(results)

    out = LOG_DIR / "backtest_pregame.csv"
    results.to_csv(out, index=False)
    log.info(f"Saved → {out}")


if __name__ == "__main__":
    main()
