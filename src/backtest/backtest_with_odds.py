"""
backtest_with_odds.py
---------------------
用真實莊家線做有意義的回測。

邏輯：
  1. 載入模型預測結果（test_predictions.csv）
  2. 載入歷史賠率（odds_2023_24.csv）
  3. 兩者配對（日期 + 隊名）
  4. 計算每場的期望值
  5. Kelly 公式決定下注比例
  6. 輸出真實 ROI

使用方式：
  python src/backtest/backtest_with_odds.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import pandas as pd
import numpy as np
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

RAW_DIR   = Path(__file__).parent.parent.parent / CFG["data"]["raw_dir"]
MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "saved"
LOG_DIR   = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

CONF_HIGH = CFG["output"]["confidence_levels"]["high"]   # 3.0
PRED_STD  = 10.0   # 模型預測標準差（用於機率估算）


# ── 機率估算 ──────────────────────────────────────────────────

from scipy.stats import norm

def model_prob_over(pred: float, line: float,
                    std: float = PRED_STD) -> float:
    """
    用常態分布估算模型認為「過線」的機率。
    pred ~ N(pred, std²)
    """
    return float(1 - norm.cdf(line, loc=pred, scale=std))


def implied_prob(american_odds: int) -> float:
    """美式賠率 → 市場隱含機率（含 vig）"""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def remove_vig(over_prob: float, under_prob: float) -> tuple[float, float]:
    """
    移除莊家抽水，還原真實機率。
    over_prob + under_prob > 1（因為有 vig）
    """
    total = over_prob + under_prob
    return over_prob / total, under_prob / total


# ── 期望值計算 ────────────────────────────────────────────────

def american_to_decimal(odds: int) -> float:
    """美式賠率 → 十進位賠率（含本金）"""
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100 / abs(odds)) + 1


def expected_value(model_p: float, decimal_odds: float) -> float:
    """
    期望值 = model_p × (decimal_odds - 1) - (1 - model_p)
    > 0 表示有正期望值，值得下注
    """
    return model_p * (decimal_odds - 1) - (1 - model_p)


def kelly_fraction(model_p: float, decimal_odds: float,
                   fraction: float = 0.25) -> float:
    """
    Kelly 公式（使用 1/4 Kelly 降低波動）：
    f = (p × b - q) / b
    b = decimal_odds - 1
    """
    b = decimal_odds - 1
    q = 1 - model_p
    kelly = (model_p * b - q) / b
    return max(0, kelly * fraction)   # 不做空，取 1/4 Kelly


# ── 主回測邏輯 ────────────────────────────────────────────────

def load_predictions() -> pd.DataFrame:
    path = MODEL_DIR / "test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run ensemble.py first: {path}")
    return pd.read_csv(path, parse_dates=["game_date"])


def load_odds(season: str = "2023-24") -> pd.DataFrame:
    path = RAW_DIR / f"odds_{season.replace('-','_')}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No odds file found: {path}\n"
            f"Run: python src/scraper/odds_scraper.py --mode historical --season {season}"
        )
    return pd.read_csv(path)


def match_predictions_with_odds(preds: pd.DataFrame,
                                 odds: pd.DataFrame) -> pd.DataFrame:
    """每場預測配對對應的莊家線。"""
    from src.scraper.odds_scraper import match_odds_to_games
    return match_odds_to_games(preds, odds)


def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """
    對每場已配對賠率的比賽計算：
      - 模型對每條線的機率
      - 期望值
      - Kelly 建議比例
      - 實際結果
    """
    if "odds_total_line" not in df.columns:
        raise ValueError("No odds data in DataFrame. Run match first.")

    # 只回測有賠率資料的場次
    df = df[df["odds_total_line"].notna()].copy()
    log.info(f"Backtesting {len(df)} games with odds data")

    results = []
    for _, row in df.iterrows():
        pred     = row["pred_ensemble"]
        line     = row["odds_total_line"]
        actual   = row["total_score"]
        o_price  = int(row.get("odds_over_price",  -110))
        u_price  = int(row.get("odds_under_price", -110))

        # 模型機率
        m_over  = model_prob_over(pred, line)
        m_under = 1 - m_over

        # 市場隱含機率（移除 vig）
        mkt_over_raw  = implied_prob(o_price)
        mkt_under_raw = implied_prob(u_price)
        mkt_over, mkt_under = remove_vig(mkt_over_raw, mkt_under_raw)

        # 賠率轉十進位
        dec_over  = american_to_decimal(o_price)
        dec_under = american_to_decimal(u_price)

        # 期望值
        ev_over  = expected_value(m_over,  dec_over)
        ev_under = expected_value(m_under, dec_under)

        # Kelly
        k_over  = kelly_fraction(m_over,  dec_over)
        k_under = kelly_fraction(m_under, dec_under)

        # 決定下注方向（選期望值最高的那邊）
        if ev_over > ev_under and ev_over > 0:
            bet_side   = "OVER"
            bet_ev     = ev_over
            bet_kelly  = k_over
            bet_dec    = dec_over
            bet_won    = actual > line
        elif ev_under > 0:
            bet_side   = "UNDER"
            bet_ev     = ev_under
            bet_kelly  = k_under
            bet_dec    = dec_under
            bet_won    = actual <= line
        else:
            bet_side   = "SKIP"    # 無正期望值，不下注
            bet_ev     = max(ev_over, ev_under)
            bet_kelly  = 0
            bet_dec    = 1
            bet_won    = None

        # 損益（以投入 1 單位計算）
        if bet_side == "SKIP" or bet_kelly == 0:
            pnl = 0
        elif bet_won:
            pnl = bet_kelly * (bet_dec - 1)
        else:
            pnl = -bet_kelly

        results.append({
            "game_date":     row["game_date"],
            "home_team":     row["home_team"],
            "away_team":     row["away_team"],
            "pred":          round(pred, 1),
            "line":          line,
            "actual":        actual,
            "model_p_over":  round(m_over,  3),
            "mkt_p_over":    round(mkt_over, 3),
            "edge_over":     round(m_over - mkt_over, 3),
            "ev_over":       round(ev_over,  4),
            "ev_under":      round(ev_under, 4),
            "bet_side":      bet_side,
            "bet_ev":        round(bet_ev,   4),
            "bet_kelly":     round(bet_kelly, 4),
            "bet_won":       bet_won,
            "pnl":           round(pnl, 4),
        })

    return pd.DataFrame(results)


# ── 報告輸出 ──────────────────────────────────────────────────

def print_report(results: pd.DataFrame):
    bets = results[results["bet_side"] != "SKIP"]
    skip = results[results["bet_side"] == "SKIP"]

    total_pnl   = bets["pnl"].sum()
    total_risk  = bets["bet_kelly"].sum()
    roi         = total_pnl / total_risk if total_risk > 0 else 0
    win_rate    = bets["bet_won"].mean()
    avg_edge    = bets["edge_over"].abs().mean()
    avg_ev      = bets["bet_ev"].mean()

    print(f"\n{'═'*55}")
    print(f"{'  Real Backtest (with actual odds lines)':^55}")
    print(f"{'═'*55}")
    print(f"  Total games:     {len(results)}")
    print(f"  Bets placed:     {len(bets)} ({len(bets)/len(results)*100:.0f}%)")
    print(f"  Skipped (no EV): {len(skip)}")
    print(f"  Win rate:        {win_rate:.1%}")
    print(f"  Avg edge:        {avg_edge:.1%}")
    print(f"  Avg EV/bet:      {avg_ev:.1%}")
    print(f"  Total P&L:       {total_pnl:+.3f} units")
    print(f"  Total risk:      {total_risk:.3f} units")
    print(f"  ROI:             {roi:+.1%}")
    print(f"{'═'*55}\n")

    # Over vs Under 分析
    over_bets  = bets[bets["bet_side"] == "OVER"]
    under_bets = bets[bets["bet_side"] == "UNDER"]
    print(f"  OVER  bets: {len(over_bets):4}  "
          f"win rate: {over_bets['bet_won'].mean():.1%}  "
          f"P&L: {over_bets['pnl'].sum():+.3f}")
    print(f"  UNDER bets: {len(under_bets):4}  "
          f"win rate: {under_bets['bet_won'].mean():.1%}  "
          f"P&L: {under_bets['pnl'].sum():+.3f}")
    print()

    # 依 edge 大小分組
    print(f"  {'Edge 區間':<15}  {'場次':>5}  {'勝率':>7}  {'ROI':>8}")
    print(f"  {'─'*40}")
    bets_copy = bets.copy()
    bets_copy["edge_bin"] = pd.cut(
        bets_copy["edge_over"].abs(),
        bins=[0, 0.03, 0.06, 0.10, 0.20, 1.0],
        labels=["0-3%", "3-6%", "6-10%", "10-20%", ">20%"]
    )
    for label, grp in bets_copy.groupby("edge_bin", observed=True):
        if len(grp) == 0:
            continue
        grp_roi = grp["pnl"].sum() / grp["bet_kelly"].sum()
        print(f"  {str(label):<15}  {len(grp):>5}  "
              f"{grp['bet_won'].mean():>6.1%}  {grp_roi:>+7.1%}")
    print()


# ── 進入點 ────────────────────────────────────────────────────

def main():
    preds = load_predictions()

    try:
        odds = load_odds("2023-24")
    except FileNotFoundError as e:
        log.warning(str(e))
        log.info(
            "\n尚未有賠率資料，先以模擬賠率示範回測邏輯...\n"
            "（所有場次假設莊家線 = 賽季平均 220，賠率 -110）"
        )
        # 用模擬資料示範回測格式
        preds["odds_total_line"]  = 220.0
        preds["odds_over_price"]  = -110
        preds["odds_under_price"] = -110
        results = run_backtest(preds)
        print_report(results)
        return

    # 配對真實賠率
    merged  = match_predictions_with_odds(preds, odds)
    results = run_backtest(merged)
    print_report(results)

    # 儲存結果
    out = LOG_DIR / "backtest_real_odds.csv"
    results.to_csv(out, index=False)
    log.info(f"Saved → {out}")


if __name__ == "__main__":
    main()
