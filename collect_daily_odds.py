"""
collect_daily_odds.py
---------------------
每天賽前執行，自動收集當天所有 NBA 比賽的莊家線。

用法：
  python collect_daily_odds.py          # 手動執行
  python collect_daily_odds.py --watch  # 持續監控，每 30 分鐘更新一次

Windows 排程設定（每天 11:00 AM 執行）：
  1. 開啟「工作排程器」
  2. 建立基本工作 → 每天 11:00
  3. 動作：python C:\\path\\to\\collect_daily_odds.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
import time
from datetime import datetime

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config" / "config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

RAW_DIR = Path(__file__).parent / CFG["data"]["raw_dir"]
RAW_DIR.mkdir(parents=True, exist_ok=True)


def collect_once() -> int:
    """抓一次即時賠率，回傳儲存的場數。"""
    from src.scraper.odds_scraper import fetch_live_odds

    df = fetch_live_odds()
    if df.empty:
        log.info("No games found today.")
        return 0

    # 去重
    df = df.drop_duplicates(
        subset=["home_team", "away_team", "bookmaker", "total_line"]
    )

    # 每場比賽只保留最優惠的賠率（over_price 最高 = 賠率最好）
    best = (df.sort_values("over_price", ascending=False)
              .drop_duplicates(subset=["home_team", "away_team"])
              .copy())

    # 加上抓取時間
    best["fetched_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 存到日期命名的 CSV
    today     = datetime.now().strftime("%Y%m%d")
    save_path = RAW_DIR / f"odds_{today}.csv"

    if save_path.exists():
        existing = pd.read_csv(save_path)
        combined = pd.concat([existing, best]).drop_duplicates(
            subset=["home_team", "away_team", "bookmaker"]
        )
    else:
        combined = best

    combined.to_csv(save_path, index=False)

    n_games = best["home_team"].nunique()
    log.info(f"Saved {n_games} games, {len(combined)} rows → {save_path.name}")

    # 印出今天的賠率摘要
    print(f"\n  今日比賽（{today}）：")
    print(f"  {'客場':<25} {'主場':<25} {'最佳線':>8}  {'賠率':>6}")
    print(f"  {'─'*70}")
    for _, row in best.iterrows():
        print(f"  {row['away_team']:<25} {row['home_team']:<25} "
              f"{row['total_line']:>8.1f}  {row['over_price']:>6}")
    print()

    return n_games


def watch(interval_minutes: int = 30):
    """持續監控模式：每 interval_minutes 分鐘更新一次。"""
    log.info(f"Watch mode: collecting every {interval_minutes} minutes")
    log.info("Press Ctrl+C to stop")

    while True:
        try:
            collect_once()
            log.info(f"Next update in {interval_minutes} minutes...")
            time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            log.info("Stopped.")
            break


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true",
                        help="Keep running and update every 30 minutes")
    parser.add_argument("--interval", type=int, default=30,
                        help="Update interval in minutes (default: 30)")
    args = parser.parse_args()

    if args.watch:
        watch(interval_minutes=args.interval)
    else:
        collect_once()
