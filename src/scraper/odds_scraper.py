"""
odds_scraper.py
---------------
從 The Odds API 抓取 NBA 比賽的歷史賽前 total 線。

功能：
  1. 抓歷史賽前開盤線（用於回測）
  2. 抓即時賠率（用於實戰）
  3. 把賠率跟 games CSV 配對，輸出 odds.csv

需要先申請免費 API key：
  https://the-odds-api.com/

設定方式（兩種擇一）：
  1. 建立 .env 檔案：ODDS_API_KEY=your_key_here
  2. 環境變數：set ODDS_API_KEY=your_key_here（Windows）

使用方式：
  python src/scraper/odds_scraper.py --mode historical
  python src/scraper/odds_scraper.py --mode live
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import logging
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

RAW_DIR = Path(__file__).parent.parent.parent / CFG["data"]["raw_dir"]
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL  = "https://api.the-odds-api.com/v4"
SPORT_KEY = "basketball_nba"


# ── API 工具 ──────────────────────────────────────────────────

def get_api_key() -> str:
    key = os.getenv("ODDS_API_KEY", "")
    if not key:
        raise ValueError(
            "ODDS_API_KEY not set.\n"
            "1. 前往 https://the-odds-api.com/ 申請免費 key\n"
            "2. 在專案根目錄建立 .env 檔案：\n"
            "   ODDS_API_KEY=your_key_here"
        )
    return key


def api_get(endpoint: str, params: dict) -> dict | None:
    """帶錯誤處理的 API GET。"""
    params["apiKey"] = get_api_key()
    try:
        r = requests.get(
            f"{BASE_URL}/{endpoint}",
            params=params,
            timeout=15,
            proxies={},        # 繞過 Windows 隱藏 proxy
        )
        remaining = r.headers.get("x-requests-remaining", "?")
        log.info(f"  API requests remaining: {remaining}")

        if r.status_code == 401:
            log.error("Invalid API key.")
            return None
        if r.status_code == 422:
            log.error(f"Bad request: {r.text}")
            return None
        if r.status_code == 429:
            log.error("Rate limit exceeded.")
            return None

        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"API error: {e}")
        return None


# ── 歷史賽前線 ────────────────────────────────────────────────

def fetch_historical_odds(date_str: str) -> list[dict]:
    """
    抓某一天的歷史賽前 total 線。
    date_str 格式：'2023-11-01T00:00:00Z'

    注意：歷史端點需要付費方案（Developer+）
    免費方案只能抓即時資料。
    """
    # 歷史端點（需付費方案）
    data = api_get(
        f"sports/{SPORT_KEY}/odds-history",
        params={
            "regions":    "us",
            "markets":    "totals",
            "oddsFormat": "american",
            "date":       date_str,
        }
    )
    # 若歷史端點失敗（免費方案），嘗試即時端點
    if not data:
        data = api_get(
            f"sports/{SPORT_KEY}/odds",
            params={
                "regions":    "us",
                "markets":    "totals",
                "oddsFormat": "american",
            }
        )
    if not data:
        return []

    rows = []
    for game in data:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        time = game.get("commence_time", "")

        for bookmaker in game.get("bookmakers", []):
            bk_key = bookmaker.get("key", "")
            for market in bookmaker.get("markets", []):
                if market.get("key") != "totals":
                    continue
                outcomes = market.get("outcomes", [])
                over  = next((o for o in outcomes if o["name"] == "Over"),  None)
                under = next((o for o in outcomes if o["name"] == "Under"), None)
                if not over:
                    continue

                rows.append({
                    "game_id_odds":   game.get("id", ""),
                    "home_team":      home,
                    "away_team":      away,
                    "commence_time":  time,
                    "bookmaker":      bk_key,
                    "total_line":     float(over.get("point", 0)),
                    "over_price":     int(over.get("price", -110)),
                    "under_price":    int(under.get("price", -110)) if under else -110,
                })
    return rows


def implied_prob(american_odds: int) -> float:
    """美式賠率 → 隱含機率"""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def fetch_historical_season(season: str) -> pd.DataFrame:
    """
    抓整個賽季的歷史賽前線。
    season 格式：'2023-24'

    注意：The Odds API 歷史資料需要付費方案，
    免費方案只有近期資料。
    """
    # 賽季日期範圍（常規賽大約 10 月到 4 月）
    year_start = int(season.split("-")[0])
    dates = pd.date_range(
        start=f"{year_start}-10-01",
        end=f"{year_start+1}-04-30",
        freq="7D",    # 每週抓一次（節省 API 額度）
    )

    all_rows = []
    for dt in dates:
        date_str = dt.strftime("%Y-%m-%dT00:00:00Z")
        log.info(f"Fetching odds for week of {date_str[:10]}...")
        rows = fetch_historical_odds(date_str)
        all_rows.extend(rows)
        time.sleep(1)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["season"] = season

    # 計算隱含機率
    df["over_implied_prob"]  = df["over_price"].apply(implied_prob)
    df["under_implied_prob"] = df["under_price"].apply(implied_prob)
    df["vig"] = df["over_implied_prob"] + df["under_implied_prob"] - 1  # 莊家抽水

    log.info(f"Fetched {len(df)} odds records for {season}")
    return df


# ── 即時賠率 ──────────────────────────────────────────────────

def fetch_live_odds() -> pd.DataFrame:
    """
    抓今天所有 NBA 比賽的即時 total 線。
    每 60 秒呼叫一次（配合 live_scraper.py）
    """
    data = api_get(
        f"sports/{SPORT_KEY}/odds",
        params={
            "regions":    "us",
            "markets":    "totals",
            "oddsFormat": "american",
        }
    )
    if not data:
        return pd.DataFrame()

    rows = []
    for game in data:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        for bookmaker in game.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "totals":
                    continue
                outcomes = market.get("outcomes", [])
                over  = next((o for o in outcomes if o["name"] == "Over"),  None)
                under = next((o for o in outcomes if o["name"] == "Under"), None)
                if not over:
                    continue
                rows.append({
                    "home_team":    home,
                    "away_team":    away,
                    "commence_time": game.get("commence_time", ""),
                    "bookmaker":    bookmaker.get("key", ""),
                    "total_line":   float(over.get("point", 0)),
                    "over_price":   int(over.get("price", -110)),
                    "under_price":  int(under.get("price", -110)) if under else -110,
                    "fetched_at":   datetime.now(timezone.utc).isoformat(),
                })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── 配對賽程 + 賠率 ───────────────────────────────────────────

def match_odds_to_games(games_df: pd.DataFrame,
                         odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    把 The Odds API 的賠率配對到 games CSV 的每場比賽。

    配對邏輯：
      - 日期相符（±1 天）
      - 主客場隊名相符（模糊匹配）

    回傳：games_df 加上 total_line, over_price, under_price 欄位
    """
    if odds_df.empty:
        log.warning("No odds data to match.")
        return games_df

    # 統一隊名縮寫 → 全名對照表
    ABBR_TO_FULL = {
        "ATL": "Atlanta Hawks",     "BOS": "Boston Celtics",
        "BKN": "Brooklyn Nets",     "CHA": "Charlotte Hornets",
        "CHI": "Chicago Bulls",     "CLE": "Cleveland Cavaliers",
        "DAL": "Dallas Mavericks",  "DEN": "Denver Nuggets",
        "DET": "Detroit Pistons",   "GSW": "Golden State Warriors",
        "HOU": "Houston Rockets",   "IND": "Indiana Pacers",
        "LAC": "LA Clippers",       "LAL": "Los Angeles Lakers",
        "MEM": "Memphis Grizzlies", "MIA": "Miami Heat",
        "MIL": "Milwaukee Bucks",   "MIN": "Minnesota Timberwolves",
        "NOP": "New Orleans Pelicans", "NYK": "New York Knicks",
        "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic",
        "PHI": "Philadelphia 76ers","PHX": "Phoenix Suns",
        "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings",
        "SAS": "San Antonio Spurs", "TOR": "Toronto Raptors",
        "UTA": "Utah Jazz",         "WAS": "Washington Wizards",
        "NJN": "Brooklyn Nets",     "NOH": "New Orleans Pelicans",
        "SEA": "Oklahoma City Thunder", "VAN": "Memphis Grizzlies",
    }

    # 把 odds_df 的隊名轉成縮寫（反向查詢）
    FULL_TO_ABBR = {v: k for k, v in ABBR_TO_FULL.items()}

    def to_abbr(name: str) -> str:
        # 嘗試精確匹配
        if name in FULL_TO_ABBR:
            return FULL_TO_ABBR[name]
        # 嘗試部分匹配
        for full, abbr in FULL_TO_ABBR.items():
            if any(w in name for w in full.split()):
                return abbr
        return name

    odds_df = odds_df.copy()
    odds_df["home_abbr"] = odds_df["home_team"].apply(to_abbr)
    odds_df["away_abbr"] = odds_df["away_team"].apply(to_abbr)
    odds_df["date"] = pd.to_datetime(odds_df["commence_time"]).dt.date

    # 每場比賽取最多交易量的 bookmaker（用 draftkings 優先）
    PREFERRED_BOOKS = ["draftkings", "fanduel", "betmgm", "caesars"]

    def best_line(sub: pd.DataFrame) -> pd.Series:
        for bk in PREFERRED_BOOKS:
            row = sub[sub["bookmaker"] == bk]
            if not row.empty:
                return row.iloc[0]
        return sub.iloc[0]

    games_df = games_df.copy()
    games_df["game_date_dt"] = pd.to_datetime(games_df["game_date"]).dt.date

    total_lines, over_prices, under_prices = [], [], []

    for _, game in games_df.iterrows():
        gdate = game["game_date_dt"]
        home  = game["home_team"]
        away  = game["away_team"]

        # 找配對的賠率（日期 ±1 天，隊名相符）
        candidates = odds_df[
            (odds_df["date"].apply(lambda d: abs((d - gdate).days) <= 1)) &
            (odds_df["home_abbr"] == home) &
            (odds_df["away_abbr"] == away)
        ]

        if candidates.empty:
            total_lines.append(None)
            over_prices.append(None)
            under_prices.append(None)
        else:
            best = best_line(candidates)
            total_lines.append(best["total_line"])
            over_prices.append(best["over_price"])
            under_prices.append(best["under_price"])

    games_df["odds_total_line"] = total_lines
    games_df["odds_over_price"]  = over_prices
    games_df["odds_under_price"] = under_prices

    matched = games_df["odds_total_line"].notna().sum()
    log.info(f"Matched {matched} / {len(games_df)} games with odds")

    return games_df


# ── 進入點 ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NBA odds scraper")
    parser.add_argument("--mode", choices=["historical", "live", "test"],
                        default="test")
    parser.add_argument("--season", default="2023-24")
    args = parser.parse_args()

    if args.mode == "test":
        # 只測試 API key 和連線
        log.info("Testing API connection...")
        try:
            key = get_api_key()
            log.info(f"API key found: {key[:8]}...")
            data = api_get(
                f"sports/{SPORT_KEY}/odds",
                params={"regions": "us", "markets": "totals",
                        "oddsFormat": "american"}
            )
            if data:
                log.info(f"Connection OK. Found {len(data)} upcoming games.")
                if data:
                    game = data[0]
                    log.info(f"Sample: {game['away_team']} @ {game['home_team']}")
            else:
                log.error("Connection failed.")
        except ValueError as e:
            log.error(str(e))

    elif args.mode == "live":
        df = fetch_live_odds()
        if not df.empty:
            # 去除重複列
            df = df.drop_duplicates(
                subset=["home_team", "away_team", "bookmaker", "total_line"]
            )

            # 用日期命名，保留每天的快照
            from datetime import datetime
            today = datetime.now().strftime("%Y%m%d")
            path = RAW_DIR / f"odds_{today}.csv"

            # 若今天已有資料，合併後去重
            if path.exists():
                existing = pd.read_csv(path)
                df = pd.concat([existing, df]).drop_duplicates(
                    subset=["home_team", "away_team", "bookmaker", "total_line"]
                )

            df.to_csv(path, index=False)
            log.info(f"Saved {len(df)} live odds → {path}")
            print(df[["home_team", "away_team", "bookmaker",
                       "total_line", "over_price"]].to_string())

    elif args.mode == "historical":
        season = args.season
        log.info(f"Fetching historical odds for {season}...")
        log.warning(
            "Note: The Odds API historical data requires a paid plan.\n"
            "Free plan only covers recent games.\n"
            "Starting with test fetch..."
        )
        df = fetch_historical_season(season)
        if not df.empty:
            path = RAW_DIR / f"odds_{season.replace('-','_')}.csv"
            df.to_csv(path, index=False)
            log.info(f"Saved → {path}")


if __name__ == "__main__":
    main()
