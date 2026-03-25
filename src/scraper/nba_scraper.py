"""
nba_scraper.py
--------------
從 nba_api 抓取歷史比賽資料，包含：
  - 每場比賽基本資訊（主客場、日期、最終比分）
  - 各節得分（Q1, Q2, Q3, Q4）
  - 球隊進階統計（Pace, OffRtg, DefRtg）

輸出：
  data/raw/games_{season}.csv        每場比賽 + 各節比分
  data/raw/team_stats_{season}.csv   球隊賽季進階統計

使用方式：
  while( python src/scraper/nba_scraper.py --season 2023-24 ){ break }
  python src/scraper/nba_scraper.py
  python src/scraper/nba_scraper.py --season 2023-24
  python src/scraper/nba_scraper.py --all
"""

import argparse
import time
import logging
import os
from pathlib import Path

# 停用 Windows 自動偵測的隱藏 proxy（會攔截並阻擋某些 HTTPS 連線）
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

import pandas as pd
import yaml
from nba_api.stats.endpoints import (
    leaguegamelog,
    boxscoresummaryv2,
    leaguedashteamstats,
)
from nba_api.stats.static import teams
import requests

# 建立共用 session，明確設定 proxies={} 強制直連（繞過 Windows 隱藏 proxy）
_session = requests.Session()
_session.proxies = {}

# 注入到 nba_api
try:
    from nba_api.stats.library.http import NBAStatsHTTP
    NBAStatsHTTP.set_session(_session)
except Exception:
    pass


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

RAW_DIR = Path(__file__).parent.parent.parent / CFG["data"]["raw_dir"]
RAW_DIR.mkdir(parents=True, exist_ok=True)

DELAY = CFG["data"]["request_delay"]   # 每次 API call 之間的等待秒數


# ── 工具函式 ──────────────────────────────────────────────────
def safe_request(fn, *args, retries=5, **kwargs):
    """
    帶重試機制的 API 呼叫包裝器。
    每次失敗後等待時間指數成長。
    """
    for attempt in range(retries):
        try:
            time.sleep(DELAY)
            return fn(*args, **kwargs)
        except Exception as e:
            wait = DELAY * (2 ** attempt)
            log.warning(
                f"Request failed (attempt {attempt+1}/{retries}): "
                f"{type(e).__name__} — retrying in {wait:.1f}s"
            )
            time.sleep(wait)
    log.error(f"All {retries} attempts failed for {fn.__name__}")
    return None


def season_type_str(season: str) -> str:
    """'2023-24' → '2023-24'（nba_api 格式已正確）"""
    return season


# ── 主要抓取函式 ───────────────────────────────────────────────

def fetch_game_list(season: str) -> pd.DataFrame:
    """
    抓取該賽季所有比賽清單（常規賽）。
    回傳欄位：GAME_ID, GAME_DATE, TEAM_ID, TEAM_ABBREVIATION,
              MATCHUP, WL, PTS, ...
    """
    log.info(f"Fetching game list for {season}...")
    result = safe_request(
        leaguegamelog.LeagueGameLog,
        season=season,
        season_type_all_star="Regular Season",
    )
    if result is None:
        return pd.DataFrame()

    df = result.get_data_frames()[0]
    log.info(f"  → {len(df)} team-game rows fetched")
    return df


def fetch_quarter_scores(game_id: str) -> dict:
    """
    抓取單場比賽各節比分。
    回傳：
      {
        "home_q1": 32, "home_q2": 28, "home_q3": 30, "home_q4": 29,
        "away_q1": 27, "away_q2": 31, "away_q3": 26, "away_q4": 25,
        "home_total": 119, "away_total": 109
      }
    """
    result = safe_request(boxscoresummaryv2.BoxScoreSummaryV2, game_id=game_id)
    if result is None:
        return {}

    line_score = result.line_score.get_data_frame()

    if len(line_score) < 2:
        return {}

    # line_score 第 0 列是客場，第 1 列是主場（nba_api 慣例）
    away = line_score.iloc[0]
    home = line_score.iloc[1]

    def safe_int(val):
        try:
            return int(val) if pd.notna(val) else 0
        except Exception:
            return 0

    return {
        "home_q1":    safe_int(home.get("PTS_QTR1")),
        "home_q2":    safe_int(home.get("PTS_QTR2")),
        "home_q3":    safe_int(home.get("PTS_QTR3")),
        "home_q4":    safe_int(home.get("PTS_QTR4")),
        "home_total": safe_int(home.get("PTS")),
        "away_q1":    safe_int(away.get("PTS_QTR1")),
        "away_q2":    safe_int(away.get("PTS_QTR2")),
        "away_q3":    safe_int(away.get("PTS_QTR3")),
        "away_q4":    safe_int(away.get("PTS_QTR4")),
        "away_total": safe_int(away.get("PTS")),
    }


def build_games_df(season: str) -> pd.DataFrame:
    """
    整合比賽清單 + 各節比分，輸出每場一列的 DataFrame。
    支援斷點續抓：每場抓完立刻寫入 CSV，中斷後重跑自動跳過已抓的場次。

    最終欄位：
      game_id, game_date, season,
      home_team, away_team,
      home_q1, home_q2, home_q3, home_q4, home_total,
      away_q1, away_q2, away_q3, away_q4, away_total,
      total_score, h1_total, q1_total
    """
    games_path = RAW_DIR / f"games_{season.replace('-','_')}.csv"

    raw = fetch_game_list(season)
    if raw.empty:
        return pd.DataFrame()

    home_games = raw[raw["MATCHUP"].str.contains(r"vs\.")].copy()
    away_games = raw[raw["MATCHUP"].str.contains("@")].copy()

    home_games = home_games.rename(columns={
        "TEAM_ABBREVIATION": "home_team",
        "TEAM_ID": "home_team_id",
        "PTS": "home_total_check",
    })
    away_games = away_games.rename(columns={
        "TEAM_ABBREVIATION": "away_team",
        "TEAM_ID": "away_team_id",
        "PTS": "away_total_check",
    })

    merged = home_games[["GAME_ID", "GAME_DATE", "home_team", "home_team_id", "home_total_check"]].merge(
        away_games[["GAME_ID", "away_team", "away_team_id", "away_total_check"]],
        on="GAME_ID",
    )

    # ── 斷點續抓：載入已存的進度 ──────────────────────────────
    games_path = RAW_DIR / f"games_{season.replace('-','_')}.csv"
    if games_path.exists():
        existing   = pd.read_csv(games_path, dtype={"game_id": str})
        done_ids   = set(existing["game_id"].astype(str))
        log.info(f"Resuming: {len(done_ids)} / {len(merged)} games already done")
    else:
        existing = pd.DataFrame()
        done_ids = set()

    total_games = len(merged)
    new_rows    = []

    for idx, (_, row) in enumerate(merged.iterrows()):
        game_id = str(row["GAME_ID"])

        # 已抓過，直接跳過
        if game_id in done_ids:
            continue

        log.info(f"  [{idx+1}/{total_games}] game_id={game_id}  "
                 f"{row['away_team']} @ {row['home_team']}")

        q = fetch_quarter_scores(game_id)
        if not q:
            log.warning(f"  Skipping {game_id} — no quarter data")
            continue

        total = q["home_total"] + q["away_total"]
        h1    = (q["home_q1"] + q["home_q2"]) + (q["away_q1"] + q["away_q2"])
        q1    = q["home_q1"] + q["away_q1"]

        game_row = {
            "game_id":      game_id,
            "game_date":    row["GAME_DATE"],
            "season":       season,
            "home_team":    row["home_team"],
            "away_team":    row["away_team"],
            "home_team_id": row["home_team_id"],
            "away_team_id": row["away_team_id"],
            "home_q1": q["home_q1"], "home_q2": q["home_q2"],
            "home_q3": q["home_q3"], "home_q4": q["home_q4"],
            "home_total": q["home_total"],
            "away_q1": q["away_q1"], "away_q2": q["away_q2"],
            "away_q3": q["away_q3"], "away_q4": q["away_q4"],
            "away_total": q["away_total"],
            "total_score": total,
            "h1_total":    h1,
            "q1_total":    q1,
        }
        new_rows.append(game_row)

        # ── 每抓完一場立刻追加寫入 CSV ──────────────────────
        row_df = pd.DataFrame([game_row])
        write_header = not games_path.exists()
        row_df.to_csv(games_path, mode="a", header=write_header, index=False)

    # 回傳完整資料（已存 + 本次新抓）
    if new_rows:
        log.info(f"Newly fetched: {len(new_rows)} games")
    else:
        log.info("No new games to fetch")

    if games_path.exists():
        df = pd.read_csv(games_path).sort_values("game_date").reset_index(drop=True)
        log.info(f"Total in file: {len(df)} games for {season}")
        return df

    return pd.DataFrame()


def fetch_team_advanced_stats(season: str) -> pd.DataFrame:
    """
    抓取球隊賽季進階統計：Pace, OffRtg, DefRtg 等。
    這些數字會用在「賽前特徵」。
    """
    log.info(f"Fetching team advanced stats for {season}...")
    result = safe_request(
        leaguedashteamstats.LeagueDashTeamStats,
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
    )
    if result is None:
        return pd.DataFrame()

    df = result.get_data_frames()[0]
    cols_keep = [
        "TEAM_ID", "TEAM_NAME", "GP",
        "E_PACE", "PACE",
        "E_OFF_RATING", "OFF_RATING",
        "E_DEF_RATING", "DEF_RATING",
        "E_NET_RATING", "NET_RATING",
        "AST_RATIO", "TM_TOV_PCT",
    ]
    cols_keep = [c for c in cols_keep if c in df.columns]
    df = df[cols_keep].copy()
    df["season"] = season
    log.info(f"  → {len(df)} teams")
    return df


# ── 進入點 ────────────────────────────────────────────────────

def scrape_season(season: str, skip_stats: bool = False):
    """
    抓一個賽季的完整資料。
    比賽資料支援斷點續抓，team stats 失敗時自動跳過不影響主流程。
    """
    # 比賽資料
    build_games_df(season)

    # 球隊進階統計（選用，失敗不影響後續特徵工程）
    if skip_stats:
        log.info("Skipping team stats (--skip-stats flag set)")
        return

    stats_path = RAW_DIR / f"team_stats_{season.replace('-','_')}.csv"
    if stats_path.exists():
        log.info(f"Stats already exists, skipping: {stats_path}")
    else:
        log.info("Fetching team advanced stats (optional, skipping if fails)...")
        df_stats = fetch_team_advanced_stats(season)
        if not df_stats.empty:
            df_stats.to_csv(stats_path, index=False)
            log.info(f"Saved → {stats_path}")
        else:
            log.warning(
                "Team stats unavailable — this is OK. "
                "manual_features.py will estimate pace from game data instead."
            )


def main():
    parser = argparse.ArgumentParser(description="NBA data scraper")
    parser.add_argument("--season", type=str, default=None,
                        help="Single season, e.g. 2023-24")
    parser.add_argument("--all", action="store_true",
                        help="Scrape all seasons defined in config.yaml")
    parser.add_argument("--skip-stats", action="store_true",
                        help="Skip team advanced stats (use if API keeps failing)")
    args = parser.parse_args()

    skip = args.skip_stats

    if args.all:
        seasons = CFG["data"]["seasons"]
        log.info(f"Scraping {len(seasons)} seasons: {seasons}")
        for s in seasons:
            scrape_season(s, skip_stats=skip)
    elif args.season:
        scrape_season(args.season, skip_stats=skip)
    else:
        latest = CFG["data"]["seasons"][-1]
        log.info(f"No season specified, defaulting to {latest}")
        scrape_season(latest, skip_stats=skip)


if __name__ == "__main__":
    main()