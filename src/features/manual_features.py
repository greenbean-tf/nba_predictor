"""
manual_features.py
------------------
從歷史比賽資料手工設計特徵，分兩層：

  Layer 1 — 賽前特徵（pregame）
    兩隊近 N 場 rolling 統計、Pace、OffRtg、DefRtg、
    主客場、背靠背、休息天數

  Layer 2 — 盤中特徵（in-game）
    Q1 / 半場合計、半場進度比、即時 Pace 推算

使用方式：
  python src/features/manual_features.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from pathlib import Path

import pandas as pd
import numpy as np
import yaml

# ── 設定 ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

RAW_DIR      = Path(__file__).parent.parent.parent / CFG["data"]["raw_dir"]
PROC_DIR     = Path(__file__).parent.parent.parent / CFG["data"]["processed_dir"]
FEAT_DIR     = Path(__file__).parent.parent.parent / CFG["data"]["features_dir"]
WINDOWS      = CFG["features"]["rolling_windows"]       # [5, 10, 20]
MIN_GAMES    = CFG["features"]["min_games_required"]    # 10

PROC_DIR.mkdir(parents=True, exist_ok=True)
FEAT_DIR.mkdir(parents=True, exist_ok=True)


# ── Step 1：合併所有賽季資料 ───────────────────────────────────

def load_all_games() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("games_*.csv"))
    if not files:
        raise FileNotFoundError(f"No games CSV found in {RAW_DIR}. "
                                "Run nba_scraper.py first.")
    dfs = [pd.read_csv(f, parse_dates=["game_date"]) for f in files]
    df  = pd.concat(dfs, ignore_index=True).sort_values("game_date")
    before = len(df)
    df = df[df["total_score"] > 0]
    removed = before - len(df)
    if removed > 0:
        log.warning(f"Removed {removed} rows with total_score=0 (incomplete game data)")
    log.info(f"Loaded {len(df)} games from {len(files)} season files")
    return df


def load_all_team_stats() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("team_stats_*.csv"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


# ── Step 2：賽前特徵（Layer 1）────────────────────────────────

def add_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    對每支球隊計算 rolling 統計，避免 data leakage（用 shift(1)）。

    新增欄位（以 W=10 為例）：
      home_avg_scored_10   客場 → 主場近 10 場平均得分
      home_avg_allowed_10  主場近 10 場平均失分
      away_avg_scored_10
      away_avg_allowed_10
      home_avg_total_10    主場視角的近 10 場總得分平均
      away_avg_total_10
    """
    log.info("Adding rolling stats...")

    # 建立「從每支球隊視角」的長格式表
    home_view = df[["game_id", "game_date", "home_team",
                    "home_total", "away_total", "total_score"]].copy()
    home_view.rename(columns={
        "home_team":   "team",
        "home_total":  "scored",
        "away_total":  "allowed",
    }, inplace=True)
    home_view["is_home"] = True

    away_view = df[["game_id", "game_date", "away_team",
                    "away_total", "home_total", "total_score"]].copy()
    away_view.rename(columns={
        "away_team":   "team",
        "away_total":  "scored",
        "home_total":  "allowed",
    }, inplace=True)
    away_view["is_home"] = False

    long = pd.concat([home_view, away_view]).sort_values("game_date")

    for W in WINDOWS:
        long[f"avg_scored_{W}"]  = (long.groupby("team")["scored"]
                                        .transform(lambda x: x.shift(1).rolling(W, min_periods=3).mean()))
        long[f"avg_allowed_{W}"] = (long.groupby("team")["allowed"]
                                        .transform(lambda x: x.shift(1).rolling(W, min_periods=3).mean()))
        long[f"avg_total_{W}"]   = (long.groupby("team")["total_score"]
                                        .transform(lambda x: x.shift(1).rolling(W, min_periods=3).mean()))

    # 計算各隊的得分穩定度（標準差）
    long["std_scored_10"] = (long.groupby("team")["scored"]
                                 .transform(lambda x: x.shift(1).rolling(10, min_periods=3).std()))

    # 拆回主客場視角，merge 回 df
    home_feats = long[long["is_home"]].set_index("game_id")
    away_feats = long[~long["is_home"]].set_index("game_id")

    for W in WINDOWS:
        df = df.join(home_feats[[f"avg_scored_{W}", f"avg_allowed_{W}", f"avg_total_{W}"]].add_prefix("home_"), on="game_id")
        df = df.join(away_feats[[f"avg_scored_{W}", f"avg_allowed_{W}", f"avg_total_{W}"]].add_prefix("away_"), on="game_id")

    df = df.join(home_feats[["std_scored_10"]].rename(columns={"std_scored_10": "home_std_scored_10"}), on="game_id")
    df = df.join(away_feats[["std_scored_10"]].rename(columns={"std_scored_10": "away_std_scored_10"}), on="game_id")

    return df


def add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算兩隊距上一場比賽的休息天數（含背靠背偵測）。
    """
    log.info("Adding rest days...")

    def get_last_game_date(group):
        group = group.sort_values("game_date")
        group["last_game_date"] = group["game_date"].shift(1)
        return group

    home_view = df[["game_id", "game_date", "home_team"]].copy()
    home_view = home_view.groupby("home_team", group_keys=False).apply(get_last_game_date)
    home_view["home_rest_days"] = (home_view["game_date"] - home_view["last_game_date"]).dt.days.clip(upper=14)
    home_view["home_b2b"] = (home_view["home_rest_days"] == 1).astype(int)

    away_view = df[["game_id", "game_date", "away_team"]].copy()
    away_view = away_view.groupby("away_team", group_keys=False).apply(get_last_game_date)
    away_view["away_rest_days"] = (away_view["game_date"] - away_view["last_game_date"]).dt.days.clip(upper=14)
    away_view["away_b2b"] = (away_view["away_rest_days"] == 1).astype(int)

    df = df.join(home_view.set_index("game_id")[["home_rest_days", "home_b2b"]], on="game_id")
    df = df.join(away_view.set_index("game_id")[["away_rest_days", "away_b2b"]], on="game_id")

    return df


def add_pregame_implied_total(df: pd.DataFrame) -> pd.DataFrame:
    """
    用兩隊 rolling 數字推算「隱含總分」：
      implied_total = home_avg_scored_10 + away_avg_scored_10
                    + home_avg_allowed_10 + away_avg_allowed_10) / 2
    """
    df["implied_total_10"] = (
        df["home_avg_scored_10"] + df["away_avg_scored_10"] +
        df["home_avg_allowed_10"] + df["away_avg_allowed_10"]
    ) / 2
    return df


def add_team_advanced_stats(df: pd.DataFrame,
                             stats: pd.DataFrame) -> pd.DataFrame:
    """
    把球隊進階統計（Pace, OffRtg, DefRtg）join 進來。
    """
    if stats.empty:
        log.warning("No team stats file found, skipping advanced stats.")
        return df

    log.info("Joining team advanced stats...")
    stats_clean = stats[["TEAM_ID", "season", "PACE", "OFF_RATING", "DEF_RATING"]].copy()
    stats_clean.columns = ["team_id", "season", "pace", "off_rtg", "def_rtg"]

    # home
    df = df.merge(
        stats_clean.rename(columns=lambda c: f"home_{c}" if c != "season" else c),
        left_on=["home_team_id", "season"],
        right_on=["home_team_id", "season"],
        how="left",
    )
    # away
    df = df.merge(
        stats_clean.rename(columns=lambda c: f"away_{c}" if c != "season" else c),
        left_on=["away_team_id", "season"],
        right_on=["away_team_id", "season"],
        how="left",
    )

    # Pace 匹配度（兩隊 pace 的平均 → 代表這場預計節奏）
    df["pace_matchup"] = (df["home_pace"] + df["away_pace"]) / 2

    return df


# ── Step 3：盤中特徵（Layer 2）────────────────────────────────

def add_ingame_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算盤中特徵（需要有 q1_total、h1_total）。

    新增欄位：
      q1_pace_ratio  : Q1 合計 / (implied_total_10 / 4)
      h1_pace_ratio  : 半場合計 / (implied_total_10 / 2)  ← 核心特徵
      q1_home_lead   : Q1 主場領先分
      h1_home_lead   : 半場主場領先分
      q2_total       : 第二節合計（h1 - q1）
      q1_vs_season   : Q1 合計 vs 兩隊本季平均第一節
    """
    log.info("Adding in-game features...")

    # 半場 / Q1 進度比
    df["q1_pace_ratio"] = df["q1_total"] / (df["implied_total_10"] / 4).replace(0, np.nan)
    df["h1_pace_ratio"] = df["h1_total"] / (df["implied_total_10"] / 2).replace(0, np.nan)

    # 領先分
    df["q1_home_lead"] = df["home_q1"] - df["away_q1"]
    df["h1_home_lead"] = (df["home_q1"] + df["home_q2"]) - (df["away_q1"] + df["away_q2"])

    # 第二節合計
    df["q2_total"] = df["h1_total"] - df["q1_total"]

    # Q1 vs 本季平均 Q1（滾動）
    df["q1_vs_avg"] = df["q1_total"] - (df["implied_total_10"] / 4)

    return df


# ── Step 4：清理 + 輸出 ───────────────────────────────────────

def drop_insufficient_history(df: pd.DataFrame) -> pd.DataFrame:
    """移除 rolling 欄位有 NaN 的列（歷史場次不足）。"""
    key_cols = [f"home_avg_scored_{WINDOWS[1]}",
                f"away_avg_scored_{WINDOWS[1]}"]
    before = len(df)
    df = df.dropna(subset=key_cols)
    log.info(f"Dropped {before - len(df)} rows with insufficient history "
             f"({len(df)} remaining)")
    return df


def build_features(save: bool = True) -> pd.DataFrame:
    """完整特徵工程流程。"""
    games = load_all_games()
    stats = load_all_team_stats()

    df = games.copy()
    df = add_rolling_stats(df)
    df = add_rest_days(df)
    df = add_pregame_implied_total(df)
    df = add_team_advanced_stats(df, stats)
    df = add_ingame_features(df)
    df = drop_insufficient_history(df)

    if save:
        out_path = FEAT_DIR / "features.csv"
        df.to_csv(out_path, index=False)
        log.info(f"Saved features → {out_path}  shape={df.shape}")

    return df


def build_features_live(game: dict, team_history: pd.DataFrame,
                         implied_total: float) -> pd.Series:
    """
    即時預測用：從單場比賽資料 + 歷史資料即時計算特徵。

    Args:
        game          : live_scraper 回傳的 dict
        team_history  : 兩隊近期比賽歷史（從 features.csv 過濾）
        implied_total : 莊家開盤線（或用 rolling 推算）

    Returns:
        pd.Series，與 features.csv 欄位相同
    """
    home = game["home_team"]
    away = game["away_team"]

    def team_rolling(team: str, col: str, W: int) -> float:
        mask = ((team_history["home_team"] == team) |
                (team_history["away_team"] == team))
        sub = team_history[mask].tail(W)
        if sub.empty:
            return np.nan
        scored  = sub.apply(lambda r: r["home_total"] if r["home_team"] == team
                             else r["away_total"], axis=1)
        allowed = sub.apply(lambda r: r["away_total"] if r["home_team"] == team
                             else r["home_total"], axis=1)
        return scored.mean() if col == "scored" else allowed.mean()

    feats = {}
    for W in WINDOWS:
        feats[f"home_avg_scored_{W}"]  = team_rolling(home, "scored",  W)
        feats[f"home_avg_allowed_{W}"] = team_rolling(home, "allowed", W)
        feats[f"away_avg_scored_{W}"]  = team_rolling(away, "scored",  W)
        feats[f"away_avg_allowed_{W}"] = team_rolling(away, "allowed", W)

    feats["implied_total_10"] = implied_total
    feats["q1_total"]         = game["q1_total"]
    feats["h1_total"]         = game["h1_total"]
    feats["q1_pace_ratio"]    = game["q1_total"] / (implied_total / 4) if implied_total else np.nan
    feats["h1_pace_ratio"]    = game["h1_total"] / (implied_total / 2) if implied_total else np.nan
    feats["q1_home_lead"]     = game["home_q1"] - game["away_q1"]
    feats["h1_home_lead"]     = (game["home_q1"] + game["home_q2"]) - (game["away_q1"] + game["away_q2"])
    feats["q2_total"]         = game["h1_total"] - game["q1_total"]
    feats["q1_vs_avg"]        = game["q1_total"] - (implied_total / 4)

    return pd.Series(feats)


# ── 進入點 ────────────────────────────────────────────────────

if __name__ == "__main__":
    df = build_features(save=True)
    print("\n特徵一覽：")
    print(df.describe().round(2).to_string())
    print(f"\n欄位數：{len(df.columns)}  資料筆數：{len(df)}")
