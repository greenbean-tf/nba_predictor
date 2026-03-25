"""
ab_simulator.py
---------------
同時模擬賽前模型 vs 半場模型的投資績效。

使用方式：
  python src/live/ab_simulator.py --live --capital 10000
  python src/live/ab_simulator.py --status --capital 10000
  python src/live/ab_simulator.py --report
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse, json, logging, pickle, threading, time
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import norm
import yaml

# ── Log 設定：同時寫檔案和螢幕，強制每行 flush ────────────────
_BASE_DIR = Path(__file__).parent.parent.parent
_LOG_DIR  = _BASE_DIR / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "simulator.log"

_file_handler   = logging.FileHandler(_LOG_FILE, encoding="utf-8")
_stdout_handler = logging.StreamHandler(sys.stdout)

for _h in [_file_handler, _stdout_handler]:
    _h.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))

logging.basicConfig(level=logging.INFO, handlers=[_file_handler, _stdout_handler], force=True)

# 強制每條 log 立刻寫入磁碟（解決睡眠時記錄消失的問題）
# 強制每條 log 立刻寫入磁碟（包含 FileHandler）
_orig_emit = logging.StreamHandler.emit
def _flush_emit(self, record):
    _orig_emit(self, record)
    self.flush()
    # FileHandler 額外呼叫 os.fsync 確保寫入磁碟（不只是 OS 快取）
    if hasattr(self, 'stream') and hasattr(self.stream, 'fileno'):
        try:
            import os
            os.fsync(self.stream.fileno())
        except (OSError, AttributeError):
            pass
logging.StreamHandler.emit = _flush_emit

log = logging.getLogger(__name__)

CONFIG_PATH = _BASE_DIR / "config" / "config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

FEAT_DIR  = _BASE_DIR / CFG["data"]["features_dir"]
MODEL_DIR = _BASE_DIR / "models" / "saved"
LOG_DIR   = _LOG_DIR
AB_LOG    = LOG_DIR / "ab_simulation.jsonl"

MIN_EV            = 0.02
MIN_EDGE          = 0.03
KELLY_FRAC        = 0.25
PRED_STD_PREGAME  = 13.0
PRED_STD_HALFTIME = 9.0

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

ESPN_ABBR_FIX = {
    "GS": "GSW", "NO": "NOP", "SA": "SAS",
    "NY": "NYK", "UTAH": "UTA", "PHO": "PHX",
    "WSH": "WAS", "GSD": "GSW", "NJ": "BKN",
    "SEA": "OKC", "NOR": "NOP", "CHA": "CHA",
    "NJN": "BKN", "CHH": "CHA", "NOH": "NOP",
}


# ── 模擬帳戶 ──────────────────────────────────────────────────

class SimAccount:
    def __init__(self, name: str, capital: float, kelly_frac: float = KELLY_FRAC):
        self.name       = name
        self.total      = capital
        self.initial    = capital
        self.kelly_frac = kelly_frac
        self._bets: list = []
        self._load()

    @property
    def committed(self) -> float:
        return sum(b["amount"] for b in self._bets
                   if b["result"] is None and b["model"] == self.name
                   and b["date"] == str(date.today()))

    @property
    def available(self) -> float:
        return self.total - self.committed

    def kelly_amount(self, model_p: float, market_p: float) -> float:
        if market_p <= 0 or market_p >= 1:
            return 0
        f = max(0, (model_p - market_p) / (1 - market_p) * self.kelly_frac)
        return round(self.available * f, 2)

    def bet(self, game: str, line: float, side: str,
            model_p: float, market_p: float,
            pred: float, trigger: str) -> dict | None:
        edge = model_p - market_p
        ev   = model_p - market_p  # Polymarket: EV = edge

        if edge < MIN_EDGE or ev < MIN_EV:
            return None

        amount = self.kelly_amount(model_p, market_p)
        if amount < 1:
            return None

        # 同一場比賽同一線同一方向不重複下注
        for b in self._bets:
            if (b["game"] == game and b["line"] == line
                    and b["side"] == side and b["model"] == self.name):
                return None

        record = {
            "model":    self.name,
            "date":     str(date.today()),
            "ts":       datetime.now().isoformat(),
            "trigger":  trigger,
            "game":     game,
            "line":     line,
            "side":     side,
            "pred":     pred,
            "model_p":     round(model_p,  4),
            "market_p":    round(market_p, 4),
            "filled_price": round(market_p, 4),  # 模擬用 CLOB ask 當 filled_price
            "edge":        round(edge, 4),
            "ev":          round(ev,   4),
            "amount":   amount,
            "capital":  self.available,
            "result":   None,
            "pnl":      None,
        }
        self._bets.append(record)
        self._save(record)

        log.info(f"  [{self.name}] BET {game} {line} {side} "
                 f"${amount:.0f} | EV={ev:+.1%} | avail=${self.available:.0f}")
        return record

    def settle(self, game: str, line: float, actual_total: float):
        changed = False
        for b in self._bets:
            if (b["game"] == game and b["line"] == line
                    and b["result"] is None and b["model"] == self.name):
                won = (actual_total > line) if b["side"] == "OVER" else (actual_total < line)
                if won:
                    # 用 filled_price 計算（模擬版 = CLOB ask，比 Gamma mid 準確）
                    fp     = b.get("filled_price") or b.get("market_p", 0.5)
                    payout = b["amount"] / fp
                    pnl    = round(payout - b["amount"], 2)
                    self.total += pnl
                else:
                    pnl = -b["amount"]
                    self.total += pnl
                b["result"]       = "win" if won else "loss"
                b["pnl"]          = round(pnl, 2)
                b["actual_total"] = actual_total
                log.info(f"  [{self.name}] {'WIN' if won else 'LOSS'} "
                         f"{game} {line} {b['side']} ${b['amount']:.0f} pnl={pnl:+.2f}")
                changed = True
        if changed:
            self._rewrite()

    def stats(self) -> dict:
        settled = [b for b in self._bets if b["result"] is not None
                   and b["model"] == self.name]
        if not settled:
            return {
                "n": 0, "wins": 0, "win_rate": 0,
                "total_pnl": 0, "roi": 0,
                "capital": self.total,
                "gain_pct": (self.total - self.initial) / self.initial,
            }
        wins = sum(1 for b in settled if b["result"] == "win")
        pnl  = sum(b["pnl"] for b in settled)
        roi  = pnl / sum(b["amount"] for b in settled) if settled else 0
        return {
            "n": len(settled), "wins": wins,
            "win_rate":  wins / len(settled),
            "total_pnl": pnl, "roi": roi,
            "capital":   self.total,
            "gain_pct":  (self.total - self.initial) / self.initial,
        }

    def _load(self):
        self._bets = []
        if not AB_LOG.exists():
            return
        with open(AB_LOG, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    b = json.loads(line)
                    if b.get("model") == self.name:
                        self._bets.append(b)
                        if b.get("pnl") is not None:
                            self.total += b["pnl"]

    def _save(self, record: dict):
        with open(AB_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
            f.flush()

    def _rewrite(self):
        all_records = []
        if AB_LOG.exists():
            with open(AB_LOG, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        if r.get("model") != self.name:
                            all_records.append(r)
        all_records.extend(self._bets)
        with open(AB_LOG, "w", encoding="utf-8") as f:
            for r in all_records:
                f.write(json.dumps(r, default=str) + "\n")
            f.flush()


# ── 特徵建構 ──────────────────────────────────────────────────

def build_features(home_abbr, away_abbr, history, feat_cols,
                   home_q1=0, away_q1=0, home_q2=0, away_q2=0):
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
            col  = f"{side}_{stat}"
            vals = (history[history[f"{side}_team"] == team]
                    .tail(20)[col].dropna() if col in history.columns else pd.Series())
            feat[col] = float(vals.mean()) if len(vals) else 100.0
    feat["pace_matchup"] = (feat["home_pace"] + feat["away_pace"]) / 2

    today = pd.Timestamp(date.today())
    for side, team in [("home", home_abbr), ("away", away_abbr)]:
        mask = (history["home_team"] == team) | (history["away_team"] == team)
        last = history[mask]["game_date"].max()
        days = (today - pd.Timestamp(last)).days if pd.notna(last) else 3
        feat[f"{side}_rest_days"] = min(int(days), 14)
        feat[f"{side}_b2b"]       = 1 if days == 1 else 0

    q1_total = home_q1 + away_q1
    h1_total = home_q1 + away_q1 + home_q2 + away_q2
    implied  = feat["implied_total_10"]
    feat.update({
        "home_q1": home_q1, "home_q2": home_q2,
        "away_q1": away_q1, "away_q2": away_q2,
        "q1_total": q1_total, "h1_total": h1_total,
        "q2_total": home_q2 + away_q2,
        "q1_pace_ratio": q1_total / (implied/4) if implied > 0 else 1,
        "h1_pace_ratio": h1_total / (implied/2) if implied > 0 else 1,
        "q1_home_lead": home_q1 - away_q1,
        "h1_home_lead": (home_q1+home_q2) - (away_q1+away_q2),
        "q1_vs_avg": q1_total - (implied/4),
    })

    row = pd.Series(feat)
    for c in feat_cols:
        if c not in row.index:
            row[c] = 0
    return row[feat_cols].to_frame().T.fillna(0)


# ── 主模擬器 ──────────────────────────────────────────────────

class ABSimulator:
    def __init__(self, capital: float = 10000.0):
        self.pregame  = SimAccount("pregame",  capital)
        self.halftime = SimAccount("halftime", capital)

        self.xgb_half = pickle.load(open(MODEL_DIR / "xgb.pkl",         "rb"))
        self.lgb_half = pickle.load(open(MODEL_DIR / "lgb.pkl",         "rb"))
        self.xgb_pre  = pickle.load(open(MODEL_DIR / "pregame_xgb.pkl", "rb"))
        self.lgb_pre  = pickle.load(open(MODEL_DIR / "pregame_lgb.pkl", "rb"))

        self.history    = pd.read_csv(FEAT_DIR / "features.csv", parse_dates=["game_date"])
        self.half_feats = list(self.xgb_half.feature_names_in_)
        self.pre_feats  = list(self.xgb_pre.feature_names_in_)
        log.info(f"ABSimulator ready | pregame=${capital:,.0f} halftime=${capital:,.0f}")

    def _predict(self, xgb_m, lgb_m, feat_df) -> float:
        return round((float(xgb_m.predict(feat_df)[0]) +
                      float(lgb_m.predict(feat_df)[0])) / 2, 1)

    def _get_markets(self, home_name: str, away_name: str) -> list:
        from src.scraper.polymarket_scraper import fetch_nba_game_events, parse_total_markets
        from src.live.polymarket_trader import PolymarketTrader
        events  = fetch_nba_game_events()
        markets = parse_total_markets(events)
        h, a    = home_name.lower(), away_name.lower()
        matched = [m for m in markets
                   if (h in m["home_team"].lower() or h in m["away_team"].lower())
                   and (a in m["away_team"].lower() or a in m["home_team"].lower())]

        # 用 CLOB orderbook ask price 取代 Gamma mid（更準確）
        try:
            trader = PolymarketTrader(dry_run=True)
            if trader._client:
                for m in matched:
                    for outcome, prob_key, opp_key in [
                        ("Over",  "over_prob",  "under_prob"),
                        ("Under", "under_prob", "over_prob"),
                    ]:
                        token_id = trader.get_token_id(m["condition_id"], outcome)
                        if not token_id:
                            continue
                        book = trader._client.get_order_book(token_id)
                        if not book:
                            continue
                        asks = book.get("asks", []) if isinstance(book, dict) else (getattr(book, "asks", []) or [])
                        if asks:
                            a0       = asks[0]
                            ask_price = float(a0.get("price", 0) if isinstance(a0, dict) else getattr(a0, "price", 0))
                            if ask_price > 0:
                                m[prob_key] = ask_price
                                m[opp_key]  = round(1 - ask_price, 4)
        except Exception as e:
            log.debug(f"CLOB ask 取得失敗，使用 Gamma mid：{e}")

        return matched

    def _place_bets(self, account: SimAccount, markets: list,
                    pred: float, std: float, game: str, trigger: str):
        """
        每場比賽每個方向只下最佳的一條線（EV 最高的）。
        避免同場多條線高度相關的重複風險。
        """
        # 全場只選單一最佳注（EV 最高），與 pregame 邏輯一致
        best = None  # {"side", "line", "model_p", "market_p", "ev"}

        for m in markets:
            line        = m["line"]
            mkt_over    = m["over_prob"]
            mkt_under   = m["under_prob"]
            model_over  = float(1 - norm.cdf(line, loc=pred, scale=std))
            model_under = 1 - model_over

            for side, model_p, market_p in [
                ("OVER",  model_over,  mkt_over),
                ("UNDER", model_under, mkt_under),
            ]:
                # 只在模型預測方向下注
                if side == "OVER"  and pred <= line:
                    continue
                if side == "UNDER" and pred >= line:
                    continue

                # 過濾極端市場價格
                if market_p < 0.10 or market_p > 0.90:
                    continue

                ev   = model_p - market_p
                edge = model_p - market_p
                if ev >= MIN_EV and edge >= MIN_EDGE:
                    if best is None or ev > best["ev"]:
                        best = {
                            "side": side, "line": line,
                            "model_p": model_p, "market_p": market_p, "ev": ev,
                        }

        if best is not None:
            account.bet(game, best["line"], best["side"],
                        best["model_p"], best["market_p"], pred, trigger)

    def on_pregame(self, away_name: str, home_name: str):
        home_abbr = TEAM_NAME_MAP.get(home_name, home_name[:3].upper())
        away_abbr = TEAM_NAME_MAP.get(away_name, away_name[:3].upper())
        feat_df   = build_features(home_abbr, away_abbr, self.history, self.pre_feats)
        pred_raw  = self._predict(self.xgb_pre, self.lgb_pre, feat_df)
        pred = pred_raw
        log.info(f"[PREGAME] {away_name} @ {home_name} | pred={pred}")
        markets = self._get_markets(home_name, away_name)
        if not markets:
            log.warning(f"  No Polymarket markets for {away_name} @ {home_name}")
            return
        self._place_bets(self.pregame, markets, pred, PRED_STD_PREGAME,
                         f"{away_name} @ {home_name}", "pregame")

    def on_halftime(self, game_dict: dict):
        # ESPN 原始縮寫 → 標準縮寫 → 全名
        away_abbr = ESPN_ABBR_FIX.get(game_dict["away_team"], game_dict["away_team"])
        home_abbr = ESPN_ABBR_FIX.get(game_dict["home_team"], game_dict["home_team"])
        away_name = ABBR_TO_FULL.get(away_abbr, away_abbr)
        home_name = ABBR_TO_FULL.get(home_abbr, home_abbr)
        feat_df   = build_features(
            home_abbr, away_abbr, self.history, self.half_feats,
            home_q1=game_dict["home_q1"], away_q1=game_dict["away_q1"],
            home_q2=game_dict["home_q2"], away_q2=game_dict["away_q2"],
        )
        pred = self._predict(self.xgb_half, self.lgb_half, feat_df)
        h1   = (game_dict["home_q1"]+game_dict["home_q2"],
                game_dict["away_q1"]+game_dict["away_q2"])
        log.info(f"[HALFTIME] {away_name} @ {home_name} | H1={h1[1]}-{h1[0]} | pred={pred}")
        markets = self._get_markets(home_name, away_name)
        if not markets:
            log.warning(f"  No Polymarket markets for {away_name} @ {home_name}")
            return
        self._place_bets(self.halftime, markets, pred, PRED_STD_HALFTIME,
                         f"{away_name} @ {home_name}", "halftime")

    def print_report(self):
        ps = self.pregame.stats()
        hs = self.halftime.stats()
        print(f"\n{'═'*62}")
        print(f"{'  A/B Simulation Report':^62}")
        print(f"{'═'*62}")
        print(f"  {'指標':<20}  {'賽前模型':>16}  {'半場模型':>16}")
        print(f"  {'─'*58}")

        def fmt(s, key, pct=False, dollar=False):
            v = s.get(key, 0)
            if pct:    return f"{v:>+.1%}"
            if dollar: return f"${v:>,.0f}"
            return str(v)

        for label, key, pct, dollar in [
            ("下注場次", "n",         False, False),
            ("勝率",     "win_rate",  True,  False),
            ("總損益",   "total_pnl", False, True),
            ("ROI",      "roi",       True,  False),
            ("當前資金", "capital",   False, True),
            ("總報酬率", "gain_pct",  True,  False),
        ]:
            pv = fmt(ps, key, pct, dollar)
            hv = fmt(hs, key, pct, dollar)
            if key in ("win_rate","roi","gain_pct","total_pnl"):
                try:
                    pv += " ★" if float(ps.get(key,0)) > float(hs.get(key,0)) else ""
                    hv += " ★" if float(hs.get(key,0)) > float(ps.get(key,0)) else ""
                except Exception:
                    pass
            print(f"  {label:<20}  {pv:>16}  {hv:>16}")
        print(f"{'═'*62}\n")
        self._print_bets_detail()

    def _print_bets_detail(self):
        """印出 ab_simulation.jsonl 的完整下注明細，按日期分組"""
        import json
        from collections import defaultdict

        if not AB_LOG.exists():
            print("  (找不到 ab_simulation.jsonl)\n")
            return

        rows = []
        with open(AB_LOG, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass

        if not rows:
            print("  (ab_simulation.jsonl 無記錄)\n")
            return

        # 標題列
        HDR = (f"  {'模型':<10} {'比賽':<26} {'線':>6} {'方向':<6} "
               f"{'股價':>5} {'下注':>7} {'EV':>6} {'預測':>5} {'實際':>5} {'結果':<6} {'損益':>9}")
        SEP = f"  {'─'*94}"

        def print_row(r, result_str=None):
            ev_str     = f"{r.get('ev', 0):>+.1%}"
            pred_str   = f"{r['pred']:.0f}"      if r.get("pred")         else "?"
            actual_str = f"{r['actual_total']:.0f}" if r.get("actual_total") is not None else "?"
            price_str  = f"{r.get('market_p',0):.2f}"
            pnl_str    = (f"${r['pnl']:>+,.2f}"  if r.get("pnl") is not None else "待結算")
            rs         = result_str or ("WIN ✅" if r.get("result") == "win" else
                                        "LOSS ❌" if r.get("result") == "loss" else "待結算")
            print(f"  {r.get('model',''):<10} {r.get('game',''):<26} "
                  f"{r.get('line',0):>6.1f} {r.get('side',''):<6} "
                  f"{price_str:>5} ${r.get('amount',0):>6,.0f} "
                  f"{ev_str:>6} {pred_str:>5} {actual_str:>5} {rs:<6} {pnl_str:>9}")

        def day_summary(day_rows):
            settled = [r for r in day_rows if r.get("result")]
            pending = [r for r in day_rows if not r.get("result")]
            wins    = sum(1 for r in settled if r["result"] == "win")
            pnl     = sum(r.get("pnl", 0) or 0 for r in settled)
            bet     = sum(r.get("amount", 0) for r in settled)
            roi     = pnl / bet if bet > 0 else 0
            return settled, pending, wins, pnl, bet, roi

        # 按日期分組
        by_date = defaultdict(list)
        for r in rows:
            by_date[r.get("date", "unknown")].append(r)

        all_settled, all_wins, all_pnl, all_bet = [], 0, 0, 0

        for day in sorted(by_date.keys()):
            day_rows = by_date[day]
            settled, pending, wins, pnl, bet, roi = day_summary(day_rows)
            all_settled += settled
            all_wins    += wins
            all_pnl     += pnl
            all_bet     += bet

            print(f"\n{'═'*98}")
            print(f"  📅 {day}　　{len(settled)} 注已結算 {len(pending)} 注待結算　"
                  f"勝 {wins} 負 {len(settled)-wins}　"
                  f"損益 ${pnl:>+,.2f}　ROI {roi:>+.1%}")
            print(f"{'═'*98}")
            print(HDR)
            print(SEP)

            for r in day_rows:
                print_row(r)

            if settled:
                pre_pnl  = sum(r.get("pnl",0) or 0 for r in settled if r["model"]=="pregame")
                half_pnl = sum(r.get("pnl",0) or 0 for r in settled if r["model"]=="halftime")
                print(f"  {'─'*94}")
                print(f"  賽前 ${pre_pnl:>+,.2f}　半場 ${half_pnl:>+,.2f}")

        # 全部總結
        if len(by_date) > 1:
            all_roi = all_pnl / all_bet if all_bet > 0 else 0
            print(f"\n{'═'*98}")
            print(f"  📊 全期總結　{len(all_settled)} 注已結算　"
                  f"勝 {all_wins} 負 {len(all_settled)-all_wins}　"
                  f"勝率 {all_wins/len(all_settled):.1%}　"
                  f"總損益 ${all_pnl:>+,.2f}　ROI {all_roi:>+.1%}")
            print(f"{'═'*98}\n")
        else:
            print(f"{'─'*98}\n")

    def print_status(self):
        print(f"\n  [賽前模型]  資金=${self.pregame.total:,.0f}  "
              f"可用=${self.pregame.available:,.0f}")
        print(f"  [半場模型]  資金=${self.halftime.total:,.0f}  "
              f"可用=${self.halftime.available:,.0f}\n")


# ── 進入點 ────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live",    action="store_true")
    parser.add_argument("--status",  action="store_true")
    parser.add_argument("--report",  action="store_true")
    parser.add_argument("--capital", type=float, default=10000)
    args = parser.parse_args()

    sim = ABSimulator(capital=args.capital)

    if args.status:
        sim.print_status()

    elif args.report:
        sim.print_report()

    elif args.live:
        from src.scraper.live_scraper import poll
        import requests

        # Thread 1：偵測比賽結束 → 立刻結算
        def settle_loop():
            settled_ids = set()
            while True:
                try:
                    r = requests.get(
                        "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
                        timeout=10, proxies={}
                    )
                    for ev in r.json().get("events", []):
                        if ev["status"]["type"]["state"] != "post":
                            continue
                        gid = ev["id"]
                        if gid in settled_ids:
                            continue
                        comp  = ev["competitions"][0]
                        home  = next(c for c in comp["competitors"] if c["homeAway"] == "home")
                        away  = next(c for c in comp["competitors"] if c["homeAway"] == "away")
                        total = int(home.get("score", 0)) + int(away.get("score", 0))
                        ha    = ESPN_ABBR_FIX.get(home["team"]["abbreviation"], home["team"]["abbreviation"])
                        aa    = ESPN_ABBR_FIX.get(away["team"]["abbreviation"], away["team"]["abbreviation"])
                        key   = f"{ABBR_TO_FULL.get(aa, aa)} @ {ABBR_TO_FULL.get(ha, ha)}"
                        pending = [b for b in sim.pregame._bets + sim.halftime._bets
                                   if b["game"] == key and b["result"] is None]
                        if pending:
                            log.info(f"Game finished: {key} total={total} → settling {len(pending)} bets")
                            lines = set(b["line"] for b in pending)
                            for line in lines:
                                sim.pregame.settle(key, line, total)
                                sim.halftime.settle(key, line, total)
                            sim.print_report()
                        settled_ids.add(gid)
                except Exception as e:
                    log.warning(f"Settle loop error: {e}")
                time.sleep(300)

        # Thread 2：偵測賽前比賽 → 賽前模型下注
        def pregame_loop():
            triggered = set()
            while True:
                try:
                    r = requests.get(
                        "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
                        timeout=10, proxies={}
                    )
                    for ev in r.json().get("events", []):
                        if ev["status"]["type"]["state"] != "pre":
                            continue
                        gid = ev["id"]
                        if gid in triggered:
                            continue
                        triggered.add(gid)
                        comp      = ev["competitions"][0]
                        home      = next(c for c in comp["competitors"] if c["homeAway"] == "home")
                        away      = next(c for c in comp["competitors"] if c["homeAway"] == "away")
                        ha        = ESPN_ABBR_FIX.get(home["team"]["abbreviation"], home["team"]["abbreviation"])
                        aa        = ESPN_ABBR_FIX.get(away["team"]["abbreviation"], away["team"]["abbreviation"])
                        away_full = ABBR_TO_FULL.get(aa, aa)
                        home_full = ABBR_TO_FULL.get(ha, ha)
                        sim.on_pregame(away_full, home_full)
                except Exception as e:
                    log.warning(f"Pregame loop error: {e}")
                time.sleep(300)

        threading.Thread(target=settle_loop,  daemon=True).start()
        threading.Thread(target=pregame_loop, daemon=True).start()
        log.info("Live mode started — pregame + halftime A/B simulation")
        log.info(f"  Log file: {_LOG_FILE}")
        log.info(f"  Pregame capital:  ${sim.pregame.total:,.0f}")
        log.info(f"  Halftime capital: ${sim.halftime.total:,.0f}")

        # 主執行緒：偵測半場結束 → 半場模型下注
        # 同時追蹤今天是否所有比賽都結束
        def on_trigger(game: dict, quarter: int):
            if quarter == 2:
                sim.on_halftime(game)
                sim.print_status()

        # 用 dict 取代 nonlocal（Python 的 nonlocal 只能用在 nested function 裡）
        _state = {"started": False, "consecutive_done": 0}

        def check_all_done(games: list) -> bool:
            live_count     = sum(1 for g in games if g["status"] == "in")
            finished_count = sum(1 for g in games if g["status"] == "post")
            pre_count      = sum(1 for g in games if g["status"] == "pre")

            if live_count > 0 or pre_count > 0:
                _state["started"] = True
                _state["consecutive_done"] = 0
                return False

            if _state["started"] and finished_count > 0 and live_count == 0 and pre_count == 0:
                _state["consecutive_done"] += 1
                if _state["consecutive_done"] >= 5:
                    return True
            else:
                _state["consecutive_done"] = 0

            return False

        # 自訂 poll 迴圈（加入自動結束判斷）
        import time as _time
        from src.scraper.live_scraper import QuarterTracker, fetch_scoreboard, POLL_INTERVAL
        tracker = QuarterTracker()
        log.info("Live polling started (interval=60s, trigger quarters=[1, 2])")

        while True:
            from datetime import datetime as _dt
            now   = _dt.now().strftime("%H:%M:%S")
            games = fetch_scoreboard()

            if not games:
                log.info(f"[{now}] No games found or API error, retrying...")
            else:
                live  = [g for g in games if g["status"] == "in"]
                done  = [g for g in games if g["status"] == "post"]
                pre   = [g for g in games if g["status"] == "pre"]

                log.info(f"[{now}] Games: {len(pre)} upcoming / "
                         f"{len(live)} live / {len(done)} finished")

                from src.scraper.live_scraper import _log_game
                for game in live:
                    _log_game(game)
                    triggered_quarter = tracker.check(game)
                    if triggered_quarter:
                        log.info(
                            f"  ★ TRIGGER Q{triggered_quarter} END: "
                            f"{game['away_team']} @ {game['home_team']}  "
                            f"score={game['away_total']}-{game['home_total']}"
                        )
                        on_trigger(game, triggered_quarter)

                for game in done:
                    tracker.reset_game(game["game_id"])

                # 檢查是否全部結束
                if check_all_done(games):
                    log.info("All games finished for today. Shutting down...")
                    sim.print_report()
                    sys.exit(0)

            _time.sleep(POLL_INTERVAL)