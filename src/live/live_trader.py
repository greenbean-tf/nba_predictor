"""
live_trader.py
--------------
真實資金的 NBA total 下注系統。
與 ab_simulator.py 完全獨立，不共用 log 或資料。

使用方式：
  python src/live/live_trader.py --help
  python src/live/live_trader.py --mode both --capital 100 --kelly 0.1
  python src/live/live_trader.py --mode halftime --capital 50 --min-ev 0.05
  python src/live/live_trader.py --status
  python src/live/live_trader.py --report

參數說明：
  --mode         pregame / halftime / both（預設 both）
  --capital      總資金 USDC（預設 100）
  --kelly        Kelly 比例 0~1（預設 0.25 = quarter Kelly）
  --min-ev       最小 EV 門檻（預設 0.03）
  --min-edge     最小 edge 門檻（預設 0.03）
  --max-bet      單注上限 USDC（預設 500）
  --max-pct      單注最大占總資金比例（預設 0.20 = 20%）
  --pregame-std  賽前模型預測標準差（預設 13.0）
  --halftime-std 半場模型預測標準差（預設 9.0）
  --dry-run      模擬模式，不實際下注（預設）
  --live         真實下注模式（需要明確指定）
"""

import sys
import os
import json
import logging
import argparse
import threading
import time
import pickle
from pathlib import Path
from datetime import datetime, date

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from scipy.stats import norm
import yaml
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

# ── 設定 ──────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent.parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

LOG_DIR   = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

TRADER_LOG  = LOG_DIR / "live_trader.log"
TRADER_JSONL_PREGAME  = LOG_DIR / "live_trades_pregame.jsonl"
TRADER_JSONL_HALFTIME = LOG_DIR / "live_trades_halftime.jsonl"

# ── Logging ───────────────────────────────────────────────────

def setup_logging():
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level   = logging.INFO,
        format  = fmt,
        datefmt = "%H:%M:%S",
        handlers = [
            logging.FileHandler(TRADER_LOG, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force = True,
    )

log = logging.getLogger(__name__)

# ── 從 ab_simulator 引用共用邏輯 ─────────────────────────────

from src.live.ab_simulator import (
    TEAM_NAME_MAP, ABBR_TO_FULL, ESPN_ABBR_FIX,
    build_features,
)
from src.scraper.polymarket_scraper import fetch_nba_game_events, parse_total_markets
from src.scraper.live_scraper import fetch_scoreboard, QuarterTracker, POLL_INTERVAL


# ── 設定類 ────────────────────────────────────────────────────

class TraderConfig:
    def __init__(self, args):
        self.mode         = args.mode
        self.kelly        = args.kelly
        self.min_ev       = args.min_ev
        self.min_edge     = args.min_edge
        self.max_bet      = args.max_bet
        self.max_pct      = args.max_pct
        self.pregame_std  = args.pregame_std
        self.halftime_std = args.halftime_std
        self.live         = args.live

        # 各自獨立資金
        total = args.capital
        if self.mode == "both":
            self.pregame_capital  = args.pregame_capital  if args.pregame_capital  else total / 2
            self.halftime_capital = args.halftime_capital if args.halftime_capital else total / 2
        elif self.mode == "pregame":
            self.pregame_capital  = total
            self.halftime_capital = 0.0
        else:
            self.pregame_capital  = 0.0
            self.halftime_capital = total

    @property
    def total_capital(self):
        return self.pregame_capital + self.halftime_capital

    def __str__(self):
        mode_str = "🔴 LIVE" if self.live else "🟡 DRY-RUN"
        lines = [
            f"\n{'═'*55}",
            f"  Live Trader 設定 [{mode_str}]",
            f"{'═'*55}",
            f"  模式         : {self.mode}",
        ]
        if self.mode == "both":
            lines += [
                f"  賽前資金     : ${self.pregame_capital:,.2f} USDC",
                f"  半場資金     : ${self.halftime_capital:,.2f} USDC",
                f"  總資金       : ${self.total_capital:,.2f} USDC",
            ]
        else:
            lines.append(f"  資金         : ${self.total_capital:,.2f} USDC")
        lines += [
            f"  Kelly 比例   : {self.kelly:.0%}",
            f"  最小 EV      : {self.min_ev:.1%}",
            f"  最小 Edge    : {self.min_edge:.1%}",
            f"  單注上限     : ${self.max_bet:.0f}",
            f"  單注最大占比 : {self.max_pct:.0%}",
            f"  賽前標準差   : {self.pregame_std}",
            f"  半場標準差   : {self.halftime_std}",
            f"{'═'*55}",
        ]
        return "\n".join(lines)


# ── 資金帳戶 ──────────────────────────────────────────────────

class TradingAccount:
    """管理真實資金的帳戶，記錄到 live_trades.jsonl"""

    def __init__(self, cfg: TraderConfig, name: str, capital: float):
        self.cfg     = cfg
        self.name    = name          # "pregame" or "halftime"
        self.total   = capital
        self.initial = capital
        self._trades: list[dict] = []
        self._load_existing()

    @property
    def _jsonl_path(self) -> Path:
        return TRADER_JSONL_PREGAME if self.name == "pregame" else TRADER_JSONL_HALFTIME

    def _load_existing(self):
        if not self._jsonl_path.exists():
            return
        with open(self._jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self._trades.append(json.loads(line))
                    except Exception:
                        pass
        log.info(f"[{self.name}] 載入 {len(self._trades)} 筆歷史交易（資金以 Polymarket 實際餘額為準）")

    @property
    def available(self) -> float:
        """扣除未結算下注後的可用資金"""
        pending = sum(
            t.get("amount", 0)
            for t in self._trades
            if t.get("result") is None
        )
        return max(0, self.total - pending)

    def kelly_amount(self, model_p: float, market_p: float,
                     real_balance: float = None) -> float:
        """計算 Kelly 下注金額，優先使用 Polymarket 實際餘額"""
        edge = model_p - market_p
        f    = (edge / (1 - market_p)) * self.cfg.kelly
        f    = max(0, min(f, self.cfg.max_pct))
        # 用實際餘額（如果有），否則用模擬 available
        base = real_balance if real_balance is not None else self.available
        amt  = base * f
        amt  = min(amt, self.cfg.max_bet)
        return round(amt, 2)

    def record_bet(self, record: dict):
        """記錄下注"""
        self._trades.append(record)
        with open(self._jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
            f.flush()

    def settle(self, game: str, line: float, actual_total: float):
        """結算比賽"""
        changed = False
        for t in self._trades:
            if (t.get("game") == game and
                t.get("line") == line and
                t.get("result") is None):
                won = (actual_total > line) if t["side"] == "OVER" else (actual_total < line)
                # 用實際成交價計算損益（必須有 filled_price，否則用 market_p 但標記警告）
                fp = t.get("filled_price")
                if fp is None:
                    fp = t.get("market_p", 0.5)
                    log.warning(f"  結算用 market_p={fp:.4f}（filled_price 未記錄，可能不準確）")
                pnl = round((t["amount"] / fp - t["amount"]), 2) if won else -t["amount"]
                t["result"]       = "win" if won else "loss"
                t["pnl"]          = round(pnl, 2)
                t["actual_total"] = actual_total
                # 不自動更新 total，資金以 Polymarket 實際餘額為準
                log.info(
                    f"  [{'WIN ✅' if won else 'LOSS ❌'}] {game} {line} {t['side']} "
                    f"${t['amount']:.0f} pnl={pnl:+.2f} | order={t.get('order_id','?')}"
                )
                changed = True
        if changed:
            self._rewrite()

    def _rewrite(self):
        with open(self._jsonl_path, "w", encoding="utf-8") as f:
            for t in self._trades:
                f.write(json.dumps(t, default=str) + "\n")


# ── 主要交易器 ────────────────────────────────────────────────

class LiveTrader:

    def __init__(self, cfg: TraderConfig):
        self.cfg              = cfg
        self.pregame_account  = TradingAccount(cfg, "pregame",  cfg.pregame_capital)
        self.halftime_account = TradingAccount(cfg, "halftime", cfg.halftime_capital)
        self._trader          = None
        self._load_models()

    def _load_models(self):
        MODEL_DIR = BASE_DIR / "models" / "saved"
        FEAT_DIR  = BASE_DIR / CFG["data"]["features_dir"]

        self.xgb_pre  = pickle.load(open(MODEL_DIR / "pregame_xgb.pkl", "rb"))
        self.lgb_pre  = pickle.load(open(MODEL_DIR / "pregame_lgb.pkl", "rb"))
        self.xgb_half = pickle.load(open(MODEL_DIR / "xgb.pkl", "rb"))
        self.lgb_half = pickle.load(open(MODEL_DIR / "lgb.pkl", "rb"))

        self.pre_feats  = list(self.xgb_pre.feature_names_in_)
        self.half_feats = list(self.xgb_half.feature_names_in_)
        self.history    = pd.read_csv(FEAT_DIR / "features.csv", parse_dates=["game_date"])

        log.info("模型載入完成")

    def _get_trader(self):
        if self._trader is None:
            from src.live.polymarket_trader import PolymarketTrader
            self._trader = PolymarketTrader(dry_run=not self.cfg.live)
        return self._trader

    def _predict(self, xgb_m, lgb_m, feat_df) -> float:
        p_xgb = float(xgb_m.predict(feat_df)[0])
        p_lgb = float(lgb_m.predict(feat_df)[0])
        return round((p_xgb + p_lgb) / 2, 1)

    def _get_markets(self, home_name: str, away_name: str) -> list:
        events  = fetch_nba_game_events()
        markets = parse_total_markets(events)
        h, a    = home_name.lower(), away_name.lower()
        matched = [m for m in markets
                   if (h in m["home_team"].lower() or h in m["away_team"].lower())
                   and (a in m["away_team"].lower() or a in m["home_team"].lower())]

        # 完全使用 CLOB ask price，不使用 Gamma mid
        # Gamma 的 over_prob/under_prob 先清空，必須由 CLOB 填入
        for m in matched:
            m["over_prob"]  = None
            m["under_prob"] = None

        trader = self._get_trader()
        valid_markets = []

        if trader and trader._client:
            for m in matched:
                clob_ok = True
                for outcome, token_key, prob_key, opp_key in [
                    ("Over",  "token_id_over",  "over_prob",  "under_prob"),
                    ("Under", "token_id_under", "under_prob", "over_prob"),
                ]:
                    MAX_RETRIES = 3
                    RETRY_WAIT  = 5
                    ask_price   = None

                    for attempt in range(1, MAX_RETRIES + 1):
                        try:
                            token_id = m.get(token_key) or trader.get_token_id(m["condition_id"], outcome)
                            if token_id:
                                m[token_key] = token_id
                                book = trader._client.get_order_book(token_id)
                                if book:
                                    asks = book.get("asks", []) if isinstance(book, dict) else (getattr(book, "asks", []) or [])
                                    if asks:
                                        # asks 從高到低，最後一個是 best ask
                                        a_best    = asks[-1]
                                        ask_price = float(a_best.get("price", 0) if isinstance(a_best, dict) else getattr(a_best, "price", 0))
                                        if ask_price > 0:
                                            break
                        except Exception as e:
                            log.debug(f"  CLOB ask 嘗試 {attempt}/{MAX_RETRIES}（{outcome}）失敗：{e}")

                        if attempt < MAX_RETRIES:
                            time.sleep(RETRY_WAIT)

                    if ask_price and ask_price > 0:
                        m[prob_key] = ask_price
                        m[opp_key]  = round(1 - ask_price, 4)
                    else:
                        log.warning(f"  CLOB ask 取得失敗（{m.get('question','')} {outcome}），排除此市場")
                        clob_ok = False
                        break

                if clob_ok:
                    # 確認 ask 有足夠流動性（asks 數量 >= 3）
                    token_id = m.get("token_id_over")
                    if token_id:
                        try:
                            book = trader._client.get_order_book(token_id)
                            asks = book.get("asks", []) if isinstance(book, dict) else (getattr(book, "asks", []) or [])
                            if len(asks) < 3:
                                log.info(f"  流動性不足（asks={len(asks)}），排除 {m.get('question','')}")
                                clob_ok = False
                        except:
                            pass

                if clob_ok:
                    valid_markets.append(m)

        return valid_markets

    def _find_best_bet(self, markets: list, pred: float, std: float) -> dict | None:
        """找出 EV 最高的一注（方向一致 + 市場價格合理）"""
        best = None
        for m in markets:
            line       = m["line"]
            model_over = float(1 - norm.cdf(line, loc=pred, scale=std))

            for side, model_p, market_p in [
                ("OVER",  model_over,     m["over_prob"]),
                ("UNDER", 1 - model_over, m["under_prob"]),
            ]:
                # CLOB ask 未取得（None）→ 跳過
                if market_p is None: continue

                # 方向一致
                if side == "OVER"  and pred <= line: continue
                if side == "UNDER" and pred >= line: continue

                # 市場價格極端過濾（CLOB ask 超出合理範圍）
                if market_p < 0.10 or market_p > 0.90: continue

                # model_p 極端過濾（模型過度自信，不可靠）
                if model_p > 0.95 or model_p < 0.05: continue

                # pred 跟 line 差距過大過濾（超過 2.5 個標準差，模型可能失準）
                if abs(pred - line) > 2.5 * std: continue

                edge = model_p - market_p
                ev   = model_p / market_p - 1   # 真實 EV（賠率意義）
                if ev >= self.cfg.min_ev and edge >= self.cfg.min_edge:
                    if best is None or ev > best["ev"]:
                        best = {
                            "line": line, "side": side,
                            "model_p": model_p, "market_p": market_p,
                            "edge": round(edge, 4),
                            "ev": round(ev, 4),
                            "condition_id": m["condition_id"],
                        }
        return best

    def _place_bet(self, game: str, pred: float, best: dict, trigger: str):
        account  = self.pregame_account if trigger == "pregame" else self.halftime_account
        trader   = self._get_trader()
        mode_str = "🔴 LIVE" if self.cfg.live else "🟡 DRY-RUN"

        MAX_RETRY   = 10
        RETRY_DELAY = 20   # 秒

        for attempt in range(1, MAX_RETRY + 1):
            # 每次重試都重新計算（市場可能已移動）
            # 查詢 Polymarket 實際餘額，確保不超出可用資金
            real_balance = None
            try:
                real_balance = trader.get_balance()
                if real_balance <= 0:
                    log.warning(f"  Polymarket 餘額 ${real_balance:.2f}，跳過下注")
                    return
                log.info(f"  Polymarket 餘額：${real_balance:.2f}")
            except Exception as e:
                log.warning(f"  無法查詢餘額（{e}），使用模擬帳戶計算")

            amount = account.kelly_amount(best["model_p"], best["market_p"],
                                          real_balance=real_balance)
            if amount < 1.0:
                log.info(f"  下注金額 ${amount:.2f} 過小，跳過")
                return

            # 取得 token_id
            token_id = trader.get_token_id(best["condition_id"], best["side"].capitalize())
            if not token_id:
                log.warning(f"  找不到 token_id，跳過")
                return

            attempt_str = f" (第{attempt}次)" if attempt > 1 else ""
            log.info(
                f"  [{trigger}] {mode_str} BET{attempt_str} {game} "
                f"{best['line']} {best['side']} ${amount:.2f} | "
                f"EV={best['ev']:+.1%} | avail=${account.available:.2f}"
            )

            # 下單
            result = trader.place_bet(
                token_id     = token_id,
                outcome      = best["side"],
                amount_usdc  = amount,
                market_price = best["market_p"],
            )

            order_success = result.get("success", False)
            order_id      = result.get("order_id")
            order_error   = result.get("error")
            actual_price  = result.get("actual_price")   # 實際成交價（FOK 成交後從 get_trades 取得）
            actual_size   = result.get("actual_size")

            # 判斷訂單是否實際成交
            trade_success = False
            trade_status  = None
            if order_success and order_id and order_id != "DRY_RUN":
                if actual_price is not None:
                    # polymarket_trader 已查到實際成交 → 代表 MATCHED
                    trade_success = True
                    trade_status  = "MATCHED"
                    log.info(f"  訂單成交 ✅ price={actual_price:.4f} size={actual_size:.4f} | order={order_id}")
                else:
                    # 備用：直接查訂單狀態
                    try:
                        order_info = trader._client.get_order(order_id)
                        if order_info and isinstance(order_info, dict):
                            trade_status  = order_info.get("status", "UNKNOWN")
                            trade_success = trade_status == "MATCHED"
                            log.info(f"  訂單狀態：{trade_status} | order_id={order_id}")
                        else:
                            trade_status  = "UNKNOWN"
                            trade_success = order_success
                            log.warning(f"  get_order 回傳空值，以下單成功為準")
                    except Exception as e:
                        log.warning(f"  無法查詢訂單狀態：{e}")
                        trade_status  = "UNKNOWN"
                        trade_success = order_success
            elif order_id == "DRY_RUN":
                trade_success = True
                trade_status  = "DRY_RUN"
                actual_price  = best["market_p"]
                actual_size   = round(amount / best["market_p"], 4)

            # 記錄這次嘗試
            record = {
                "account":       account.name,
                "date":          str(date.today()),
                "ts":            datetime.now().isoformat(),
                "trigger":       trigger,
                "game":          game,
                "line":          best["line"],
                "side":          best["side"],
                "pred":          round(pred, 1),
                "model_p":       round(best["model_p"], 4),
                "market_p":      round(best["market_p"], 4),
                "filled_price":  round(actual_price, 4) if actual_price else None,
                "edge":          round(best.get("edge", best["model_p"] - best["market_p"]), 4),
                "ev":            round(best["ev"], 4),
                "amount":        round(amount, 2),
                "capital":       round(account.total, 2),
                "order_id":      order_id,
                "order_success": order_success,
                "trade_status":  trade_status,
                "trade_success": trade_success,
                "attempt":       attempt,
                "dry_run":       not self.cfg.live,
                "result":        None,
                "pnl":           None,
                "actual_total":  None,
            }

            if trade_success:
                log.info(f"  交易成功 ✅ | order={order_id} | ${amount:.2f}")
                account.record_bet(record)
                return
            elif order_success and not trade_success:
                # 下單 API 成功但訂單未成交（FOK 取消）
                retry_msg = f"，{RETRY_DELAY}秒後重試 ({attempt}/{MAX_RETRY})" if attempt < MAX_RETRY else "，已達重試上限"
                log.warning(f"  下單送出但未成交（{trade_status}，流動性不足）{retry_msg}")
                record["order_success"] = True
                record["trade_success"] = False
                record["trade_status"]  = f"FOK_CANCELED: {trade_status}"
            else:
                # 下單 API 失敗
                retry_msg = f"，{RETRY_DELAY}秒後重試 ({attempt}/{MAX_RETRY})" if attempt < MAX_RETRY else "，已達重試上限"
                log.warning(f"  下單失敗：{order_error}{retry_msg}")
                record["order_success"] = False
                record["trade_success"] = False
                record["trade_status"]  = f"ORDER_FAILED: {order_error}"

            if attempt < MAX_RETRY:
                time.sleep(RETRY_DELAY)
                # 重新抓市場賠率（價格可能已移動）
                markets = self._get_markets(
                    game.split(" @ ")[1], game.split(" @ ")[0]
                )
                if markets:
                    new_best = self._find_best_bet(markets, pred,
                        self.cfg.pregame_std if trigger == "pregame" else self.cfg.halftime_std)
                    if new_best:
                        best = new_best
                    else:
                        log.info("  重試時已無正 EV，放棄")
                        return

        log.warning(f"  達到最大重試次數 {MAX_RETRY}，放棄下注")

    # ── pregame ───────────────────────────────────────────────

    def on_pregame(self, away_name: str, home_name: str):
        if self.cfg.mode == "halftime":
            return

        home_abbr = TEAM_NAME_MAP.get(home_name, home_name[:3].upper())
        away_abbr = TEAM_NAME_MAP.get(away_name, away_name[:3].upper())
        feat_df   = build_features(home_abbr, away_abbr, self.history, self.pre_feats)
        pred      = self._predict(self.xgb_pre, self.lgb_pre, feat_df)
        game      = f"{away_name} @ {home_name}"
        log.info(f"[PREGAME] {game} | pred={pred}")

        markets = self._get_markets(home_name, away_name)
        log.info(f"  [PREGAME] {game} 找到 {len(markets)} 個市場")
        if not markets:
            log.warning(f"  No Polymarket markets for {game}")
            return

        best = self._find_best_bet(markets, pred, self.cfg.pregame_std)
        if best:
            self._place_bet(game, pred, best, "pregame")
        else:
            best_evs = []
            for m in markets:
                model_over = float(1 - norm.cdf(m["line"], loc=pred, scale=self.cfg.pregame_std))
                for side, mp, mkt in [("OVER", model_over, m["over_prob"]), ("UNDER", 1-model_over, m["under_prob"])]:
                    ev   = mp / mkt - 1 if mkt > 0 else 0
                    edge = mp - mkt
                    best_evs.append(f"{m['line']}{side[0]} ask={mkt:.2f} EV={ev:+.1%} edge={edge:+.1%}")
            log.info(f"  [PREGAME] {game} 無正 EV，各線：{', '.join(best_evs[:4])}")

    # ── halftime ──────────────────────────────────────────────

    def on_halftime(self, game_dict: dict):
        if self.cfg.mode == "pregame":
            return

        away_abbr = ESPN_ABBR_FIX.get(game_dict["away_team"], game_dict["away_team"])
        home_abbr = ESPN_ABBR_FIX.get(game_dict["home_team"], game_dict["home_team"])
        away_name = ABBR_TO_FULL.get(away_abbr, away_abbr)
        home_name = ABBR_TO_FULL.get(home_abbr, home_abbr)
        game      = f"{away_name} @ {home_name}"

        # 防止重複：這場已有未結算的 halftime 下注記錄就跳過（不限日期）
        for acct in [self.pregame_account, self.halftime_account]:
            for t in acct._trades:
                if (t.get("trigger") == "halftime" and
                    t.get("game") == game and
                    t.get("result") is None):
                    log.info(f"  [HALFTIME] {game} 已有未結算下注，跳過重複")
                    return

        feat_df = build_features(
            home_abbr, away_abbr, self.history, self.half_feats,
            home_q1 = game_dict["home_q1"], away_q1 = game_dict["away_q1"],
            home_q2 = game_dict["home_q2"], away_q2 = game_dict["away_q2"],
        )
        pred  = self._predict(self.xgb_half, self.lgb_half, feat_df)
        h1    = game_dict["home_q1"] + game_dict["away_q1"] + game_dict["home_q2"] + game_dict["away_q2"]
        log.info(f"[HALFTIME] {game} | H1={game_dict['away_q1']+game_dict['away_q2']}-{game_dict['home_q1']+game_dict['home_q2']} | pred={pred}")

        markets = self._get_markets(home_name, away_name)
        log.info(f"  [HALFTIME] {game} 找到 {len(markets)} 個市場")
        if not markets:
            log.warning(f"  No Polymarket markets for {game}")
            return

        best = self._find_best_bet(markets, pred, self.cfg.halftime_std)
        if best:
            self._place_bet(game, pred, best, "halftime")
        else:
            best_evs = []
            for m in markets:
                model_over = float(1 - norm.cdf(m["line"], loc=pred, scale=self.cfg.halftime_std))
                for side, mp, mkt in [("OVER", model_over, m["over_prob"]), ("UNDER", 1-model_over, m["under_prob"])]:
                    ev   = mp / mkt - 1 if mkt > 0 else 0
                    edge = mp - mkt
                    best_evs.append(f"{m['line']}{side[0]} ask={mkt:.2f} EV={ev:+.1%} edge={edge:+.1%}")
            log.info(f"  [HALFTIME] {game} 無正 EV，各線：{', '.join(best_evs[:4])}")

    # ── 結算 ──────────────────────────────────────────────────

    def settle_finished(self, games: list):
        settled_ids = {
            t["game"]
            for acct in [self.pregame_account, self.halftime_account]
            for t in acct._trades
            if t.get("result") is not None
        }
        for game in games:
            if game["status"] != "post":
                continue
            total     = game["home_total"] + game["away_total"]
            away_name = ABBR_TO_FULL.get(ESPN_ABBR_FIX.get(game["away_team"], game["away_team"]), game["away_team"])
            home_name = ABBR_TO_FULL.get(ESPN_ABBR_FIX.get(game["home_team"], game["home_team"]), game["home_team"])
            game_str  = f"{away_name} @ {home_name}"

            for acct in [self.pregame_account, self.halftime_account]:
                pending = [t for t in acct._trades
                           if t.get("game") == game_str and t.get("result") is None]
                if pending:
                    log.info(f"Game finished: {game_str} total={total} → settling {len(pending)} [{acct.name}]")
                    for t in pending:
                        acct.settle(game_str, t["line"], total)

    # ── 報告 ──────────────────────────────────────────────────

    def print_report(self):
        all_accounts = []
        if self.cfg.mode in ("pregame", "both"):
            all_accounts.append(self.pregame_account)
        if self.cfg.mode in ("halftime", "both"):
            all_accounts.append(self.halftime_account)

        all_trades  = [t for acct in all_accounts for t in acct._trades]
        settled     = [t for t in all_trades if t.get("result") is not None]
        pending     = [t for t in all_trades if t.get("result") is None]

        print(f"\n{'═'*70}")
        print(f"  Live Trader Report {'🔴 LIVE' if self.cfg.live else '🟡 DRY-RUN'}")
        print(f"{'═'*70}")
        print(f"  {'帳戶':<12} {'初始':>10} {'目前':>10} {'可用':>10} {'報酬率':>8}")
        print(f"  {'─'*52}")
        for acct in all_accounts:
            gain = (acct.total - acct.initial) / acct.initial if acct.initial else 0
            print(f"  {acct.name:<12} ${acct.initial:>9,.2f} ${acct.total:>9,.2f} ${acct.available:>9,.2f} {gain:>+7.1%}")
        total_now  = sum(a.total   for a in all_accounts)
        total_init = sum(a.initial for a in all_accounts)
        print(f"  {'─'*52}")
        print(f"  {'合計':<12} ${total_init:>9,.2f} ${total_now:>9,.2f} {(total_now-total_init)/total_init:>+7.1%}")

        if settled:
            wins    = sum(1 for t in settled if t["result"] == "win")
            pnl     = sum(t.get("pnl", 0) or 0 for t in settled)
            bet_tot = sum(t.get("amount", 0) for t in settled)
            print(f"\n  已結算：{len(settled)} 注 | 勝 {wins} 負 {len(settled)-wins} | 勝率 {wins/len(settled):.1%}")
            print(f"  總損益：${pnl:>+,.2f} | ROI {pnl/bet_tot:+.1%}")

        if pending:
            print(f"\n  待結算：{len(pending)} 注")

        trades = all_trades

        # 明細
        if trades:
            print(f"\n{'─'*70}")
            print(f"  {'日期':<12} {'觸發':<10} {'比賽':<24} {'線':>6} {'方向':<6} {'股價':>5} {'下注':>7} {'EV':>6} {'預測':>5} {'實際':>5} {'結果'}")
            print(f"  {'─'*68}")
            from collections import defaultdict
            by_date = defaultdict(list)
            for t in trades:
                by_date[t.get("date", "?")].append(t)

            for day in sorted(by_date.keys()):
                day_trades = by_date[day]
                day_pnl    = sum(t.get("pnl", 0) or 0 for t in day_trades if t.get("result"))
                print(f"\n  📅 {day}  ({len(day_trades)} 注, 損益 ${day_pnl:+,.2f})")
                for t in day_trades:
                    actual = f"{t['actual_total']:.0f}" if t.get("actual_total") is not None else "?"
                    result = "WIN ✅" if t.get("result") == "win" else ("LOSS ❌" if t.get("result") == "loss" else "待結算")
                    pred   = f"{t['pred']:.0f}" if t.get("pred") else "?"
                    print(
                        f"  {'':<12} {t.get('trigger',''):<10} {t.get('game',''):<24} "
                        f"{t.get('line',0):>6.1f} {t.get('side',''):<6} "
                        f"{t.get('market_p',0):>5.2f} ${t.get('amount',0):>6,.0f} "
                        f"{t.get('ev',0):>+5.1%} {pred:>5} {actual:>5} {result}"
                    )

        print(f"{'═'*70}\n")


# ── 主程式 ────────────────────────────────────────────────────

def run_live(cfg: TraderConfig):
    trader = LiveTrader(cfg)
    mode_str = "🔴 真實交易" if cfg.live else "🟡 DRY-RUN 模式"
    log.info(f"Live Trader 啟動 [{mode_str}]  模式: {cfg.mode} | 總資金: ${cfg.total_capital} | Kelly: {cfg.kelly:.0%}")
    log.info(f"  Log: {TRADER_LOG}")
    log.info(f"  賽前記錄: {TRADER_JSONL_PREGAME}")
    log.info(f"  半場記錄: {TRADER_JSONL_HALFTIME}")

    # pregame 輪詢（獨立 thread）
    settled_games: set[str] = set()
    _state = {"started": False, "done_count": 0}

    def pregame_loop():
        # 啟動時載入所有未結算的賽前下注，避免重複下注（不限日期，跨午夜也有效）
        seen_pregame: set[str] = set()

        for acct in [trader.pregame_account, trader.halftime_account]:
            for t in acct._trades:
                if t.get("trigger") == "pregame" and t.get("result") is None:
                    g = t.get("game", "")
                    if " @ " in g:
                        away, home = g.split(" @ ", 1)
                        home_abbr = TEAM_NAME_MAP.get(home, home[:3].upper())
                        away_abbr = TEAM_NAME_MAP.get(away, away[:3].upper())
                        seen_pregame.add(f"{away_abbr}@{home_abbr}")

        if seen_pregame:
            log.info(f"pregame_loop：已有未結算賽前下注 {len(seen_pregame)} 場，跳過重複")

        while True:
            try:
                games = fetch_scoreboard()
                for g in games:
                    if g["status"] != "pre":
                        continue
                    key = f"{g['away_team']}@{g['home_team']}"
                    if key in seen_pregame:
                        continue
                    seen_pregame.add(key)
                    away = ABBR_TO_FULL.get(ESPN_ABBR_FIX.get(g["away_team"], g["away_team"]), g["away_team"])
                    home = ABBR_TO_FULL.get(ESPN_ABBR_FIX.get(g["home_team"], g["home_team"]), g["home_team"])
                    trader.on_pregame(away, home)
            except Exception as e:
                log.error(f"pregame_loop error: {e}")
            time.sleep(300)  # 5 分鐘

    def settle_loop():
        while True:
            try:
                games = fetch_scoreboard()
                trader.settle_finished(games)
            except Exception as e:
                log.error(f"settle_loop error: {e}")
            time.sleep(300)

    threading.Thread(target=settle_loop,  daemon=True).start()
    if cfg.mode in ("pregame", "both"):
        threading.Thread(target=pregame_loop, daemon=True).start()

    # 主 thread：halftime 偵測
    tracker = QuarterTracker()

    # 啟動時載入今天已觸發半場的比賽，避免重複下注
    today_str = str(date.today())
    for acct in [trader.pregame_account, trader.halftime_account]:
        for t in acct._trades:
            if t.get("date") == today_str and t.get("trigger") == "halftime":
                g = t.get("game", "")
                if " @ " in g:
                    away, home = g.split(" @ ", 1)
                    home_abbr = TEAM_NAME_MAP.get(home, home[:3].upper())
                    away_abbr = TEAM_NAME_MAP.get(away, away[:3].upper())
                    # 找到對應的 game_id 並標記 Q2 已觸發
                    # 用 game string 作 proxy key
                    fake_id = f"{away_abbr}_{home_abbr}_today"
                    tracker._triggered.setdefault(fake_id, set()).add(2)

    if any(tracker._triggered):
        log.info(f"halftime tracker：今天已觸發 {len(tracker._triggered)} 場半場，跳過重複")

    log.info("開始 polling...")

    while True:
        try:
            now   = datetime.now().strftime("%H:%M:%S")
            games = fetch_scoreboard()
            if not games:
                log.info(f"[{now}] No games, retrying...")
                time.sleep(POLL_INTERVAL)
                continue

            live  = [g for g in games if g["status"] == "in"]
            done  = [g for g in games if g["status"] == "post"]
            pre   = [g for g in games if g["status"] == "pre"]

            log.info(f"[{now}] Games: {len(pre)} upcoming / {len(live)} live / {len(done)} finished")

            for g in live:
                q = tracker.check(g)
                if q == 2 and cfg.mode in ("halftime", "both"):
                    log.info(f"  ★ HALFTIME: {g['away_team']} @ {g['home_team']}")
                    trader.on_halftime(g)

            for g in done:
                tracker.reset_game(g["game_id"])

            # 自動結束判斷
            if _state["started"] and len(done) > 0 and len(live) == 0 and len(pre) == 0:
                _state["done_count"] += 1
                if _state["done_count"] >= 5:
                    log.info("All games finished. Shutting down.")
                    trader.print_report()
                    sys.exit(0)
            else:
                if len(live) > 0 or len(pre) > 0:
                    _state["started"] = True
                _state["done_count"] = 0

        except Exception as e:
            log.error(f"main loop error: {e}")

        time.sleep(POLL_INTERVAL)


# ── CLI ───────────────────────────────────────────────────────

def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Polymarket NBA Live Trader")

    # 模式
    parser.add_argument("--mode",         default="both",
                        choices=["pregame","halftime","both"],
                        help="下注模式（預設 both）")
    # 資金
    parser.add_argument("--capital",          type=float, default=100.0,
                        help="總資金 USDC（預設 100，both 模式自動各半）")
    parser.add_argument("--pregame-capital",  type=float, default=None,
                        help="賽前帳戶資金（both 模式用，不設定則各半）")
    parser.add_argument("--halftime-capital", type=float, default=None,
                        help="半場帳戶資金（both 模式用，不設定則各半）")
    # Kelly
    parser.add_argument("--kelly",        type=float, default=0.50,
                        help="Kelly 比例 0~1（預設 0.25）")
    # EV/Edge
    parser.add_argument("--min-ev",       type=float, default=0.03,
                        help="最小 EV 門檻（預設 0.03）")
    parser.add_argument("--min-edge",     type=float, default=0.03,
                        help="最小 Edge 門檻（預設 0.03）")
    # 下注上限
    parser.add_argument("--max-bet",      type=float, default=500.0,
                        help="單注最大金額 USDC（預設 500）")
    parser.add_argument("--max-pct",      type=float, default=0.20,
                        help="單注最大占總資金比例（預設 0.20）")
    # 標準差
    parser.add_argument("--pregame-std",  type=float, default=13.0,
                        help="賽前模型標準差（預設 13.0）")
    parser.add_argument("--halftime-std", type=float, default=9.0,
                        help="半場模型標準差（預設 9.0）")
    # 真實 / dry-run
    parser.add_argument("--live",         action="store_true",
                        help="真實下注（不加此參數為 dry-run）")
    parser.add_argument("--yes", "-y",    action="store_true",
                        help="跳過確認提示（排程用）")
    # 其他指令
    parser.add_argument("--status",       action="store_true",
                        help="查看目前持倉與資金")
    parser.add_argument("--report",       action="store_true",
                        help="印出完整報告")

    args = parser.parse_args()
    cfg  = TraderConfig(args)

    if args.status or args.report:
        trader = LiveTrader(cfg)
        trader.print_report()
        return

    # 真實交易確認
    if cfg.live:
        print(cfg)
        print(f"\n⚠️  即將用真實 USDC 下注！")
        print(f"   模式：{cfg.mode}  總資金：${cfg.total_capital}  Kelly：{cfg.kelly:.0%}")
        if not args.yes:
            print("   確認執行？(y/N)")
            if input().strip().lower() != "y":
                print("取消")
                return
        else:
            print("   --yes 指定，自動確認")
    else:
        print(cfg)
        print("\n🟡 DRY-RUN 模式（不實際下注）\n")

    run_live(cfg)


if __name__ == "__main__":
    main()