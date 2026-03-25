"""
capital_tracker.py
------------------
方法 C：動態剩餘資金 Kelly

每次下注後從可用資金中扣除，
下一注根據「當下剩餘資金」重新計算 Kelly。

使用方式：
  from src.live.capital_tracker import CapitalTracker
  tracker = CapitalTracker(total_capital=10000)
  tracker.add_bet(...)

  或直接執行查看今日狀況：
  python src/live/capital_tracker.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import logging
from datetime import datetime, date
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

LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

BETS_FILE = LOG_DIR / "bets_log.jsonl"


class CapitalTracker:
    """
    動態剩餘資金 Kelly 追蹤器。

    核心概念：
      available = total - sum(已下注但未結算的金額)
      每次觸發時，Kelly 以 available 為基礎計算，不是 total
    """

    def __init__(self, total_capital: float = 10000.0,
                 kelly_frac: float = 0.25,
                 min_ev: float = 0.02,
                 min_edge: float = 0.03):
        self.total_capital = total_capital
        self.kelly_frac    = kelly_frac
        self.min_ev        = min_ev
        self.min_edge      = min_edge
        self._load_today_bets()

    # ── 資金計算 ──────────────────────────────────────────────

    @property
    def committed(self) -> float:
        """今日已下注但尚未結算的總金額。"""
        return sum(
            b["amount"] for b in self._today_bets
            if b.get("result") is None
        )

    @property
    def available(self) -> float:
        """當前可用資金 = 總資金 - 已下注未結算。"""
        return self.total_capital - self.committed

    @property
    def realized_pnl(self) -> float:
        """今日已結算的損益。"""
        pnl = 0
        for b in self._today_bets:
            if b.get("result") == "win":
                pnl += b["amount"] * b["ev"]   # 近似
            elif b.get("result") == "loss":
                pnl -= b["amount"]
        return pnl

    # ── Kelly 計算（以剩餘資金為基礎）────────────────────────

    def kelly_amount(self, model_p: float, market_p: float) -> float:
        """
        動態 Kelly：以「當前可用資金」計算下注金額。

        f = (model_p - market_p) / (1 - market_p)  × kelly_frac
        amount = available × f
        """
        if market_p <= 0 or market_p >= 1:
            return 0
        f = (model_p - market_p) / (1 - market_p)
        f = max(0, f * self.kelly_frac)
        return round(self.available * f, 2)

    def should_bet(self, model_p: float, market_p: float,
                   side: str) -> tuple[bool, str]:
        """
        判斷這注是否值得下。
        回傳 (bool, reason)
        """
        if self.available <= 0:
            return False, "No available capital"

        edge = model_p - market_p
        ev   = (model_p / market_p) - 1 if market_p > 0 else -1

        if edge < self.min_edge:
            return False, f"Edge {edge:+.1%} < min {self.min_edge:.0%}"
        if ev < self.min_ev:
            return False, f"EV {ev:+.1%} < min {self.min_ev:.0%}"

        return True, f"Edge={edge:+.1%} EV={ev:+.1%}"

    # ── 下注記錄 ──────────────────────────────────────────────

    def add_bet(self, game: str, line: float, side: str,
                model_p: float, market_p: float,
                quarter: int, pred: float) -> dict | None:
        """
        記錄一筆下注，回傳下注資訊。
        """
        ok, reason = self.should_bet(model_p, market_p, side)
        if not ok:
            log.info(f"  Skip {game} {line} {side}: {reason}")
            return None

        amount = self.kelly_amount(model_p, market_p)
        if amount < 1:
            log.info(f"  Skip {game} {line} {side}: amount too small ({amount:.2f})")
            return None

        edge = model_p - market_p
        ev   = (model_p / market_p) - 1

        bet = {
            "id":         f"{datetime.now().strftime('%H%M%S')}_{game[:10]}_{line}_{side}",
            "timestamp":  datetime.now().isoformat(),
            "date":       str(date.today()),
            "game":       game,
            "line":       line,
            "side":       side,
            "quarter":    quarter,
            "pred":       pred,
            "model_p":    round(model_p,  4),
            "market_p":   round(market_p, 4),
            "edge":       round(edge, 4),
            "ev":         round(ev,   4),
            "amount":     amount,
            "capital_at_bet": self.available,
            "result":     None,      # win / loss / push（事後填入）
        }

        self._today_bets.append(bet)
        self._save_bet(bet)

        log.info(
            f"  BET: {game} | {line} {side} | "
            f"${amount:.0f} ({amount/self.available:.1%} of ${self.available:.0f}) | "
            f"EV={ev:+.1%}"
        )
        return bet

    def settle_bet(self, bet_id: str, result: str):
        """
        結算一筆下注（result: 'win' / 'loss' / 'push'）。
        """
        for bet in self._today_bets:
            if bet["id"] == bet_id:
                bet["result"] = result
                if result == "win":
                    # Polymarket: 贏了拿回 1/market_p 倍（含本金）
                    payout = bet["amount"] / bet["market_p"]
                    profit = payout - bet["amount"]
                    self.total_capital += profit
                    log.info(f"  WIN:  {bet['game']} +${profit:.2f}")
                elif result == "loss":
                    self.total_capital -= bet["amount"]
                    log.info(f"  LOSS: {bet['game']} -${bet['amount']:.2f}")
                self._update_bet(bet)
                return
        log.warning(f"Bet {bet_id} not found")

    # ── 今日狀況報告 ──────────────────────────────────────────

    def print_status(self):
        bets = self._today_bets
        pending = [b for b in bets if b["result"] is None]
        settled = [b for b in bets if b["result"] is not None]

        print(f"\n{'═'*60}")
        print(f"  今日資金狀況 ({date.today()})")
        print(f"{'═'*60}")
        print(f"  總資金：    ${self.total_capital:>10,.2f}")
        print(f"  已下注：    ${self.committed:>10,.2f}  ({self.committed/self.total_capital:.1%})")
        print(f"  可用資金：  ${self.available:>10,.2f}  ({self.available/self.total_capital:.1%})")
        print(f"  今日損益：  ${self.realized_pnl:>+10,.2f}")
        print(f"{'─'*60}")

        if pending:
            print(f"\n  待結算（{len(pending)} 注）：")
            print(f"  {'比賽':<25}  {'線':>6}  {'方向':<6}  {'金額':>8}  {'EV':>6}")
            print(f"  {'─'*55}")
            for b in pending:
                print(f"  {b['game']:<25}  {b['line']:>6.1f}  {b['side']:<6}  "
                      f"${b['amount']:>7,.0f}  {b['ev']:>+5.1%}")

        if settled:
            print(f"\n  已結算（{len(settled)} 注）：")
            for b in settled:
                icon = "✅" if b["result"] == "win" else "❌"
                print(f"  {icon} {b['game']} {b['line']} {b['side']} "
                      f"${b['amount']:,.0f} → {b['result']}")

        print(f"{'═'*60}\n")

    def print_next_kelly(self, opportunities: list[dict]):
        """
        給定一批機會，印出以「當前可用資金」計算的 Kelly 建議。
        opportunities = [{"game":..., "line":..., "side":...,
                          "model_p":..., "market_p":..., "ev":...}]
        """
        if not opportunities:
            return

        print(f"\n  可用資金 ${self.available:,.0f} 的 Kelly 建議：")
        print(f"  {'比賽':<25}  {'線':>6}  {'方向':<6}  {'EV':>6}  {'建議金額':>10}  {'佔可用':>7}")
        print(f"  {'─'*65}")

        total_amount = 0
        for opp in sorted(opportunities, key=lambda x: x["ev"], reverse=True):
            ok, reason = self.should_bet(
                opp["model_p"], opp["market_p"], opp["side"]
            )
            if not ok:
                continue
            amount = self.kelly_amount(opp["model_p"], opp["market_p"])
            pct    = amount / self.available if self.available > 0 else 0
            total_amount += amount
            print(f"  {opp['game']:<25}  {opp['line']:>6.1f}  {opp['side']:<6}  "
                  f"{opp['ev']:>+5.1%}  ${amount:>9,.0f}  {pct:>6.1%}")

        print(f"  {'─'*65}")
        print(f"  總建議下注：${total_amount:,.0f}  "
              f"({total_amount/self.available:.1%} 可用 / "
              f"{total_amount/self.total_capital:.1%} 總資金)\n")

    # ── 持久化 ────────────────────────────────────────────────

    def _load_today_bets(self):
        today = str(date.today())
        self._today_bets = []
        if not BETS_FILE.exists():
            return
        with open(BETS_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    bet = json.loads(line)
                    if bet.get("date") == today:
                        self._today_bets.append(bet)
        log.info(f"Loaded {len(self._today_bets)} today's bets from log")

    def _save_bet(self, bet: dict):
        with open(BETS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(bet, default=str) + "\n")

    def _update_bet(self, bet: dict):
        """更新 JSONL 中的結算結果（重寫整個檔案）。"""
        all_bets = []
        if BETS_FILE.exists():
            with open(BETS_FILE, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_bets.append(json.loads(line))

        with open(BETS_FILE, "w", encoding="utf-8") as f:
            for b in all_bets:
                if b["id"] == bet["id"]:
                    f.write(json.dumps(bet, default=str) + "\n")
                else:
                    f.write(json.dumps(b, default=str) + "\n")


# ── 整合到 portfolio.py 的介面 ────────────────────────────────

def evaluate_and_bet(tracker: CapitalTracker,
                     portfolio_df: pd.DataFrame,
                     quarter: int,
                     auto_bet: bool = False) -> list[dict]:
    """
    把 portfolio.py 的輸出接到 CapitalTracker。

    auto_bet=True：自動下注（有正 EV 就下）
    auto_bet=False：只印建議，不下注
    """
    opportunities = []
    for _, row in portfolio_df.iterrows():
        game = row.get("game", "Unknown")
        for side in ["over", "under"]:
            model_p  = row[f"model_{side}"]
            market_p = row[f"mkt_{side}"]
            ev_val   = row[f"ev_{side}"]
            if ev_val >= tracker.min_ev and (model_p - market_p) >= tracker.min_edge:
                opportunities.append({
                    "game":     game,
                    "line":     row["line"],
                    "side":     side.upper(),
                    "model_p":  model_p,
                    "market_p": market_p,
                    "ev":       ev_val,
                    "pred":     row.get("pred", 0),
                })

    # 按 EV 排序
    opportunities.sort(key=lambda x: x["ev"], reverse=True)

    # 印出 Kelly 建議（以當前可用資金計算）
    tracker.print_next_kelly(opportunities)

    placed = []
    if auto_bet:
        for opp in opportunities:
            bet = tracker.add_bet(
                game=opp["game"], line=opp["line"], side=opp["side"],
                model_p=opp["model_p"], market_p=opp["market_p"],
                quarter=quarter, pred=opp["pred"],
            )
            if bet:
                placed.append(bet)

    return placed


# ── 進入點 ────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital",  type=float, default=10000,
                        help="Total capital (default: 10000)")
    parser.add_argument("--kelly",    type=float, default=0.25)
    parser.add_argument("--status",   action="store_true",
                        help="Show today's status")
    parser.add_argument("--settle",   type=str,   default=None,
                        help="Settle a bet: BET_ID:win or BET_ID:loss")
    args = parser.parse_args()

    tracker = CapitalTracker(
        total_capital=args.capital,
        kelly_frac=args.kelly,
    )

    if args.settle:
        bet_id, result = args.settle.split(":")
        tracker.settle_bet(bet_id, result)

    tracker.print_status()


# ── 自動結算（查 ESPN 最終比分）────────────────────────────────

def auto_settle_finished_games(tracker: "CapitalTracker"):
    """
    查詢 ESPN API 的已結束比賽，自動結算今日未結算的下注。
    在比賽時段結束後執行（例如凌晨）。
    """
    import requests
    from datetime import date

    pending = [b for b in tracker._today_bets if b.get("result") is None]
    if not pending:
        log.info("No pending bets to settle.")
        return

    log.info(f"Auto-settling {len(pending)} pending bets...")

    # 抓 ESPN 今日比賽最終比分
    try:
        r = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
            timeout=10, proxies={}
        )
        games = r.json().get("events", [])
    except Exception as e:
        log.error(f"ESPN fetch failed: {e}")
        return

    # 建立 final score lookup
    final_scores = {}
    for game in games:
        status = game["status"]["type"]["state"]
        if status != "post":
            continue
        comp   = game["competitions"][0]
        home   = next(c for c in comp["competitors"] if c["homeAway"] == "home")
        away   = next(c for c in comp["competitors"] if c["homeAway"] == "away")
        home_abbr = home["team"]["abbreviation"]
        away_abbr = away["team"]["abbreviation"]
        total = int(home.get("score", 0)) + int(away.get("score", 0))
        # 用隊名縮寫組成 key
        key = f"{away_abbr}@{home_abbr}"
        final_scores[key] = total
        log.info(f"  Final: {key} = {total}")

    # 結算每筆下注
    TEAM_ABBR_MAP = {v: k for k, v in {
        "Cavaliers": "CLE", "Bulls": "CHI", "Pistons": "DET", "Wizards": "WAS",
        "Clippers": "LAC",  "Pelicans": "NOP", "Lakers": "LAL", "Heat": "MIA",
        "Bucks": "MIL",     "Jazz": "UTA",  "Magic": "ORL",  "Hornets": "CHA",
        "76ers": "PHI",     "Kings": "SAC", "Suns": "PHX",   "Spurs": "SAS",
        "Warriors": "GSW",  "Nets": "BKN",  "Celtics": "BOS","Grizzlies": "MEM",
        "Hawks": "ATL",     "Mavericks": "DAL", "Nuggets": "DEN", "Thunder": "OKC",
        "Raptors": "TOR",   "Timberwolves": "MIN", "Trail Blazers": "POR",
        "Rockets": "HOU",   "Knicks": "NYK", "Pacers": "IND",
    }.items()}

    for bet in pending:
        # 解析比賽名稱找 key
        parts = bet["game"].split(" @ ")
        if len(parts) != 2:
            continue
        away_full, home_full = parts[0].strip(), parts[1].strip()
        away_abbr = TEAM_ABBR_MAP.get(away_full, away_full[:3].upper())
        home_abbr = TEAM_ABBR_MAP.get(home_full, home_full[:3].upper())
        key = f"{away_abbr}@{home_abbr}"

        if key not in final_scores:
            log.warning(f"  No final score for {key}, skipping")
            continue

        actual_total = final_scores[key]
        line  = bet["line"]
        side  = bet["side"]

        if side == "OVER":
            result = "win" if actual_total > line else "loss"
        elif side == "UNDER":
            result = "win" if actual_total < line else "loss"
        else:
            result = "push"

        log.info(f"  Settle {bet['game']} {line} {side}: "
                 f"actual={actual_total} → {result}")
        tracker.settle_bet(bet["id"], result)

    tracker.print_status()
