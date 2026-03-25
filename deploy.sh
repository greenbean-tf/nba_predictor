#!/bin/bash
# =============================================================
#  deploy.sh
#  NBA Live Trader - 伺服器部署腳本
#  執行方式：bash deploy.sh
# =============================================================

set -e  # 任何指令失敗就停止

GREEN="\033[92m"
RED="\033[91m"
YELLOW="\033[93m"
RESET="\033[0m"

ok()   { echo -e "  ${GREEN}✅ $1${RESET}"; }
fail() { echo -e "  ${RED}❌ $1${RESET}"; exit 1; }
info() { echo -e "  ${YELLOW}▶  $1${RESET}"; }
line() { echo -e "\n${GREEN}══════════════════════════════════════${RESET}"; }

# ── 設定區（部署前請確認）──────────────────────────────────────
REPO="git@github.com:greenbean-tf/nba_predictor.git"
APP_DIR="/root/nba_predictor"
LOG_DIR="$APP_DIR/logs"
TIMER_HOUR="04"    # 台灣時間幾點啟動（NBA 比賽約 04:40）
TIMER_MIN="30"
CAPITAL="100"
KELLY="0.8"
PRE_STD="14.6"
HALF_STD="10.7"
# ──────────────────────────────────────────────────────────────

line
echo "  NBA Live Trader 部署腳本"
line

# ── Step 1: 系統更新 ───────────────────────────────────────────
info "Step 1：系統更新..."
apt-get update -qq && apt-get upgrade -y -qq
ok "系統更新完成"

# ── Step 2: 安裝套件 ───────────────────────────────────────────
info "Step 2：安裝 Python / git..."
apt-get install -y -qq python3 python3-pip git
ok "Python $(python3 --version) 已安裝"

# ── Step 3: 設定時區 ───────────────────────────────────────────
info "Step 3：設定台灣時區..."
timedatectl set-timezone Asia/Taipei
TZ=$(timedatectl | grep "Time zone" | awk '{print $3}')
ok "時區：$TZ"

# ── Step 4: 建立 swap（防止 RAM 不足）─────────────────────────
if [ ! -f /swapfile ]; then
    info "Step 4：建立 2GB swap..."
    fallocate -l 2G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile > /dev/null
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
    ok "Swap 建立完成（$(free -h | grep Swap | awk '{print $2}')）"
else
    ok "Step 4：Swap 已存在，跳過"
fi

# ── Step 5: Clone 或更新 repo ──────────────────────────────────
info "Step 5：部署程式..."
if [ -d "$APP_DIR" ]; then
    info "  已有 repo，執行 git pull..."
    cd "$APP_DIR"
    git pull
    ok "程式更新完成"
else
    info "  Clone repo..."
    git clone "$REPO" "$APP_DIR"
    ok "Clone 完成"
fi

# ── Step 6: 建立必要目錄 ───────────────────────────────────────
info "Step 6：建立目錄..."
mkdir -p "$LOG_DIR"
mkdir -p "$APP_DIR/data/features"
mkdir -p "$APP_DIR/models/saved"
ok "目錄建立完成"

# ── Step 7: 安裝 Python 套件 ───────────────────────────────────
info "Step 7：安裝 Python 套件..."
cd "$APP_DIR"
pip3 install -r requirements.txt -q
ok "套件安裝完成"

# ── Step 8: 確認 .env 存在 ─────────────────────────────────────
info "Step 8：確認 .env..."
if [ -f "$APP_DIR/.env" ]; then
    ok ".env 存在"
else
    echo -e "  ${RED}❌ .env 不存在！請手動上傳：${RESET}"
    echo -e "     ${YELLOW}scp .env root@你的IP:$APP_DIR/.env${RESET}"
fi

# ── Step 9: 確認 models 存在 ───────────────────────────────────
info "Step 9：確認 models..."
MODEL_COUNT=$(ls "$APP_DIR/models/saved/"*.pkl 2>/dev/null | wc -l)
if [ "$MODEL_COUNT" -ge 4 ]; then
    ok "找到 $MODEL_COUNT 個 model 檔案"
else
    echo -e "  ${RED}❌ models 不足！請手動上傳：${RESET}"
    echo -e "     ${YELLOW}scp -r models/ root@你的IP:$APP_DIR/${RESET}"
fi

# ── Step 10: 確認 features.csv 存在 ────────────────────────────
info "Step 10：確認 features.csv..."
if [ -f "$APP_DIR/data/features/features.csv" ]; then
    ROWS=$(wc -l < "$APP_DIR/data/features/features.csv")
    ok "features.csv 存在（$ROWS 行）"
else
    echo -e "  ${RED}❌ features.csv 不存在！請手動上傳：${RESET}"
    echo -e "     ${YELLOW}scp data/features/features.csv root@你的IP:$APP_DIR/data/features/${RESET}"
fi

# ── Step 11: 建立 systemd service ─────────────────────────────
info "Step 11：建立 systemd service..."
cat > /etc/systemd/system/nba_trader.service << EOF
[Unit]
Description=NBA Live Trader
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$APP_DIR
ExecStart=/usr/bin/python3 -u src/live/live_trader.py --mode both --capital $CAPITAL --kelly $KELLY --pregame-std $PRE_STD --halftime-std $HALF_STD --live --yes
Restart=on-failure
RestartSec=30
StandardOutput=append:$LOG_DIR/live_trader.log
StandardError=append:$LOG_DIR/live_trader.log

[Install]
WantedBy=multi-user.target
EOF
ok "service 檔建立完成"

# ── Step 12: 建立 systemd timer ───────────────────────────────
info "Step 12：建立 systemd timer（每天 ${TIMER_HOUR}:${TIMER_MIN} 台灣時間啟動）..."
cat > /etc/systemd/system/nba_trader.timer << EOF
[Unit]
Description=NBA Live Trader Daily Timer

[Timer]
OnCalendar=*-*-* ${TIMER_HOUR}:${TIMER_MIN}:00
Unit=nba_trader.service

[Install]
WantedBy=timers.target
EOF
ok "timer 檔建立完成"

# ── Step 13: 啟動 timer ────────────────────────────────────────
info "Step 13：啟動 timer..."
systemctl daemon-reload
systemctl enable nba_trader.timer
systemctl start nba_trader.timer
ok "Timer 已啟動"

NEXT=$(systemctl list-timers nba_trader.timer --no-pager | grep nba | awk '{print $1, $2}')
ok "下次執行：$NEXT"

# ── 完成 ──────────────────────────────────────────────────────
line
echo -e "  ${GREEN}部署完成！${RESET}"
line
echo ""
echo "  常用指令："
echo "    查看 log：     tail -f $LOG_DIR/live_trader.log"
echo "    手動啟動：     systemctl start nba_trader"
echo "    查看 timer：   systemctl list-timers | grep nba"
echo "    查看服務：     systemctl status nba_trader"
echo "    更新程式：     bash $APP_DIR/deploy.sh"
echo ""
echo "  記得上傳以下檔案（如果還沒傳）："
echo "    .env           scp .env root@你的IP:$APP_DIR/"
echo "    models/        scp -r models/ root@你的IP:$APP_DIR/"
echo "    features.csv   scp data/features/features.csv root@你的IP:$APP_DIR/data/features/"
echo ""