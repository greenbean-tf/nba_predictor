# NBA Total Score Predictor

預測 NBA 比賽總得分是否超過莊家線（e.g. 226.5 / 228.5），
利用第一節或半場比分作為輸入，Ensemble ML 模型輸出預測。

## 專案架構

```
nba-total-predictor/
│
├── config/
│   └── config.yaml            # 所有參數集中管理
│
├── data/
│   ├── raw/                   # nba_api 原始抓取資料
│   ├── processed/             # 清洗後的比賽資料
│   └── features/              # 特徵工程輸出
│
├── notebooks/                 # EDA 與實驗用 Jupyter Notebooks
│
├── src/
│   ├── scraper/
│   │   ├── nba_scraper.py     # 歷史資料抓取 (nba_api)
│   │   └── live_scraper.py    # 即時比分抓取 (ESPN API)
│   │
│   ├── features/
│   │   ├── manual_features.py # 手工特徵（rolling stats、半場進度比等）
│   │   └── auto_features.py   # tsfresh 自動特徵生成
│   │
│   ├── models/
│   │   ├── xgb_model.py       # XGBoost
│   │   ├── lgb_model.py       # LightGBM
│   │   ├── lstm_model.py      # PyTorch LSTM
│   │   └── ensemble.py        # Weighted ensemble
│   │
│   ├── backtest/
│   │   └── backtest.py        # 回測框架
│   │
│   └── live/
│       └── predictor.py       # 即時預測主程式
│
├── models/
│   └── saved/                 # 訓練好的模型檔案 (.pkl / .pt)
│
├── logs/                      # 預測紀錄與回測結果
│
├── requirements.txt
└── README.md
```

## 快速開始

```bash
pip install -r requirements.txt

# Step 1: 抓歷史資料
python src/scraper/nba_scraper.py

# Step 2: 特徵工程
python src/features/manual_features.py
python src/features/auto_features.py

# Step 3: 訓練模型
python src/models/ensemble.py

# Step 4: 回測
python src/backtest/backtest.py

# Step 5: 即時預測
python src/live/predictor.py
```

## 預測流程

```
比賽進行中 (ESPN API polling, 每 60 秒)
    ↓
第一節結束 → 觸發 Q1 預測（信心中等）
    ↓
半場結束   → 觸發 H1 預測（高信心，主力）
    ↓
Ensemble 輸出預測總分
    ↓
比較莊家線 → 建議 over / under
```

## 模型組成

| 模型 | 權重 | 強項 |
|---|---|---|
| XGBoost | 40% | 表格特徵、快速 |
| LightGBM | 40% | 大資料量、記憶體效率 |
| LSTM | 20% | 時序趨勢、球隊動態 |

## digital ocean
更新 serve code
ssh root@168.144.73.250
bash /root/nba_predictor/deploy.sh

# auto service
nano /etc/systemd/system/nba_trader.service
nano /etc/systemd/system/nba_trader.timer
--更改後開始reload
systemctl daemon-reload
systemctl restart nba_trader.timer
--確認觸發時間
systemctl list-timers | grep nba

# 看 log
tail -f /root/nba_predictor/logs/live_trader.log
cat /root/nba_predictor/logs/live_trader.log
# 下載到本機(windows 執行)
scp root@168.144.73.250:/root/nba_predictor/logs/live_trader.log C:\Users\User\Desktop\code\nba_predictor\live_trader.log

scp root@168.144.73.250:/root/nba_predictor/logs/live_trades_pregame.jsonl C:\Users\User\Desktop\code\nba_predictor\live_trades_pregame.jsonl

scp root@168.144.73.250:/root/nba_predictor/logs/live_trades_halftime.jsonl C:\Users\User\Desktop\code\nba_predictor\live_trades_halftime.jsonl

# 看服務狀態
systemctl status nba_trader

# 重啟
systemctl restart nba_trader

# 停止
systemctl stop nba_trader

# 手動跑（測試用）
cd /root/nba_predictor
python3 src/live/live_trader.py --report
```

---

# 注意事項
```
1. .env 只用 scp 傳一次，之後不動
2. models/*.pkl 和 features.csv 也只傳一次
   （之後重新訓練模型才需要更新）
3. logs/ 和 *.jsonl 不進 git，只在伺服器上
4. 程式每天自動結束後 systemd 會重啟等待隔天
5. 每月費用 $6 USD，用信用卡付
'''