---
title: QuantMind AI Helper
emoji: 🔮
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7080
---

# QuantMind

> **Quant engine meets multi-agent intelligence — with real broker execution.**

QuantMind is an end-to-end quantitative trading toolkit for **Chinese A-shares** that fuses a deep-learning price forecaster (Kronos + QLib) with a multi-agent LLM reasoning layer (TradingAgents) to produce high-conviction trading signals — then executes them through paper trading, live order export, or **real broker orders via MiniQMT**.

---

## ✨ Features

| Layer | What it does |
|---|---|
| **Kronos Forecaster** | Transformer-based time-series model trained on A-share price history. Predicts 5–20 day forward return distribution. |
| **QLib Backtest** | Walk-forward backtest engine with transaction costs, multiple strategy modes (Forecast Trend, Mean Reversion, Breakout Confirmation). |
| **T+0 AI Analysis** | Real-time technical indicators (VWAP, RSI, MACD, Bollinger, volume) with LLM-powered Chinese-language interpretation. |
| **TradingAgents** | Multi-agent LLM pipeline: Market → News → Sentiment → Fundamentals analysts → Bull/Bear debate → Risk team → Portfolio manager → Final decision. |
| **Consensus Engine** | Merges Kronos quant signal (BUY / HOLD / SELL) with TradingAgents qualitative decision (5-tier) into a single conviction rating. |
| **Paper Trading** | Virtual portfolio with cash management, lot-size enforcement, P&L tracking, trade history. |
| **Live Order Export** | Generates structured order files for manual execution at your broker. |
| **🏦 Real Broker Trading** | MiniQMT (xtquant) integration — execute real orders at 30+ Chinese brokers (国金, 华泰, 中泰, 银河, etc). Risk checks, order history, one-click execution from signals. |
| **LLM Flexibility** | Works with OpenAI, DeepSeek, Qwen/DashScope, Ollama (local), or any OpenAI-compatible endpoint. Per-user API key configuration. |
| **Multi-User Auth** | Flask-Login authentication, per-user data isolation, persistent sessions. |
| **Web UI** | Single-page Flask app — bilingual (EN/ZH), real-time quote polling, unified settings modal with tabbed LLM + Broker config. |

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      Web UI  (Flask)                         │
│   Symbol / Date ─► Backtest ─► Signal ─► Paper / Live Trade  │
│                          │                    │              │
│              ┌───────────▼──────────┐   ┌─────▼─────┐       │
│              │   AI Multi-Agent     │   │  Broker   │       │
│              │   Analysis Panel     │   │  Service  │       │
│              └───────────┬──────────┘   │ (MiniQMT) │       │
│                          │              └─────┬─────┘       │
└──────────────────────────┼────────────────────┼─────────────┘
                           │                    │
        ┌──────────────────┼──────────┐         ▼
        ▼                  ▼          ▼    Real Broker
  Kronos Engine      TradingAgents   T+0   (xtquant → Exchange)
  (QLib + HF model)  (Multi-LLM)    AI
        │                  │
        └────────┬─────────┘
                 ▼
          Consensus Signal
          (STRONG BUY ► STRONG SELL)
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Windows (paths use `\`; Linux/macOS require minor path tweaks)
- An LLM API key **or** [Ollama](https://ollama.com) running locally

### 1. Clone

```bash
git clone https://github.com/zbcui/quantmind.git
cd quantmind
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> TradingAgents is installed from GitHub automatically via `requirements.txt`.

### 3. Download QLib data (optional — required for backtests)

```powershell
.\setup_qlib.ps1
```

### 4. Run

```bash
python app.py
```

Open [http://127.0.0.1:7080](http://127.0.0.1:7080)

Default login: `admin@quantmind.local` / `admin` (auto-created on first run).

### 5. Configure LLM & Broker

Click the **⚙ 设置** button in the top-right corner:

- **🤖 AI / LLM tab** — Choose provider (Ollama, DeepSeek, OpenAI, Qwen, Custom), set base URL, API key, model
- **🏦 券商交易 tab** — Enable real trading, set broker name, account ID, MiniQMT path, risk limits

All settings are saved per-user in the database.

---

## 📊 Usage

### Forecasting & Backtest

1. Enter a stock symbol (e.g. `601169` for Bank of Beijing)
2. Set date range, lookback window, prediction horizon
3. Choose a strategy and click **Run Analysis**
4. View forecast chart, backtest equity curve, and T+0 indicators

### T+0 AI Analysis

1. T+0 indicators load automatically when you enter a symbol
2. Click **🤖 AI 解读** to run combined analysis (T0 instant + TradingAgents background)
3. When a **BUY/SELL** signal appears, the **🏦 执行实盘下单** button shows up automatically

### AI Multi-Agent Analysis

1. The **🤖 AI 解读** button launches both T0 analysis and TradingAgents in parallel
2. The pipeline runs 9 agents (~2–5 min for a full run)
3. Expand accordions to read each agent's report
4. The **Consensus** box merges AI decision with Kronos quant signal
5. If consensus is BUY or SELL, the **🏦 按信号执行实盘下单** button appears

### Paper Trading

1. After analysing, click **Run + Execute Paper Trade**
2. Monitor your portfolio value, positions, and trade history in the **Portfolio** page

### Live Order Export & Real Broker Trading

- **Export**: Click **Run + Export Live Order** to generate a structured JSON order file
- **One-click broker**: Click **🏦 一键下单到券商** in the export results to send the order to your broker via MiniQMT
- **Manual order**: Click the **🏦 手动下单** button to open a standalone order form (symbol, side, price, volume)
- **Order history**: View all broker orders with the **📋 委托记录** button

> ⚠️ Real broker trading requires a local MiniQMT client running and a broker account with programmatic trading enabled. See [Broker Setup](#-broker-setup-miniqmt) below.

---

## 🏦 Broker Setup (MiniQMT)

Real broker trading uses the [MiniQMT](https://dict.thinktrader.net/nativeApi/start_now.html) SDK (`xtquant`).

### Prerequisites

1. **Broker account** with programmatic trading enabled (开通程序化交易权限)
2. **MiniQMT client** installed and logged in on the same machine
3. `pip install xtquant` (or use the SDK provided by your broker)

### Configuration

In the **⚙ 设置 → 🏦 券商交易** tab:

| Field | Description |
|---|---|
| 启用实盘交易 | Master switch — must be checked to execute real orders |
| 券商名称 | Your broker name (e.g. 国金证券) |
| 资金账号 | Your trading account ID |
| MiniQMT 路径 | Path to MiniQMT `userdata_mini` directory |
| 单笔最大金额 | Max order value per trade (risk control) |
| 每日最大下单次数 | Max daily order count (risk control) |
| 下单前需要确认 | Show confirmation dialog before each order |

### Risk Controls

- **Max order value**: Rejects single orders exceeding the configured limit
- **Daily order limit**: Blocks new orders after hitting the daily cap
- **Lot size**: Enforces 100-share minimum for A-share BUY orders
- **Confirmation dialog**: Shows order details and requires explicit confirmation

### Supported Brokers

Any broker offering MiniQMT/QMT access, including: 国金证券, 华泰证券, 中泰证券, 银河证券, 中信证券, 招商证券, 广发证券, and 20+ others.

> 💡 On cloud deployments (HF Spaces), broker features show appropriate error messages since MiniQMT requires a local client.

---

## 🗂 Project Structure

```
quantmind/
├── app.py                    # Flask server, auth, all API routes (30+ endpoints)
├── config.py                 # ToolkitConfig dataclass
├── analysis_service.py       # Backtest & prediction orchestration
├── broker_service.py         # MiniQMT broker integration (orders, risk, account)
├── kronos_engine.py          # Kronos model loader & inference
├── data_sources.py           # AKShare / QLib data adapters, real-time quotes
├── strategy_catalog.py       # Strategy definitions & signal logic
├── llm_service.py            # LLM config, proxy detection, T+0 AI analysis
├── trading_service.py        # Paper & live trade execution
├── trade_storage.py          # SQLite persistence, user management, schema migration
├── trading_agents_service.py # TradingAgents integration (async jobs, consensus)
├── predict_stock.py          # CLI prediction script
├── single_stock_backtest.py  # CLI backtest script
├── setup_qlib.ps1            # QLib data download helper
├── Dockerfile                # Docker config for HF Spaces deployment
├── requirements.txt
├── data/
│   ├── llm_config.json       # LLM credentials (git-ignored)
│   └── qlib_cn/              # QLib CN data (git-ignored)
├── outputs/                  # Generated charts, portfolios, orders (git-ignored)
├── static/                   # CSS / JS / ECharts assets
└── templates/
    └── index.html            # Single-page web UI (bilingual, tabbed settings)
```

---

## 🌐 Deployment

### Local

```bash
python app.py    # → http://127.0.0.1:7080
```

### Docker (Hugging Face Spaces)

The app deploys to HF Spaces via Docker:

```dockerfile
# Key settings in Dockerfile:
ENV KRONOS_TOOLKIT_OUTPUT=/data   # Persistent volume for DB
ENV LANG=C.UTF-8                  # UTF-8 locale for Chinese text
EXPOSE 7080
```

Push to HF:

```bash
git push hf master:main --force
```

> The SQLite database persists at `/data` on HF Spaces persistent volume, surviving redeploys.

---

## ⚙ Configuration Reference

All runtime config lives in `config.py` (`ToolkitConfig`). Key fields:

| Field | Default | Description |
|---|---|---|
| `default_lookback` | 400 | Days of history fed to Kronos |
| `default_pred_len` | 20 | Forecast horizon (trading days) |
| `signal_threshold` | 0.0 | Minimum predicted return to trigger BUY |
| `transaction_cost_rate` | 0.0015 | Round-trip cost (0.15%) |
| `default_paper_cash` | 100 000 | Starting virtual cash (CNY) |
| `lot_size` | 100 | A-share lot size |

### LLM Providers

| Provider | `provider` value | `base_url` |
|---|---|---|
| Ollama (local) | `Ollama` | `http://localhost:11434` |
| DeepSeek | `DeepSeek` | `https://api.deepseek.com` |
| OpenAI | `OpenAI` | `https://api.openai.com` |
| Qwen / DashScope | `Qwen` | `https://dashscope.aliyuncs.com/compatible-mode` |
| Any OpenAI-compatible | `Custom` | your endpoint |

---

## 🤖 TradingAgents Integration Details

QuantMind wraps the open-source [TradingAgents](https://github.com/TauricResearch/TradingAgents) framework as an async background service:

- **Ticker mapping**: bare codes like `601169` → `601169.SS` (Shanghai) / `000651.SZ` (Shenzhen)
- **Proxy**: corporate proxy is restored before each agent run so `yfinance` can reach Yahoo Finance
- **Pre-flight**: LLM connectivity and quota are tested before queuing a job
- **Job lifecycle**: `pending → running → done | failed` persisted in SQLite
- **Consensus logic**: 5-tier AI signal × 3-tier quant signal → `STRONG_BUY / LEAN_BUY / HOLD / LEAN_SELL / STRONG_SELL / CONFLICT`
- **Reflection**: after each SELL trade, realised P&L is fed back to the agent's memory via `reflect_and_remember()`

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `flask` + `flask-login` | Web server with multi-user auth |
| `pyqlib` | Quantitative investment platform |
| `akshare` | Chinese A-share market data & real-time quotes |
| `torch` | Kronos model inference |
| `tradingagents` | Multi-agent LLM trading framework |
| `huggingface_hub` | Model download |
| `pyarrow` | Parquet output for analysis results |
| `xtquant` | MiniQMT broker SDK (optional, for real trading) |

---

## 🔒 Security Notes

- API keys are stored **per-user in the database**, never in git
- `data/llm_config.json` is **git-ignored** — global defaults only
- `outputs/` is git-ignored — contains portfolio state and live orders
- `data/qlib_cn/` is git-ignored — large binary data files
- Flask `secret_key` is read from `FLASK_SECRET_KEY` env var (stable across restarts)
- Broker config stored per-user with masked display in UI
- All real broker orders require explicit confirmation dialog

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Kronos](https://huggingface.co/NeoQuasar/Kronos-small) — time-series forecasting model by NeoQuasar
- [QLib](https://github.com/microsoft/qlib) — Microsoft quantitative investment platform
- [TradingAgents](https://github.com/TauricResearch/TradingAgents) — multi-agent LLM trading framework by Tauric Research
- [AKShare](https://github.com/akfamily/akshare) — Chinese financial data library
- [MiniQMT](https://dict.thinktrader.net/nativeApi/start_now.html) — broker SDK for A-share programmatic trading
