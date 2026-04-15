# QuantMind

> **Quant engine meets multi-agent intelligence.**

QuantMind is an end-to-end quantitative trading toolkit for **Chinese A-shares** that fuses a deep-learning price forecaster (Kronos + QLib) with a multi-agent LLM reasoning layer (TradingAgents) to produce high-conviction trading signals — then executes them through paper or live trading workflows.

---

## ✨ Features

| Layer | What it does |
|---|---|
| **Kronos Forecaster** | Transformer-based time-series model trained on A-share price history. Predicts 5–20 day forward return distribution. |
| **QLib Backtest** | Walk-forward backtest engine with transaction costs, multiple strategy modes (Forecast Trend, Mean Reversion, Breakout Confirmation). |
| **TradingAgents** | Multi-agent LLM pipeline: Market → News → Sentiment → Fundamentals analysts → Bull/Bear debate → Risk team → Portfolio manager → Final decision. |
| **Consensus Engine** | Merges Kronos quant signal (BUY / HOLD / SELL) with TradingAgents qualitative decision (5-tier) into a single conviction rating. |
| **Paper Trading** | Virtual portfolio with cash management, lot-size enforcement, P&L tracking, trade history. |
| **Live Order Export** | Generates structured order files for manual execution at your broker. |
| **LLM Flexibility** | Works with OpenAI, Anthropic, DeepSeek, Qwen/DashScope, Ollama (local), or any OpenAI-compatible endpoint. |
| **Web UI** | Single-page Flask app — bilingual (EN/ZH), dark-mode, real-time polling. |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Web UI  (Flask)                     │
│   Symbol / Date ─► Backtest ─► Signal ─► Paper Trade   │
│                          │                              │
│              ┌───────────▼───────────┐                  │
│              │   AI Multi-Agent      │                  │
│              │   Analysis Panel      │                  │
│              └───────────┬───────────┘                  │
└──────────────────────────┼──────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
  Kronos Engine      TradingAgents       Trade Storage
  (QLib + HF model)  (Multi-LLM agents)  (SQLite)
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

TradingAgents is installed separately as an editable package:

```bash
git clone https://github.com/TauricResearch/TradingAgents ../TradingAgents
pip install -e ../TradingAgents
```

### 3. Download QLib data (optional — required for backtests)

```powershell
.\setup_qlib.ps1
```

### 4. Configure LLM

Create (or edit) `data/llm_config.json`:

```json
{
  "provider": "OpenAI",
  "base_url": "https://api.openai.com",
  "api_key": "sk-...",
  "model": "gpt-4o-mini",
  "max_tokens": 800,
  "temperature": 0.3
}
```

**Supported providers:**

| Provider | `provider` value | `base_url` |
|---|---|---|
| OpenAI | `OpenAI` | `https://api.openai.com` |
| Anthropic / Claude | `OpenAI` | your proxy URL |
| DeepSeek | `OpenAI` | `https://api.deepseek.com` |
| Qwen / DashScope | `OpenAI` | `https://dashscope.aliyuncs.com/compatible-mode` |
| Ollama (local) | `Ollama` | `http://localhost:11434` |
| Any OpenAI-compatible | `OpenAI` | your endpoint |

### 5. Run

```bash
python app.py
```

Open [http://127.0.0.1:7080](http://127.0.0.1:7080)

---

## 📊 Usage

### Forecasting & Backtest

1. Enter a stock symbol (e.g. `601169` for Bank of Beijing)
2. Set date range, lookback window, prediction horizon
3. Choose a strategy and click **Analyse**
4. View forecast chart, backtest equity curve, and T+0 indicators

### AI Multi-Agent Analysis

1. In the **AI Multi-Agent Analysis** panel, click **▶ Run Agents**
2. The pipeline runs 9 agents in sequence (~4–5 min for a full run)
3. Expand accordions to read each agent's report (market, news, sentiment, fundamentals, debate, risk, investment plan, trader plan, final decision)
4. The **Consensus** box merges the AI decision with the Kronos quant signal

### Paper Trading

1. After analysing, click **Buy / Sell** to execute a virtual trade
2. Monitor your portfolio value, positions, and trade history in the **Paper Portfolio** section
3. P&L from closed trades automatically feeds back into TradingAgents' reflection memory

### Live Order Export

- Click **Export Live Order** to write a structured JSON order file to `outputs/manual_live_orders/`
- Execute at your broker manually or wire up to a broker API

---

## 🗂 Project Structure

```
quantmind/
├── app.py                    # Flask server & all API routes
├── config.py                 # ToolkitConfig dataclass
├── analysis_service.py       # Backtest & prediction orchestration
├── kronos_engine.py          # Kronos model loader & inference
├── data_sources.py           # AKShare / QLib data adapters
├── strategy_catalog.py       # Strategy definitions & signal logic
├── llm_service.py            # LLM config, proxy detection, T+0 analysis
├── trading_service.py        # Paper & live trade execution
├── trade_storage.py          # SQLite persistence layer
├── trading_agents_service.py # TradingAgents integration (async jobs, consensus)
├── predict_stock.py          # CLI prediction script
├── single_stock_backtest.py  # CLI backtest script
├── setup_qlib.ps1            # QLib data download helper
├── requirements.txt
├── data/
│   ├── llm_config.json       # LLM credentials (git-ignored)
│   └── qlib_cn/              # QLib CN data (git-ignored)
├── outputs/                  # Generated charts, portfolios, orders (git-ignored)
├── static/                   # CSS / JS assets
└── templates/
    └── index.html            # Single-page web UI
```

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
| `pyqlib` | Quantitative investment platform |
| `akshare` | Chinese A-share market data |
| `torch` | Kronos model inference |
| `flask` | Web server |
| `tradingagents` | Multi-agent LLM trading framework |
| `huggingface_hub` | Model download |

---

## 🔒 Security Notes

- `data/llm_config.json` is **git-ignored** — never commit API keys
- `outputs/` is git-ignored — contains portfolio state and live orders
- `data/qlib_cn/` is git-ignored — large binary data files

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Kronos](https://huggingface.co/NeoQuasar/Kronos-small) — time-series forecasting model by NeoQuasar
- [QLib](https://github.com/microsoft/qlib) — Microsoft quantitative investment platform
- [TradingAgents](https://github.com/TauricResearch/TradingAgents) — multi-agent LLM trading framework by Tauric Research
- [AKShare](https://github.com/akfamily/akshare) — Chinese financial data library
