# Copilot Instructions — QuantMind (kronos-qlib-toolkit)

## Project Overview

QuantMind is an end-to-end quantitative trading toolkit for **Chinese A-shares** that combines:
- **Kronos Forecaster** — Transformer time-series model (HuggingFace) for price prediction
- **QLib Backtest** — Walk-forward backtest engine with transaction costs
- **TradingAgents** — Multi-agent LLM pipeline (Market → News → Sentiment → Fundamentals → Debate → Risk → PM → Final)
- **Consensus Engine** — Merges quant signal with AI qualitative decision
- **Paper Trading** — Virtual portfolio with lot-size enforcement and P&L tracking
- **LLM Wiki** — Notion-based personal knowledge base (ingest, query, lint, status)
- **Web UI** — Flask single-page app, bilingual (EN/ZH), dark-mode

## Tech Stack

- **Python 3.10+**, Windows-first (paths use `\`)
- **Flask 2.3** — web server (`app.py`, port 7080)
- **PyTorch** — Kronos model inference
- **pyqlib** — Microsoft's quant investment platform
- **akshare** — Chinese A-share market data
- **SQLite** — trade storage (`trade_storage.py`)
- **Notion SDK** — LLM Wiki pages (`llm_wiki.py`)
- **OpenAI-compatible LLMs** — supports OpenAI, DeepSeek, Qwen/DashScope, Ollama

## Key Paths

| Item | Path |
|------|------|
| Project root | `C:\qlik\tools\kronos-qlib-toolkit` |
| Flask app | `app.py` |
| Config dataclass | `config.py` → `ToolkitConfig` |
| LLM config | `data/llm_config.json` (git-ignored) |
| Wiki config | `data/llm_wiki_config.json` (git-ignored) |
| QLib data | `data/qlib_cn/` (git-ignored, large binary files) |
| Outputs | `outputs/` (git-ignored — charts, portfolios, orders) |
| Web UI | `templates/index.html` + `static/` |
| TradingAgents | External: `../TradingAgents` (editable pip install) |

## Architecture

```
app.py (Flask routes)
├── analysis_service.py    — backtest & prediction orchestration
├── kronos_engine.py       — model loader & inference
├── data_sources.py        — AKShare / QLib data adapters
├── strategy_catalog.py    — strategy definitions & signal logic
├── llm_service.py         — LLM config, proxy, T+0 analysis
├── trading_service.py     — paper & live trade execution
├── trade_storage.py       — SQLite persistence
├── trading_agents_service.py — TradingAgents async jobs, consensus
└── llm_wiki.py            — Notion wiki (standalone CLI)
```

## Coding Conventions

- **Dataclass config** — all runtime config in `ToolkitConfig` (slots=True), accessed via properties
- **Type hints** — use `from __future__ import annotations`, all function signatures typed
- **Path handling** — use `pathlib.Path`, not string concatenation
- **LLM abstraction** — all LLM calls go through `llm_service.py` or `llm_wiki.py:llm_call()`. Support both OpenAI-compatible and Ollama providers
- **Error handling** — Flask routes return `jsonify({"error": msg})` with appropriate HTTP codes
- **Bilingual** — UI strings support both English and Chinese
- **No secrets in code** — API keys live in `data/llm_config.json` and `data/llm_wiki_config.json`, both git-ignored

## Important Patterns

### LLM Provider Switching
```python
cfg = load_llm_config()  # from data/llm_config.json
# provider: "OpenAI" (covers OpenAI, DeepSeek, Qwen, any compatible) or "Ollama"
# base_url auto-appends /v1 for OpenAI-compatible endpoints
```

### Stock Symbol Handling
- Bare codes: `601169`, `000651`
- Shanghai suffix: `.SS` → `601169.SS`
- Shenzhen suffix: `.SZ` → `000651.SZ`
- Use `normalize_symbol()` from `data_sources.py` for conversion

### Paper Trading
- Lot size: 100 shares (A-share rule)
- Portfolio state: `outputs/paper_portfolio.json`
- Trade DB: `outputs/trading.sqlite3`
- P&L from closed trades feeds back into TradingAgents reflection memory

### LLM Wiki (llm_wiki.py)
- Standalone CLI: `python llm_wiki.py ingest|query|lint|status`
- Notion pages organized under categories: Sources, Concepts, Entities
- Existing pages are updated (appended), not overwritten
- Cross-references use `[[Topic Name]]` notation
- Source content truncated to 15K chars on ingest

## Git Workflow Rules

- **Never commit directly to `main` or `master` branch** — these are protected
- **Always create a new branch** for fixes or features (e.g., `fix/description`, `feat/description`)
- **All changes go through pull requests** — no direct pushes to the default branch

## Security Rules

- **Never commit** `data/llm_config.json` or `data/llm_wiki_config.json`
- **Never commit** `outputs/` contents (portfolio state, trade history)
- **Never log** API keys or tokens — mask them in debug output
- **Proxy-aware** — corporate proxy is restored before TradingAgents runs (`yfinance` needs it)

## Running the App

```powershell
cd C:\qlik\tools\kronos-qlib-toolkit
python app.py
# → http://127.0.0.1:7080
```

## Testing

- No formal test suite yet — test via the Web UI or CLI scripts
- `predict_stock.py` — CLI prediction test
- `single_stock_backtest.py` — CLI backtest test
- `python llm_wiki.py status` — verify wiki connectivity

## Common Tasks

### Add a new strategy
1. Define in `strategy_catalog.py` (add to `STRATEGIES` dict)
2. Implement signal logic function
3. UI auto-discovers from `strategy_options()`

### Add a new LLM provider
1. If OpenAI-compatible: just set `provider: "OpenAI"` and the correct `base_url`
2. If native SDK needed: add a new branch in `llm_service.py` and `llm_wiki.py:llm_call()`

### Add a new data source
1. Add adapter function in `data_sources.py`
2. Wire into `analysis_service.py` via the `source` parameter
