# Deploy QuantMind to Hugging Face Spaces

## Prerequisites

- [Git](https://git-scm.com/) installed
- [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) logged in (`huggingface-cli login`)
- Push access to the HF Space repo (`qicaixin/ai-helper`)

## One-Time Setup

### 1. Add the HF Space as a git remote

```bash
cd C:\qlik\tools\kronos-qlib-toolkit
git remote add hf https://huggingface.co/spaces/qicaixin/ai-helper
```

Verify remotes:

```bash
git remote -v
# origin  https://github.com/zbcui/quantmind.git (fetch/push)
# hf      https://huggingface.co/spaces/qicaixin/ai-helper (fetch/push)
```

### 2. Key files required by HF Spaces

| File | Purpose |
|---|---|
| `Dockerfile` | Tells HF how to build and run the app (python:3.12-slim, port 7080) |
| `README.md` | YAML frontmatter at the top configures the Space (`sdk: docker`, `app_port: 7080`) |
| `app.py` | Flask must bind to `0.0.0.0` (not `127.0.0.1`) for Docker container access |

**README.md frontmatter:**

```yaml
---
title: QuantMind AI Helper
emoji: 🔮
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7080
---
```

## Deploy (Every Time)

### Step 1 — Commit and push to GitHub

```bash
git add -A
git commit -m "your commit message"
git push origin master
```

### Step 2 — Push to HF Spaces

```bash
git push hf master:main --force
```

> **Note:** HF Spaces uses `main` as the default branch, while our GitHub repo uses `master`. The `master:main` refspec handles the mapping. `--force` is used because the HF repo may have diverged.

### Step 3 — Monitor the build

Open https://huggingface.co/spaces/qicaixin/ai-helper and watch the **Build** logs. The Docker build typically takes 3–5 minutes (mostly pip installing torch/pyarrow).

Once the status shows **Running**, the app is live.

## Troubleshooting

| Issue | Solution |
|---|---|
| Build fails on `pip install` | Check `requirements.txt` for version conflicts. HF Spaces uses Linux — Windows-only packages won't work. |
| App shows "Building" forever | Click the **Logs** tab on the Space page to see build errors. |
| App starts but shows blank/error | Check **Container Logs** on HF. Common issue: `host="127.0.0.1"` instead of `host="0.0.0.0"` in `app.py`. |
| `git push hf` asks for credentials | Run `huggingface-cli login` and enter your HF token. |
| Port mismatch | Ensure `app_port` in README.md YAML matches the port in `app.py` (currently both 7080). |

## Architecture Notes

- **GitHub** (`origin`) is the source of truth for code
- **HF Spaces** (`hf`) is the deployment target — receives force-pushes from master
- The Dockerfile copies the entire repo into `/app` and runs `python app.py`
- SQLite DB and outputs are ephemeral in the container (reset on each redeploy)
- Secrets (API keys) should be configured per-user via the Settings UI after login
