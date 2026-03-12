# CLAUDE.md — BoatRace Predictor

## Project Overview

Automated Japanese boat racing (競艇) prediction and betting system. Uses PyTorch multi-task neural networks with Kelly Criterion betting strategies. Deployed on Railway with a Streamlit dashboard for monitoring.

**Language**: Python 3.11.9 (runtime.txt)
**Deployment**: Railway (NIXPACKS builder)
**Database**: PostgreSQL (psycopg2)
**UI**: Streamlit (port 8501)

## Repository Structure

```
src/                        # Core business logic
├── models.py               # PyTorch multi-task model (1st/2nd/3rd prediction)
├── predictor.py            # RealtimePredictor & EnsemblePredictor
├── betting.py              # Kelly Criterion strategies, bet recommendations
├── scheduler.py            # DynamicRaceScheduler — main orchestration loop
├── database.py             # PostgreSQL connection, schema migrations
├── scraper.py              # HTML parser for boatrace.jp
├── features.py             # 208-dimensional feature engineering
├── collector.py            # Real-time exhibition data collection
├── result_collector.py     # Post-race result settlement
├── odds_estimator.py       # Market odds analysis
streamlit_app/              # Web dashboard
├── app.py                  # Main entry point (spawns scheduler daemon)
├── pages/                  # Multi-page Streamlit views (Japanese names)
├── components/             # Reusable UI: db_utils.py, mobile_css.py
scripts/                    # Utilities: training, evaluation, backtesting
models/                     # Pre-trained .pth model files
config/                     # betting_config.json, proxy_config.json
utils/                      # timezone.py (JST helpers)
docs/                       # scheduler_bug_report.md (known defects)
.github/workflows/          # scheduler-monitor.yml (hourly health check)
```

## Key Architecture

### Startup Flow
1. Railway runs Procfile: `TZ=Asia/Tokyo streamlit run streamlit_app/app.py`
2. `app.py` writes health marker to DB, spawns scheduler as daemon thread
3. `DynamicRaceScheduler` polls every minute during 10:00–21:00 JST
4. At 23:00+ JST, results are settled and statistics refreshed

### Prediction Pipeline (per race)
1. `collector.py` → scrape exhibition data (weather, tilt, parts)
2. `scraper.py` → scrape player stats / race roster
3. `features.py` → build 208-dim feature vector (16 global + 32×6 boats)
4. `predictor.py` → PyTorch inference → probabilities for 1st/2nd/3rd
5. `odds_estimator.py` → fetch live 3-combination odds
6. `betting.py` → Kelly Criterion recommendations across 6+ strategies
7. Insert bets into PostgreSQL

### Betting Strategies (A/B testing)
Defined in `config/betting_config.json`. Six active strategies with different Kelly fractions, EV thresholds, and filter types (none, divergence, entropy, ensemble, div_confidence). Dynamic odds adjustment by race number (R11-12 stable, R2-4 chaotic).

### Database Tables
- `races` — schedule, results, weather (UNIQUE on date+venue+race_number)
- `boats` — per-race boat/player data (FK to races)
- `predictions` — model output probabilities (JSONB)
- `bets` — placed bets with strategy, odds, result (win/lose), payout
- `model_performance` — per-strategy ROI tracking
- `scheduler_health` — heartbeat/status markers

Schema auto-migrates on startup via `database.py`.

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app locally
TZ=Asia/Tokyo streamlit run streamlit_app/app.py --server.port=8501

# Train model
python scripts/train_model.py

# Evaluate model
python scripts/evaluate_model.py

# Backtest strategies
python scripts/backtest_kelly_params.py
python scripts/backtest_ev_strategy.py

# Auto-retrain
python scripts/auto_retrain.py
```

## Environment Variables

```
DATABASE_URL=postgresql://user:password@host:port/database   # Required
TZ=Asia/Tokyo                                                 # Required
PORT=8501                                                     # Streamlit port
```

See `.env.example` for the template.

## Key Dependencies

- `torch==2.1.0` / `torchvision==0.16.0` — ML inference
- `numpy<2` — pinned for legacy compatibility (do NOT upgrade to numpy 2.x)
- `pandas>=2.1.0` / `scikit-learn>=1.3.0` — data processing
- `streamlit==1.42.0` — web dashboard
- `psycopg2-binary==2.9.9` — PostgreSQL driver
- `beautifulsoup4==4.12.2` / `requests==2.31.0` — web scraping
- `schedule==1.2.0` — task scheduling
- `plotly==5.18.0` — charts

## Code Conventions

### Language
- Code is in English; UI text, page names, and user-facing strings are in Japanese
- Comments mix English and Japanese

### Timezone
- All user-facing times: JST (Asia/Tokyo)
- Database timestamps: `TIMESTAMP WITH TIME ZONE`
- Use `utils/timezone.py` helpers: `now_jst()`, `format_jst()`

### Error Handling
- Graceful degradation: scraping failures fall back to default data
- 3-retry loops for network requests
- DB connections use context managers
- Health status written to `scheduler_health` table

### Database
- Use `psycopg2` with `RealDictCursor`
- UNIQUE constraints enforce race-level deduplication
- New columns added via auto-migration (ALTER TABLE IF NOT EXISTS pattern)

### Logging
- Standard Python `logging` module
- Print with `flush=True` for Railway log visibility

## Known Issues

Documented in `docs/scheduler_bug_report.md`. Key items:
- **No daily loss limit** across strategies (theoretical max ~6.9M yen/day)
- **Kelly formula** can produce extreme values when denominator approaches zero
- **Scheduler runs as Streamlit daemon thread** (should be separate worker)
- **Settlement fixed at 23:00 JST** (delayed races may be missed)

## CI/CD

- **GitHub Actions** (`scheduler-monitor.yml`): hourly cron during race hours, checks bet counts and scheduler health, alerts to Microsoft Teams webhook
- **Railway**: auto-deploy on push to main, auto-restart on failure (max 10 retries)

## Testing

- No pytest suite currently
- JavaScript test files exist (`test_dom.js`, `test_live.js`, `test_verify.js`) for UI verification
- Strategy validation via backtesting scripts in `scripts/`

## Important Notes for AI Assistants

1. **Do not upgrade numpy to v2** — torch 2.1.0 is incompatible; the `numpy<2` pin is intentional
2. **Model files in `models/`** are binary `.pth` files — do not modify or regenerate without explicit request
3. **`config/betting_config.json`** controls live betting behavior — changes have direct financial impact
4. **Scraping targets `boatrace.jp`** with SSL verification disabled — this is intentional due to certificate issues
5. **The scheduler is a daemon thread** inside Streamlit — be aware of thread-safety when modifying shared state
6. **Boats 5-6 predictions are excluded** from betting (historically unreliable)
7. **Race numbers affect strategy**: R11-12 use tighter odds bounds, R2-4 use wider bounds
