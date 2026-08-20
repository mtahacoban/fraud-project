# Fraud Detection & Investigation System

An explainable, leakage-free, human-in-the-loop fraud decision support system: a calibrated XGBoost model scores transactions, SHAP and an LLM explain why, and a precedent-learning layer can *propose* — never finalize — an automated decision.

## Highlights

- **Explainable by construction, not as an afterthought** — every score carries its SHAP factor breakdown and a plain-English LLM report; nothing is a black-box number with no rationale attached.
- **Calibrated, and proven so** — raw XGBoost probabilities are badly miscalibrated (Brier skill score **-14.78**); isotonic calibration fixes that (**+0.735**), which is what makes the automation gate's probability threshold a meaningful signal instead of noise.
- **Two AI layers that recommend, never decide** — a scoring layer (rules + XGBoost + calibration) and a precedent layer (k-NN over the analyst's own decision history) both stop at a suggestion; the decision form is the only thing that ever closes a case.
- **Human-confirmed automation, not auto-block** — a much stricter, versioned, five-gate policy can *propose* an action in `propose` mode, but every proposal requires an explicit analyst Approve. A blind shadow-mode measurement and a reject-rate circuit breaker exist specifically so that bar is evidence-based, not a guess.
- **Learns from outcomes** — every analyst decision is written back into a precedent index; the next similar case sees it. The whole loop (score → explain → decide → learn) is closed and observable end to end via the API.

## Metrics

Held-out test set, XGBoost vs. a logistic-regression baseline (same leakage-free features, same train/val/test split, `RANDOM_STATE=42`):

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| LogReg (F1-max) | 0.5650 | 0.9780 | 0.7824 | 0.4598 | 0.5792 |
| Isolation Forest (unsupervised) | 0.0361 | 0.8910 | 0.0548 | 0.4263 | 0.0971 |
| **XGBoost (production)** | **0.8508** | **0.9940** | **0.9021** | **0.7856** | **0.8398** |

**Calibration** (Brier score, lower is better — isotonic regression on the validation set, `n=554,082`):

| | Brier score | Brier skill score (vs. base-rate baseline) |
|---|---|---|
| Raw XGBoost | 0.0466 | -14.78 |
| **Isotonic-calibrated** | **0.0008** | **+0.735** |

PR-AUC is the primary metric (0.30% base rate makes ROC-AUC alone misleading). The Isolation Forest row exists to answer "why supervised, not unsupervised anomaly detection" — see [Baseline Comparison](#baseline-comparison--why-a-supervised-model) for why its precision is so low. Full evaluation methodology: `scripts/evaluate_model.py`, `scripts/analyze_calibration.py`, `scripts/compare_isolation_forest.py`.

## Architecture

Two AI layers work together, and neither one finalizes a decision on its own:

- **AI #1 — Scoring**: a hard/soft rule engine + calibrated XGBoost model produce a hybrid risk score and risk band; SHAP explains which features drove it; an LLM turns that into a short analyst report.
- **AI #2 — Precedent & Automation**: a k-NN model over the analyst's own decision history suggests a verdict for similar new cases; a much stricter, versioned policy gate can propose a fully automated action, but it can only ever be *proposed* — a human must click Approve before anything closes.

| Module | What it does | Key files |
|---|---|---|
| Data & model training | Leakage-free feature engineering, XGBoost fraud model | `scripts/train_models.py`, `scripts/evaluate_model.py`, `models/xgb_v1.pkl` |
| Explainability | SHAP — top contributing factors per transaction | `backend/explain.py`, `scripts/analyze_shap.py` |
| Calibration | Isotonic probability calibration | `backend/calibrate.py`, `scripts/analyze_calibration.py`, `models/xgb_v1_calibrated.pkl` |
| Backend API | FastAPI service, case management, hard/soft rule engine, hybrid score → risk band | `backend/main.py`, `backend/rule_engine.py`, `backend/scoring.py` |
| Frontend | React analyst console — Overview, Triage, Simulation, Evaluation, Automation Status | `frontend/src/` |
| LLM reporting | Findings → LLM (Groq) → async report; explains, never decides | `backend/findings.py`, `backend/llm_service.py`, `backend/report_worker.py` |
| Precedent suggestions (AI #2) | k-NN over analyst history → suggestion + LLM explanation; recommends, never decides | `backend/precedent.py`, `backend/precedent_worker.py` |
| Human-confirmed automation (AI #2) | Multi-gate policy → shadow/propose → human Approve/Reject; never auto-finalizes | `backend/automation.py` |

### How a transaction is handled, end to end

1. It arrives at `POST /score` (real traffic, the Simulation screen, or bulk seeding).
2. **Scoring**: XGBoost's raw probability is calibrated and cross-checked against the hard/soft rule engine to produce a hybrid score → risk band (RED/GRAY/GREEN). GREEN auto-clears with no case; RED/GRAY opens a Case.
3. **SHAP** explains which features drove the model's contribution to that score.
4. **LLM report**: `findings.py` turns SHAP + triggered rules + risk band into a deterministic sentence list — itself a complete report — which an LLM (Groq) optionally turns into 2-3 narrated sentences. Generated in the background; `/score` never waits on it.
5. **Precedent suggestion**: the case is vectorized (six model features + risk band) and compared via k-NN (cosine similarity) against every case already decided. Weak or thin evidence → silence ("insufficient precedent"). Enough close, agreeing precedents (≥5 count, ≥85% similarity, ≥70% consensus) → a suggestion, with its own LLM explanation.
6. **The analyst decides** — the decision form, informed but never bound by either AI layer.
7. **The system learns**: that decision is written back into the precedent pool, closing the loop — the next similar case sees it as precedent.
8. **Automation gate**, running in parallel: a *stricter* multi-gate policy (≥10 precedents, ≥95% similarity, ≥90% consensus, ≥95% calibrated probability, no conflicting hard rule) decides whether the case is even automation-eligible. In `shadow` mode this is measured but never shown. In `propose` mode, an eligible fraud-direction case gets a pending proposal — the analyst must explicitly Approve or Reject; doing nothing leaves it pending forever.
9. Every step is logged and traceable: `transactions`, `scores`, `shap_explanations`, `llm_reports`, `analyst_decisions`, `precedent_index`, `precedent_explanations`, `auto_block_log`, `automation_policy_versions`.

**The throughline:** every AI layer explains or recommends; only a human decides. Scoring never picks a verdict from a calibrated probability alone (the rule engine cross-checks it); the LLM never picks a verdict from findings (it only narrates them, and isn't called when there's nothing to narrate); the precedent/automation layer never finalizes a decision from precedent alone — a suggestion stops at a suggestion, and the automation gate stops at a pending proposal a human must confirm.

```mermaid
flowchart LR
    TX[Transaction] --> SCORE["POST /score"]
    SCORE --> RULES[Rule Engine]
    SCORE --> XGB[XGBoost + Calibration]
    RULES --> HYBRID[Hybrid Score → Risk Band]
    XGB --> HYBRID
    HYBRID -->|GREEN| CLEAR[Auto-cleared, no case]
    HYBRID -->|RED / GRAY| CASE[Case Opened]
    CASE --> SHAP[SHAP Explanation]
    SHAP --> LLM1[LLM Report]
    CASE --> PRECEDENT["Precedent k-NN Suggestion (AI #2)"]
    PRECEDENT --> LLM2[LLM Explanation]
    CASE --> GATE[Automation Gate]
    GATE -->|shadow| LOG[Logged, never shown]
    GATE -->|"propose + eligible"| PROPOSAL[Pending Proposal]
    PROPOSAL --> HUMAN[Analyst: Approve / Reject]
    CASE --> DECISION[Analyst Decision]
    DECISION --> LEARN[Precedent Index Updated]
    HUMAN --> LEARN
```

## Setup & Running

```bash
# 1) Clone, then set up the backend virtual environment
python -m venv venv
venv\Scripts\Activate.ps1      # Windows
source venv/bin/activate       # Mac/Linux
pip install -r requirements.txt

# 2) Environment config
cp .env.example .env           # defaults work as-is for local SQLite dev, no edits required
cp frontend/.env.example frontend/.env

# 3) Data + models
# Download the Kaggle "PaySim synthetic financial dataset" into data/ (gitignored).
# Train from scratch, or use the models/ checkpoints already in the repo:
python scripts/train_models.py
python scripts/analyze_calibration.py
```

**Run the backend** (FastAPI, from the repo root, venv active):
```bash
uvicorn backend.main:app --reload --port 8000
```
API docs at `http://127.0.0.1:8000/docs`. First startup creates `fraud.db` (SQLite) and runs all schema migrations automatically — no separate migration step.

**Run the frontend** (React + Vite, separate terminal):
```bash
cd frontend
npm install
npm run dev
```
Opens at `http://localhost:5173`, talking to the backend at the `VITE_API_URL` in `frontend/.env`.

**Optional — seed demo data** (backend must already be running):
```bash
python scripts/seed_demo_data.py        # bulk-scores a PaySim sample through POST /score
python scripts/seed_representative.py   # a smaller, curated set of representative cases
python scripts/backfill_precedents.py   # seeds precedent_index from any existing decisions
```

**Jupyter kernel** (for `notebooks/01_exploration.ipynb`, optional):
```bash
python -m ipykernel install --user --name fraud-venv
```

**PostgreSQL instead of SQLite** (same code, `DATABASE_URL` switches it):
```bash
docker-compose up -d db
# then set DATABASE_URL=postgresql+psycopg2://fraud_user:fraud_pass@localhost:5432/fraud_detection in .env
```

## Known Limitations

Honest gaps, not hidden ones — carried forward until deliberately addressed:

- **`source="live"` doesn't distinguish real traffic from seed data.** The field currently means "not from the Simulation screen" — both real production traffic and the bulk-seeding scripts (`seed_demo_data.py`, `seed_representative.py`) write `source="live"`. If this system were ever wired to an actual live feed, that feed and historical seed data would be indistinguishable by `source` alone. Fix is a straightforward third value (e.g. `"seed"`); not yet made because it's a deliberate decision, not a default one.
- **The demo dataset is seed-heavy, by design, not by accident.** Of 77 decisions in the current database, 70 (91%) carry a test/seed reason code — the rest are real analyst-style activity. This system is currently a populated demo dataset, not a clean-slate production log; anyone using the numbers in this README as evidence of live performance should know that up front.
- **`errorBalanceOrig` dominates the model's SHAP attribution** — roughly 6x more mean |SHAP| contribution than the next-highest feature. In practice, XGBoost here is close to a single-feature balance-consistency detector, not the "rich multi-factor" model the explainability framing elsewhere might suggest. This is a property of PaySim's clean ledger structure specifically and may not hold on real-world, noisier transaction data.
- **The automation circuit breaker only watches reject rate**, not post-confirmation reversal (a confirmed decision whose case is later reopened). That's a plausible secondary signal, deliberately not built yet — `reopen_case()`'s reason is free text today, not a structured flag distinguishing "this reopening contradicts an AI confirmation" from any other reason, and reopening is a slow, infrequent, behaviorally-confounded proxy in general.

Automation-specific limitations (bias inheritance, exposure-biased vs. blind agreement metrics, evidence-gated shadow→propose transition) are documented in depth in [Human-Confirmed Automation](#human-confirmed-automation) below — they're a large enough topic to earn their own section rather than a bullet here.

---

## Technical Details

### Tech Stack

**Backend** — Python 3.12, FastAPI, SQLAlchemy ORM, SQLite for local development / PostgreSQL for production (same code, `DATABASE_URL` switches it), XGBoost (fraud model), scikit-learn (logistic-regression and Isolation Forest baselines, k-NN precedent search, isotonic calibration), SHAP (`TreeExplainer`), Groq (Llama 3.1) for LLM generation with a deterministic fallback path.

**Frontend** — React 19, Vite, React Router, Recharts (SHAP charts), lucide-react (icons). Plain CSS with a small design-token system (colors, spacing, radii as CSS custom properties) — no UI framework.

**Verification** — no test suite; correctness is checked with live API calls against a running backend (`scripts/`, and manual `curl`/browser checks), `pyflakes` for the backend and `oxlint` + `vite build` for the frontend.

### Project Structure

- `notebooks/` — exploration and modeling notebooks
- `backend/` — FastAPI service and ML logic
- `frontend/` — React + Vite analyst console
- `scripts/` — training/calibration/seeding/analysis scripts (script equivalents of the notebooks, plus demo data seeding and offline validation)
- `data/`, `models/`, `reports/` — data, trained models, output charts

### Data Model

```mermaid
erDiagram
    TRANSACTION ||--o{ SCORE : has
    TRANSACTION ||--o{ RULE_HIT : has
    TRANSACTION ||--o{ SHAP_EXPLANATION : has
    TRANSACTION ||--o| CASE : opens
    CASE ||--o{ LLM_REPORT : has
    CASE ||--o{ ANALYST_DECISION : has
    CASE ||--o| PRECEDENT_INDEX : "indexed as"
    CASE ||--o{ PRECEDENT_EXPLANATION : has
    CASE ||--o{ AUTO_BLOCK_LOG : "evaluated by"
    AUTOMATION_POLICY_VERSION ||--o{ AUTO_BLOCK_LOG : governs
```

- **`transactions`** — every scored transaction, raw PaySim fields + derived model features + the five optional "Future Signals" fields (never read by scoring — see Simulation Screen below).
- **`scores`** — one row per scoring pass: ML score, rule score, hybrid score, risk band, calibrated probability, model version.
- **`rule_hits`** — every triggered rule, typed `hard` (fraud-direction, can force RED), `soft` (contributes to the hybrid score), or `clean` (Gate B veto signal only, never shown to an analyst — see Human-Confirmed Automation).
- **`shap_explanations`** — per-feature SHAP value, magnitude, and direction for a transaction.
- **`cases`** — opened only for RED/GRAY transactions; tracks status (`OPEN`/`CLOSED`), priority, and timestamps.
- **`llm_reports`** / **`precedent_explanations`** — cached natural-language text, tagged by `source` (`groq` or `fallback`).
- **`analyst_decisions`** — the append-only decision history a case can accumulate (including reopen/re-decide cycles); the record `precedent_index` and the agreement metrics are built from.
- **`precedent_index`** — one vector + label per *decided* case, the k-NN pool AI #2 searches.
- **`auto_block_log`** — one row per automation gate evaluation, whatever the outcome (`shadow`, `proposed`, `confirmed`, `rejected`, `withdrawn`).
- **`automation_policy_versions`** — every automation policy that has ever been active, append-only; the row a proposal cites never changes retroactively.

### API Reference

All request/response bodies are typed with Pydantic models (`backend/schemas.py`); the live OpenAPI schema is available at `/docs` (Swagger UI) or `/openapi.json` on any running instance.

**Scoring & Simulation**

| Method & Path | What it does |
|---|---|
| `POST /score` | Scores one transaction: rules + XGBoost + calibration + SHAP. Opens a case if RED/GRAY. |
| `GET /simulation/templates` | Four starter scenarios for the Simulation screen. |
| `POST /simulation/run` | Scores 1..N transactions (jittered variants of one input) through the identical pipeline `/score` uses. |

**Cases**

| Method & Path | What it does |
|---|---|
| `GET /cases` | Paginated, sortable, searchable case queue. Query params: `status` (`OPEN`/`CLOSED`/`AUTO_CLEAN`), `risk_band` (`RED`/`GRAY`/`GREEN`), `q` (exact case or transaction ID), `sort` (`hybrid_score`/`created_at`), `order` (`asc`/`desc`), `limit` (≤200, default 50), `offset`. Returns `{items, total}`. |
| `GET /cases/{id}` | Full case detail — transaction, score, rule hits, SHAP factors, decision history. |
| `POST /cases/{id}/decision` | Records an analyst decision (`confirm_fraud`/`approve_clean`/`escalate`) and closes the case. |
| `POST /cases/{id}/reopen` | Reopens a closed case; prior decisions stay in history. |
| `GET /cases/{id}/report` | The LLM/deterministic case report. `{"status": "generating"}` while pending — poll until `"ready"`. |
| `GET /cases/{id}/precedents` | Nearest-neighbor precedents, the consensus summary, and its LLM explanation. |
| `GET /cases/{id}/pending-ai-decision` | Whether a pending automation proposal exists for this case (`null` if not). |
| `POST /cases/{id}/confirm-ai-decision` | Approves a pending proposal — the only way one becomes a real, closed decision. |
| `POST /cases/{id}/reject-ai-decision` | Rejects a pending proposal (`rejection_reason` required); the case stays open for a normal decision. |

**Model, Metrics & Automation**

| Method & Path | What it does |
|---|---|
| `GET /model-info` | Training metadata, calibration stats, and the baseline model comparison. |
| `GET /metrics` | Dashboard KPIs — scored/case counts by status and risk band, pending AI proposal count, active automation mode. |
| `GET /automation/status` | Active policy thresholds, blind shadow-agreement stats, reject rate, bias monitoring, circuit breaker state. |

### Key Principles

PR-AUC is the primary metric; synthetic fields never enter the model; the LLM never makes decisions; there is no fully automatic block.

### Analyst Console (Frontend)

A React + Vite single-page app (`frontend/src/`) — five screens behind a persistent sidebar, all reading from the API above:

| Screen | Route | Purpose |
|---|---|---|
| Overview | `/` | Landing page — scored/case KPIs, a live count of pending AI proposals (linking straight to Automation Status), risk-band distribution, and the five highest-risk recent cases. |
| Triage | `/triage` | Case queue and case detail in one split view — filterable/searchable/sortable list on the left, full case detail (score, SHAP, rules, LLM report, AI #2 precedent suggestion, any pending automation proposal, decision form, history) on the right. Deep-linkable (`/triage?case=182`); selecting a case never leaves the queue. |
| Simulation | `/simulation` | Build or bulk-generate transactions and score them live — see below. |
| Evaluation | `/evaluation` | Model metrics, calibration, and the baseline comparison charts. |
| Automation Status | `/automation` | The active policy's thresholds, blind shadow-agreement rate, reject rate, circuit breaker state, and bias-monitoring breakdown. |

**Case queue pagination, in numbers:** with a live case count in the low thousands (2,132 open cases in the current snapshot — see Current State below), fetching the entire queue on every page load doesn't scale. `GET /cases` does the filtering, sorting, and searching in SQL and returns one page at a time; the frontend only ever holds 50 rows in memory. The same endpoint backs both the Overview's "top 5 by risk" query and the full Triage queue — one code path, no duplicated sorting logic between screens.

### Simulation Screen

The **Simulation** screen (`/simulation`) lets an analyst build or bulk-generate transactions and watch the model score them live, without needing a real transaction feed.

- **Endpoints:**
  - `GET /simulation/templates` — four starter scenarios (`account_drain`, `normal_transfer`, `borderline`, `high_amount_cashout`) that pre-fill the form.
  - `POST /simulation/run` — takes `{transaction, count}`; scores `count` transactions (1 jittered variation of `transaction` for `count>1`) through the exact same `ScoringEngine`/rule engine used by `/score`, via a shared `_write_score()` helper — no scoring logic is duplicated or altered.
- **`is_demo` / `source` flag:** every transaction the simulator writes is tagged `source="simulator", is_demo=True` (vs. `"live"`/`False` for everything else, including `/score`). This makes simulator traffic filterable (`WHERE is_demo = true`) without touching how real transactions are scored, stored, or queued.
- **No label leakage:** the simulator, its templates, and its endpoints never accept or send `isFraud`. Whether a simulated transaction is flagged as fraud is entirely up to the model + rule engine, exactly as it is for real transactions — templates only pre-fill input fields, never the outcome.
- **"Future Signals" panel — extensibility, not a new capability:** `TransactionIn` has always carried five optional fields (`device_id`, `is_known_device`, `login_country`, `geo_velocity_flag`, `channel`) — commented "synthetic, UI/LLM only, never enters scoring/rules/automation" in `backend/schemas.py`, already stored on every `Transaction`, and already shown on Case Detail labeled "(synthetic)". The Simulation form lets an analyst set these alongside the real fields, in a visually distinct block (dashed border, muted text — deliberately neither the blue "AI explained this" styling used elsewhere nor the amber "needs your attention" styling), captioned plainly: *"These signals are illustrative — a production system would incorporate device fingerprinting and geo-velocity; they are not used in the current model or score."* This demonstrates the architecture's extensibility without pretending synthetic data carries real risk information — a real deployment could wire these into scoring; this system deliberately does not.

  **Verified never scored, not assumed:** `backend/scoring.py`, `backend/rule_engine.py`, and `backend/precedent.py` were checked for every one of these five field names — zero references. Confirmed live: the same transaction sent once with all five fields empty and once with deliberately suspicious values (`geo_velocity_flag=1`, an unrecognized `device_id`, a different `login_country`/`channel`) produced bit-identical scoring results, while the values themselves were confirmed persisted to the transaction row (so they reach Case Detail) — proving the fields flow end-to-end without ever touching a decision.

### LLM-Generated Case Reports

Every RED/GRAY case gets a short, natural-language analyst report — generated in the background, never blocking scoring, and always available even with no LLM configured at all.

**Flow:** `findings → LLM → async storage → query`
1. **Findings** (`backend/findings.py`, `build_findings()`) — a deterministic, LLM-free function that turns a case's SHAP top-5, triggered rules, risk band, band reason, and calibrated probability into a list of short factual sentences. This list is a fully valid report on its own; the LLM step is not required for the system to explain a case.
2. **LLM** (`backend/llm_service.py`, `generate_report(findings, txn_summary) -> {text, source}`) — a provider-abstract interface. The prompt instructs the model to use *only* the given findings (no invented numbers/facts) and to explain, never decide. Only `groq` is implemented today, selected by `LLM_PROVIDER`; adding another provider means adding one function and one registry entry, no call-site changes. On any failure — no key, invalid key, rate limit, timeout, missing package, any exception — the call returns `None` and `generate_report()` falls back to the `findings` text with `source="fallback"`. Nothing raises, nothing crashes, the API key is never logged.
3. **Async storage** (`backend/report_worker.py`) — report generation runs in a FastAPI `BackgroundTask`, after the triggering response has already been sent, so `/score` and `/simulation/run` latency is unaffected. Triggered two ways, both funneling through the same idempotent `ensure_report_generation()` so a case is generated at most once: proactively when a case is opened, and lazily on `GET /cases/{id}/report` if no report exists yet.
4. **Query** — `GET /cases/{id}/report` returns `{"status": "ready", "report": {...}}` or `{"status": "generating", "report": null}`. Case Detail polls this every 2s while generating, up to 10 tries (~20s), then shows a "refresh to check" message rather than polling forever.

**Principle: the LLM explains, it never decides.** The risk band, hybrid score, and calibrated probability come entirely from the rule engine + XGBoost + calibration pipeline, unchanged by this layer. The LLM only rephrases what's already been decided into readable prose — it cannot see raw transaction data, only the same findings the fallback text is built from, so its report can't say anything the deterministic path couldn't already say.

**`.env` variables:**
```
LLM_PROVIDER=groq              # only "groq" is implemented; anything else always falls back
LLM_API_KEY=                   # blank = fallback-only, no external calls, no key required
LLM_MODEL=llama-3.1-8b-instant # provider-specific model name
```

**Source badges (Case Detail):** each report shows where it came from — a blue "✨ AI-generated" badge for a real LLM response (`source="groq"`), or a neutral gray "Auto-summary" badge for the deterministic fallback (`source="fallback"`). Fallback is styled neutrally on purpose, not as a warning/error — it's a fully valid report, not a degraded one.

**Provider-agnostic by design:** this deployment uses Groq's free tier (Llama 3.1) for latency; the same interface would work with a locally-hosted model for deployments where sending case data to a cloud API isn't acceptable, without touching anything upstream of `generate_report()`. With no key configured at all, the system still produces a complete, deterministic report for every case — there is no "LLM required" failure mode.

### Precedent-Based Suggestions — AI #2

**AI #2 recommends, it never decides.** It learns from the analyst's own decision history and surfaces similar past cases as context; the existing decision form (Approve / Confirm Fraud / Escalate) is completely unchanged, sits right next to AI #2's panel, and is the only thing that ever closes a case. Automatic action on a suggestion — with the safeguards that make that safe (shadow mode, multi-condition gates, an asymmetric cost model, a circuit breaker) — is a separate, stricter layer covered below.

#### How it learns

1. **Vectorize** (`backend/precedent.build_case_vector()`): the same six model features XGBoost and SHAP use (`amount, step_hour, errorBalanceOrig, errorBalanceDest, is_transfer, is_cashout`) plus a risk-band one-hot (RED/GRAY/GREEN — GREEN is structurally always 0 since GREEN transactions never open a Case), scaled with a `StandardScaler` fit once on the initial backfill and persisted to `models/precedent_scaler.pkl`. Every later vectorization reuses those exact parameters; nothing is ever refit against live data, so there's no leakage and no drift between old and new vectors. SHAP values are deliberately left out of the vector — see Known Limitations below.
2. **Index** (`precedent_index` table): only decided cases (`case.status == "CLOSED"`) are eligible — a case's row stores its vector + the analyst's current decision as `label`. `scripts/backfill_precedents.py` seeds this from decision history; going forward, `backend.main.decide_case()` calls `precedent.add_to_precedent_index()` itself, right after each decision. Reopening a case leaves its precedent entry untouched until it's genuinely re-decided, at which point the existing row is updated in place — one case, one entry, never a duplicate.
3. **Query** (`precedent.find_precedents()`): `sklearn.neighbors.NearestNeighbors(metric="cosine")` over the whole pool, k=15, with self-exclusion applied before the index is even fit (a case can never be its own neighbor).

#### Confidence gates — when a suggestion is allowed to exist

`precedent.summarize_precedents()` is pure arithmetic over what k-NN found — no LLM, no judgment call. Two thresholds work together:

- **Similarity floor (0.5):** a neighbor below this doesn't get a vote at all — k-NN always returns up to k results even if the pool runs out of genuinely similar cases, and voting on a barely-related case would quietly pollute the recommendation. Set from observed data: a tight, genuinely-similar cluster sat at 0.88–0.91 cosine similarity, while a neighbor pulled in just to pad out to k=15 sat at 0.53.
- **Three-gate threshold**, all required: `precedent_count >= 5`, `avg_similarity >= 0.85`, `consensus_ratio >= 0.70` — the top decision's plurality share among floor-cleared neighbors, by raw count (every counted vote weighted equally, not similarity-weighted — "11 of 15 confirm_fraud, 73%" is auditable in a way a weighted score isn't). Any gate failing → `suggested_decision=None`, "insufficient precedent — use judgment". `escalate` is a real third label throughout (not folded into fraud/clean) — a suggestion of "escalate" is a valid, first-class output, though it can never itself become an automatic action, since escalating *is* "send this to a human."

#### The LLM's role — explains, never decides

`precedent.explain_precedents()` turns the deterministic summary into 2-3 plain-English sentences, reusing the report layer's exact transport unchanged (`llm_service.generate_report()` — same Groq client, provider registry, async safety, timeout, fallback chain, key handling) with its own system prompt and its own findings-builder. The LLM is given only the deterministic summary — never raw transaction data — so it cannot say anything the deterministic path couldn't already say. No suggestion → no LLM call at all: with `suggested_decision=None` there's nothing to explain, so `explain_precedents()` returns the deterministic "insufficient precedent" text directly, at zero API cost. When there is a suggestion, the explanation is generated once in a `BackgroundTask` and cached.

#### Cold start

An empty `precedent_index` or a missing `models/precedent_scaler.pkl` both mean the same thing to every code path that touches AI #2: stay silent, don't crash, let the rest of the system work exactly as before. `GET /cases/{id}/precedents` returns an empty neighbor list and "insufficient precedent"; `decide_case()` still records the decision normally, it just skips the precedent-index write.

#### Bias limitation — read before trusting a suggestion

**AI #2 imitates the analyst; it does not audit the model, the rules, or the analyst.** A suggestion is a plurality vote over this analyst's own past calls — if those past calls carried a systematic bias or blind spot, AI #2 will faithfully learn and reproduce it, presented with the same confidence as a genuinely well-calibrated pattern. Nothing here distinguishes "the analyst was consistently right" from "the analyst was consistently wrong in a consistent way." This is a real, unresolved limitation, a direct consequence of learning from decisions rather than from ground truth — and exactly why this layer stops at a suggestion and never automates on its own: a human stays in the loop specifically to catch what precedent alone cannot.

#### Agreement metric — read this before citing a percentage

`backend/precedent.compute_agreement_stats()` computes how often AI #2's suggestion matches what the analyst actually decided, in two distinct ways that answer different questions:

- **`retrospective=False` ("recorded")** — uses `analyst_decisions.ai2_suggested_decision`, captured live at the exact moment each decision was made (the suggestion the analyst could actually have seen). Methodologically the honest number for "did analysts follow AI #2" — but only exists for decisions made after this capture was added; earlier decisions are correctly `NULL` and excluded, not backfilled with a guess.
- **`retrospective=True`** — ignores the recorded column and recomputes AI #2's suggestion for every closed case against today's full pool. Answers a different question — "does AI #2's current judgment agree with past human calls" — and is not what any analyst was actually shown, since the pool has grown since most decisions were made.

**Neither number is an unbiased measurement.** Because the suggestion is visible in the UI when the analyst decides, the "recorded" agreement rate is an upper bound inflated by exposure — an analyst who partly trusts a visible suggestion drifts toward it, which is not the same as the suggestion being independently correct. A trustworthy estimate requires a blind/shadow measurement, computing the suggestion without showing it to the analyst and comparing after the fact — see Human-Confirmed Automation below for that.

`classify_agreement()` treats escalate as a fourth outcome, not folded into agree/disagree: `agree`, `disagree` (opposite fraud/clean verdicts — the real contradiction), `analyst_escalated` (analyst was more cautious than the suggestion), `analyst_decisive` (analyst gave a verdict where AI #2 only suggested escalation), `no_suggestion` (excluded from the rate's denominator).

**Traceability:** every decision is identifiable by `analyst_reason_code` — seed/test data is clearly tagged (e.g. `seed_cluster_decision` for the cluster-seeded bootstrap batch) and distinguishable from real decisions. Nothing in this database was written directly to storage; every row went through the actual API endpoints (`POST /score`, `/decision`, `/reopen`, `/confirm-ai-decision`, `/reject-ai-decision`) exactly as a real analyst or a real transaction feed would produce it.

*Future work (not correctness gaps, noted for scale):* cached explanations are invalidated on any new decision anywhere in the pool — simple and always correct, but would get expensive at a much larger pool size, where finer invalidation (only explanations whose actual top-k neighbor set changed) would pay for itself. Similarly, the similarity floor and confidence thresholds were set from a modest precedent sample — worth revisiting once real usage produces enough volume to calibrate against instead of extrapolate from.

### Human-Confirmed Automation

**No decision ever finalizes without a human clicking Approve — this is the one rule everything else here serves.** This layer builds a *second*, much stricter gate on top of the precedent suggestion: deterministic, versioned, logged, and — even when every condition is met — never self-executing. There is no code path anywhere in this system that closes a case without either the analyst's own decision or the analyst's explicit confirmation of an AI proposal.

#### Staged rollout: off → shadow → propose (there is no "full-auto")

`automation_policy_versions.mode` has exactly three values, and the system ships in the most conservative one:

- **`off`** — nothing evaluated, nothing logged. The default absence of automation.
- **`shadow`** — every decided case is evaluated against the gate and logged to `auto_block_log` (`review_status="shadow"`), but never surfaced anywhere the analyst can see it. This is how evidence accumulates before automation is trusted with anything visible. v1, the seeded default, starts here.
- **`propose`** — an eligible, fraud-direction case gets a pending proposal in Case Detail. Still not automation in the sense of taking action — it's a recommendation with a much higher bar than a plain suggestion, requiring explicit human sign-off.

There is no fourth mode that skips the analyst.

#### The multi-gate policy — five gates, all required

`automation.evaluate_auto_decision()` is pure arithmetic (no LLM, no side effects) over `precedent_summary` (reused as-is from the precedent engine — never recomputed), `calibrated_proba`, and `hard_rule_hits`:

1. **Direction is automatable** — see Asymmetry below.
2. **`avg_similarity >= policy.fraud_similarity_threshold`** (0.95 in the safe default policy).
3. **`precedent_count >= policy.min_precedent_count`** (10).
4. **`consensus_ratio >= policy.min_consensus_ratio`** (0.90).
5. **`calibrated_proba >= policy.min_calibrated_proba`** (0.95) — **Gate A**, the actual "scoring engine and precedent engine agree" check: the precedent consensus and the independently-calibrated probability are two different signals derived from completely different evidence (analyst history vs. the trained model), and both have to independently clear a high bar.
   Plus **Gate B**, hard-rule conflict — a real bidirectional filter: `policy.hard_rule_required=False` (default) means "no rule actively contradicts this" and passes by default; `hard_rule_required=True` tightens it to "a hard rule must have actually fired." Independent of that flag, Gate B unconditionally fails whenever AI #2's direction is `"fraud"` *and* `rule_engine.py`'s `clean_confirmed` rule also fired for the same transaction — a direct contradiction between "AI #2 says fraud" and "a rule strong enough to say this looks routine."

Every threshold is read from the active policy — never hardcoded — and `evaluate_auto_decision()` returns the full rationale (`reason`: every gate's outcome; `failed_gates`: just the failures, human-readable with the actual value vs. the required one, e.g. `"consensus: consensus_ratio=0.80 < 0.90 required"`) regardless of the verdict, which is what both shadow logging and the Case Detail proposal card display.

In the seed pool, roughly 1,500 cases receive a precedent suggestion under its looser gates, but under this layer's much stricter gates zero cleared all five under the safe default policy (the strongest fraud candidate found was still capped by consensus below the required threshold — a real, honest ceiling, not a bug). The mechanism was proven end-to-end by deliberately, temporarily loosening a policy version (documented in that version's own `notes` field, always reverted to the strict default immediately after) — the correct way to demonstrate a conservative gate: prove the mechanism works, then leave it in its conservative resting state.

#### Gate B's `clean_confirmed` rule — design and isolation

The rule engine's only hard rule used to be `ghost_destination` (fraud-direction only), so `hard_rule_required=False`'s default "no conflict" always passed by construction and `hard_rule_required=True` had nothing but `ghost_destination` to require. `clean_confirmed` adds a genuine clean-direction signal, so Gate B has something real to check both ways.

**Design:** `clean_confirmed` fires when `oldbalanceOrg > 0 AND amount <= 0.5 * oldbalanceOrg AND` the destination is *not* a ghost account (`oldbalanceDest`/`newbalanceDest` not both zero — the exact logical negation of `ghost_destination`). In words: the transaction moves a modest fraction of the source balance (the opposite of the `drain_account` soft rule) to an account that already has a real balance history. Verified on train+val (TRANSFER/CASH_OUT only, the same methodology used for every other rule's documented "lift"): coverage 5.63%, fraud rate 0.0008% against a 0.30% baseline — a 370x reduction, symmetric to `ghost_destination`'s 237.8x lift in the fraud direction.

A third candidate condition — "balance consistency" (`errorBalanceOrig ≈ 0`) — was measured and dropped before being added: it had a lift of 9.78x, the opposite of a clean signal, because PaySim's full-drain fraud pattern is itself exactly balance-consistent. Including it would have mislabeled a real fraud signature as clean-confirmed.

**Architectural isolation:** `clean_confirmed` is computed by its own function, `rule_engine.check_clean_confirming_rules()`, entirely separate from `check_hard_rules()`, which drives `hard_rule_flag`/`compute_hybrid_score()` and is untouched. `clean_confirmed` is persisted (`RuleHit.rule_type="clean"`, since Gate B is evaluated later from a background task that re-queries the database) but is read by exactly one consumer: `automation.evaluate_auto_decision()`'s Gate B. It never reaches scoring, calibration, or any risk-band decision — verified live, bit-for-bit identical scores before and after this rule was added. It's also excluded from every `rule_hits`/LLM-report response (`rule_type="clean"` is filtered out everywhere else) — it exists solely to inform Gate B, not as an analyst-facing signal.

**Proof the gate is decisive, not just present:** a constructed test case with the same feature signature as a confirmed-fraud precedent (99.99% similarity) but a non-ghost destination and a modest source-balance fraction triggered `clean_confirmed`. Holding every other gate constant (via a temporarily loosened, fully auditable policy version, reverted immediately after): without `clean_confirmed`, `eligible=True, failed_gates=[]`; with it, `eligible=False, failed_gates=["hard_rule_conflict: clean-confirming hard rule triggered (['clean_confirmed']) — contradicts AI #2's fraud suggestion"]` — same case, same policy, only this rule flipped the outcome.

#### Asymmetry — only fraud is ever proposed

- **`confirm_fraud`** direction: the only one that can ever become `eligible=True`.
- **`approve_clean`** direction: blocked unconditionally by `policy.auto_clean_enabled` (`False` in every policy version so far) — auto-clearing a case carries a different risk profile (a missed fraud vs. a delayed review) that this system deliberately doesn't take on in v1.
- **`escalate`** direction: blocked unconditionally, with no policy flag to override it — escalating already means "send this to a human"; automating that would be a contradiction, not a feature.

#### Human confirmation — the mechanism, not just the principle

- **`POST /cases/{id}/confirm-ai-decision`** — the only way a proposal becomes a real decision. Writes an `AnalystDecision` with `ai_proposed=True` and `ai_proposal_id` pointing at the exact `auto_block_log` row (and, through it, the exact policy version) that produced it. `auto_processed` stays `False` — always, everywhere — because nothing here ever processes itself; a human processed it, an AI proposed it. The confirmed decision is written into `precedent_index` exactly like any other.
- **`POST /cases/{id}/reject-ai-decision`** — `rejection_reason` is required at the schema level. Rejecting does not decide the case: the proposal is marked `"rejected"` and the case stays `OPEN` — the analyst then calls the ordinary decision endpoint with their own real verdict.
- **Doing nothing** — the proposal simply stays `"proposed"`, the case stays `OPEN`, indefinitely. No timeout, no default action, no auto-confirmation.

#### Rubber-stamping monitoring

A confirm/reject button pair only resists rubber-stamping if declining is genuinely easy and its rate is actually watched:

- **Friction on reject, not on approve** — a mandatory reason is the only asymmetry in the UI interaction; the buttons themselves are the same size, same component, same visual weight, specifically so neither is the "path of least resistance" by default styling.
- **`automation.reject_rate_stats(db, policy_version_id=...)`** — confirmed vs. rejected among one policy version's own proposals, gated by a minimum sample size: below that, it reports `reject_rate=None` + `"insufficient data"` rather than a statistically meaningless percentage. Global and per-policy views are both supported.

#### Circuit breaker

`automation.check_circuit_breaker()` compares the active policy's own reject rate against `circuit_breaker_max_reversal_rate`, checked at the one place a reject rate can actually move: right after a rejection commits. If tripped, it calls `activate_new_policy_version()` — the same versioning function a deliberate manual change would use — to downgrade `mode="propose"` to `"shadow"`, with `notes` recording exactly why. The breaker only downgrades *mode* — it does not also adjust the other gate thresholds, which stay whatever the tripped policy had; a human decides what the right thresholds are before ever re-enabling `propose`.

**Signal choice: reject rate only, not post-confirmation reversal.** A confirmed decision whose case is later reopened would be another plausible signal, but it's deliberately not implemented yet: `reopen_case()`'s reason is free text, not a structured flag distinguishing "this reopening contradicts an AI confirmation" from any other reason, and reopening is a slow, infrequent, behaviorally-confounded proxy. Reject rate is immediate, unambiguous, and already cleanly measured. This is a documented scope boundary, not a gap left unmentioned.

#### Policy versioning and audit trail

Every threshold change — deliberate tuning or an automatic circuit-breaker trip — is a new row, never an edit in place. `get_active_policy()` always reads whichever single row has `active=True`, so the complete threshold history survives with a `notes` explanation for each change. Critically, a proposal stays permanently bound to the policy version that produced it (`auto_block_log.policy_version_id`), not to whatever's active later — the concrete answer to "which threshold approved this, and is it still in force?" that any audit of an automation system has to be able to give.

#### Bias limitation (inherited, not resolved)

This gate is stricter than the plain precedent suggestion, but it inherits the same blind spot: it still learns from analyst decisions, not from ground truth. A systematic bias in the analyst history would clear this higher bar just as confidently as a genuinely well-calibrated pattern — a higher confidence threshold filters for *consistency*, not *correctness*. Gate A helps, since it demands an independent, model-derived signal agree too — but the model itself was trained on historical fraud labels, which carry their own labeling process and potential bias. No layer here currently audits for demographic or protected-attribute fairness (the dataset doesn't carry those attributes). `automation.bias_monitoring_stats()` is a minimal smoke-check (confirmed/rejected by transaction type) — an alarm bell, not a certification.

#### Shadow-mode neutrality vs. the suggestion agreement metric

The precedent engine's `compute_agreement_stats()` and automation's `shadow_agreement_stats()` sound similar and measure genuinely different things:

- **Suggestion (recorded) agreement**: the analyst could see the suggestion in the UI while deciding — any resulting agreement rate is an upper bound inflated by exposure.
- **Shadow agreement**: the automation verdict is computed and logged, but genuinely never shown to the analyst before or during their decision — `log_shadow_evaluation()` writes to `auto_block_log` only, and `GET /automation/status` only ever reports an aggregate, never a per-case value. This is what makes it a legitimate blind comparison rather than another exposure-biased number.

Only the shadow number is evidence a real deployment could act on.

#### Conscious limits (backlog, not gaps left unmentioned)

- **Reversal-rate signal**: not implemented (see Circuit Breaker above) — `reopen_case()` needs a structured "this reopening contradicts an AI confirmation" flag before it could be a reliable secondary trigger.
- **Shadow-to-propose transition is evidence-gated, not evidence-complete**: `shadow_agreement_stats()`'s blind measurement only becomes statistically meaningful with real volume and time. This system proves the *mechanism* end-to-end (shadow logs, aggregates, gates, the propose pathway, confirm/reject, the circuit breaker), not that today's shadow numbers alone justify flipping to `propose` in a live deployment. That gating condition is intended behavior, not an oversight.
- **Circuit breaker threshold values** (`max_reversal_rate=0.20`, `min_confirmations=5`) are reasonable starting points proven to behave correctly at both ends, not values validated against real production volume — they're policy fields specifically so they can be revised without a code change once real usage exists to calibrate against.

### Performance

`POST /score` end-to-end latency (feature prep → XGBoost → rule engine → SHAP → calibration → DB write; the async LLM report is excluded — `/score` never waits on it). Measured with `scripts/measure_latency.py`, N=150 requests, TRANSFER/CASH_OUT rows only, after a 10-request warm-up:

| | avg | median | p95 | p99 |
|---|---|---|---|---|
| Run 1 | 121.0 ms | 94.3 ms | 282.2 ms | 470.3 ms |
| Run 2 | 113.1 ms | 103.8 ms | 201.6 ms | 420.0 ms |

Faster than the 143 ms reference-paper benchmark this approach is compared against. The wider tail reflects SQLite's single-writer lock serializing against this same load's own background tasks on a single dev process, not the scoring pipeline itself — PostgreSQL (already supported) would not serialize writes this way.

### Baseline Comparison — why a supervised model

`scripts/compare_isolation_forest.py` adds an unsupervised Isolation Forest baseline evaluated on the exact same split as the production model — a comparison artifact only, touching nothing under `backend/`. Isolation Forest's precision (5.48%; 3.53% among the top-*K* most anomalous transactions, a threshold-free check) is far below usable: with no access to labels, it flags transactions as anomalous purely on how unusual their raw feature values are, which large *legitimate* transfers also produce. Its ROC-AUC (0.891) shows it ranks fraud better than random overall, but PR-AUC and top-of-ranking precision — what matters under a 0.30% base rate — are poor. This is the concrete argument for the supervised approach this system is built on.

### Current State

*A live system's row counts move with every subsequent decision — treat the numbers below as a point-in-time snapshot; `git log` or a fresh `/metrics` and `/automation/status` call are authoritative for the current state.*

```
transactions:              6,382
cases:                     2,207   (75 CLOSED, 2,132 OPEN)
precedent_index:              75   (== CLOSED cases, exactly)
analyst_decisions:             77   (includes reopen/re-decide history)
llm_reports:                   15
precedent_explanations:        12
auto_block_log:                15   (7 shadow · 5 rejected · 2 confirmed · 1 proposed)
automation_policy_versions:    13   (deliberate test loosenings and reverts, one auto-trip — active policy at safe defaults)
```
