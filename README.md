# CFRS — Collaborative Filtering Recommendation System

Matching farmers and buyers in agricultural markets using collaborative
filtering + produce/location preference in a tunable hybrid model.

## Stack
Django + DRF · Surprise · scikit-learn · pandas/numpy · SQLite (MySQL-portable) · HTML/CSS/JS + Chart.js

## Quick start
Requires **Python 3.11–3.13** (tested on 3.11 and 3.13; dependency versions are pinned in
`requirements.txt`). No environment variables are needed for local development — see
`.env.example` for the settings used when deploying.

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python ml/data/generate_dataset.py      # synthetic 15k-transaction dataset
python manage.py migrate
python manage.py train_recommender       # trains models, saves real evaluation results
python manage.py seed_demo_data          # farmers, buyers, listings, transactions (idempotent)
python manage.py runserver
```

## Demo accounts
Non-privileged demo logins (shared password `demo12345`, override with the
`CFRS_DEMO_PASSWORD` env var):
- demo_farmer@cfrs.ke — Farmer dashboard
- demo_buyer@cfrs.ke — Buyer dashboard + recommendations

The admin superuser **demo_admin@cfrs.ke** is *not* given the shared password.
`seed_demo_data` sets it from `CFRS_ADMIN_PASSWORD` if provided, otherwise
generates a random one and prints it **once** at the end of the run. Record it
then — it is never stored or published.

## Hybrid recommendation
`Hybrid = 0.40·CF + 0.35·Produce + 0.25·Location` (weights configurable in ml/recommendation/hybrid.py)

Modes: COLLABORATIVE / HYBRID / COLD_START. New buyers with no history fall
back to produce+location preference matching.

## Active-listing validation
The ML engine only scores farmers who currently hold an ACTIVE listing, so no
recommendation ever points to produce that isn't actually available now.

## Evaluation (real, from held-out test set — never hard-coded)
Measured on the deterministic dataset (`python ml/data/generate_dataset.py`
then `python manage.py train_recommender`), held-out test set:

| Model | RMSE | MAE | Precision@5 | Recall@5 | F1@5 |
|---|---|---|---|---|---|
| Popularity baseline | 0.8259 | 0.6518 | 0.3389 | 0.6699 | 0.4501 |
| User-based CF | 0.8616 | 0.6713 | 0.3585 | 0.6666 | 0.4663 |
| Item-based CF | 0.8593 | 0.6852 | 0.3499 | 0.6525 | 0.4555 |

Both collaborative-filtering models outperform the popularity baseline on
**Precision@5 and F1@5**. The baseline, however, records the best **Recall@5**
and the lowest rating error (**RMSE/MAE**). The precision gain is modest
(about +0.01 to +0.02 absolute), consistent with the 94.8% sparsity of the
interaction matrix: personalisation improves ranking precision at this data
density, not recall or rating accuracy. The figures are regenerated on every
training run and shown at /ml-analytics/ (see also
ml/models/evaluation_results.json) — none are hard-coded.

## Key commands
- `python manage.py train_recommender` — retrain + re-evaluate
- `python manage.py seed_demo_data` — populate demo data
- `python manage.py test` — run test suite

## Tests
40 automated tests covering authentication, authorization, listing validation,
the full transaction lifecycle (create -> confirm -> stock decrement -> sold),
and the ML pipeline (preprocessing, hybrid scoring, cold start). Run:
```bash
python manage.py test
```

## Transaction flow (working end-to-end)
Buyer starts a transaction from a listing (with quantity validation) -> farmer
confirms or declines from their dashboard -> on confirm, stock decrements and
the listing is marked SOLD when depleted. Authorization enforced: only the
owning farmer can confirm.

## Known limitations
- Synthetic development data only (clearly labelled; not real market data)
- Price prediction / demand forecasting / fraud detection: architecture
  present as documented extension points, not fabricated (per academic-integrity
  requirement — no results shown without a real trained model)
