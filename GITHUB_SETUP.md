# GitHub Setup & Submission Guide — CFRS

**Project:** Collaborative Filtering Recommendation System for Matching Farmers and Buyers in Agricultural Markets
**Student:** Samson Aira (166456) · ICS 4B · Supervisor: Mr. Daniel Machanje

This repository is already a Git repository with a full commit history and a
branch structure. This guide covers the remaining steps: pushing to GitHub,
creating the milestones and issues the assignment asks for, and submitting the
form.

---

## 1. What is already done in this repo

- **Git initialised** on branch `main`, with `develop` created for integration.
- **18 meaningful commits**, one per module/phase (auth → marketplace →
  ML modules → recommendation API → transactions → UI → tests → docs).
  Run `git log --oneline` to see them.
- **`.gitignore`** excludes `__pycache__/`, `venv/`, `db.sqlite3`, and the
  trained `ml/models/*.pkl` artifacts (regenerated with one command) and the
  regenerable `ml/data/transactions.csv`. The **real FAO Kenya CSVs are
  tracked** because they are source data, not build output.

> **Honesty note for the panel:** the history was created by importing an
> existing project into Git in logically grouped commits *made today*. The
> commit dates are today's — nothing is backdated to fake weeks of work. This
> is the standard "bring an existing project under version control" workflow.
> From here on, commit as you work so the history stays genuine.

---

## 2. Create the GitHub repository and push

### Option A — using the `gh` CLI (fastest)
```bash
cd cfrs
gh auth login                      # once, follow the prompts
gh repo create <org-or-username>/cfrs --private --source=. --remote=origin
git push -u origin main
git push -u origin develop
```

### Option B — using the GitHub website
1. On GitHub, click **New repository**, name it `cfrs`, keep it **Private**,
   and do **not** add a README/.gitignore (this repo already has them).
2. Copy the remote URL, then:
```bash
cd cfrs
git remote add origin https://github.com/<org-or-username>/cfrs.git
git push -u origin main
git push -u origin develop
```

Confirm both branches show on GitHub before submitting.

---

## 3. Branch structure

| Branch | Purpose |
|--------|---------|
| `main` | Stable, working code (what the panel reviews). |
| `develop` | Integration branch for ongoing work. |
| `feature/*` | One branch per new piece of work, going forward. |

**Workflow from now on** (this is what keeps history genuine):
```bash
git checkout develop
git checkout -b feature/farmer-profile-edit   # example
# ... make changes, then ...
git add -A && git commit -m "feat(users): farmer profile editing"
git push -u origin feature/farmer-profile-edit
# open a Pull Request into develop on GitHub, then merge
```

---

## 4. Milestones to create on GitHub

Create these under **Issues → Milestones → New milestone**. They map to the
project phases.

1. **M1 — Foundation & Accounts** — Django scaffold, custom Account model,
   Farmer/Buyer/Administrator roles, auth. *(done)*
2. **M2 — Marketplace Core** — produce listings, transactions with validation,
   active-listing rule. *(done)*
3. **M3 — ML Recommendation Engine** — collaborative filtering, hybrid
   recommender, cold-start, evaluation metrics. *(done)*
4. **M4 — Supporting ML Modules** — price prediction & demand forecasting on
   FAO Kenya data, fraud detection. *(done)*
5. **M5 — Integration & UI** — recommendation API, dashboards, ML analytics
   screens. *(done)*
6. **M6 — Testing & Documentation** — automated tests, Chapter 4/5 docs.
   *(in progress)*

---

## 5. Issues / tasks to create

Create these under **Issues → New issue** and assign each to a milestone. The
"done" ones document completed work; the "open" ones are genuine remaining
tasks. If you have the `gh` CLI, the block at the bottom creates them all.

**Milestone M6 (open — real remaining work):**
- Add farmer profile-edit UI form.
- Add listing-edit / deactivate UI for farmers.
- Write Chapter 5 (implementation & testing) documentation.
- Fix static-files serving in production settings (`STATICFILES_DIRS`).
- Add user-acceptance test script and record results.

**Closed on creation (documents completed work — close them right after):**
- Custom Account model with three roles — *M1*
- Registration / login / role-based dashboards — *M1*
- ProduceListing + Transaction models with active-listing validation — *M2*
- Transaction service (create/confirm/cancel, stock handling) — *M2*
- User- & item-based collaborative filtering with precision/recall@k — *M3*
- Hybrid recommender (0.40 CF + 0.35 produce + 0.25 location) + cold-start — *M3*
- Price prediction on real FAO Kenya prices — *M4*
- Demand forecasting on real FAO production data — *M4*
- Isolation-forest fraud detection — *M4*
- Recommendation DRF endpoint with role authorization — *M5*
- Buyer/Farmer/Admin dashboards + ML analytics screens — *M5*
- Automated test suite (accounts, marketplace, recommendations) — *M6*

### Create everything with `gh` (optional, run after `gh repo create`)
```bash
# milestones
for m in \
  "M1 — Foundation & Accounts" \
  "M2 — Marketplace Core" \
  "M3 — ML Recommendation Engine" \
  "M4 — Supporting ML Modules" \
  "M5 — Integration & UI" \
  "M6 — Testing & Documentation"; do
  gh api repos/:owner/:repo/milestones -f title="$m" >/dev/null
done

# a few open task issues (add the rest the same way)
gh issue create -t "Add farmer profile-edit UI form" -b "Farmers need a screen to edit their profile." -m "M6 — Testing & Documentation"
gh issue create -t "Add listing-edit / deactivate UI for farmers" -b "Farmers should edit or deactivate their own listings from the dashboard." -m "M6 — Testing & Documentation"
gh issue create -t "Write Chapter 5 documentation" -b "Implementation and testing chapter." -m "M6 — Testing & Documentation"
```

---

## 6. Submit the form

1. Confirm on GitHub: repo is created, `main` and `develop` are pushed,
   milestones and issues exist, and `git log` shows the commit history.
2. Copy the repository URL (`https://github.com/<org-or-username>/cfrs`).
3. Paste it into the submission form the lecturer sent.
4. If the form asks for it, add the supervisor / lecturer as a collaborator:
   **Settings → Collaborators → Add people**.

---

## 7. Regenerating the ignored artifacts (for anyone who clones)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python ml/data/generate_dataset.py          # rebuilds ml/data/transactions.csv
python manage.py migrate
python manage.py train_recommender          # rebuilds ml/models/*.pkl
python manage.py seed_demo_data
python manage.py runserver
```
