#!/usr/bin/env bash
# Render build script. Runs on every deploy.
set -o errexit
pip install -r requirements.txt
# Regenerate the (deterministic) dataset and train the models so no large .pkl
# files live in git. A live prediction never trains — only this build step does.
python ml/data/generate_dataset.py
python manage.py train_recommender
python manage.py collectstatic --no-input
python manage.py migrate
# Seed demo data (idempotent). Set CFRS_ADMIN_PASSWORD in the host env for a
# stable admin password; otherwise one is generated and printed in the build log.
python manage.py seed_demo_data
