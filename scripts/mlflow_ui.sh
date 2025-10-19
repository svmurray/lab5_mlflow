#!/usr/bin/env bash
set -euo pipefail
source venv/bin/activate

mlflow server \
  --backend-store-uri ./mlruns \
  --artifacts-destination ./mlartifacts \
  --serve-artifacts \
  --host 0.0.0.0 --port 5000

