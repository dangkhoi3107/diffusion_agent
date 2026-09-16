#!/usr/bin/env bash
# Build dist/camo-code.zip for Kaggle (Datasets -> New Dataset -> upload the zip).
#   bash scripts/pack_kaggle.sh
# With the Kaggle CLI configured (~/.kaggle/kaggle.json) you can create/update the dataset directly:
#   KAGGLE_USERNAME=<your-user> bash scripts/pack_kaggle.sh --upload
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="$ROOT/dist/camo-code"
rm -rf "$STAGE" "$ROOT/dist/camo-code.zip"
mkdir -p "$STAGE"
cp -r "$ROOT/camo" "$ROOT/configs" "$ROOT/docs" "$ROOT/kaggle" "$ROOT/tests" "$ROOT/run.sh" "$ROOT/requirements.txt" "$ROOT/README.md" "$STAGE/"
find "$STAGE" -name "__pycache__" -type d -prune -exec rm -rf {} +
(cd "$ROOT/dist" && zip -qr camo-code.zip camo-code)
echo "Built $ROOT/dist/camo-code.zip"

if [[ "${1:-}" == "--upload" ]]; then
  : "${KAGGLE_USERNAME:?set KAGGLE_USERNAME}"
  cat > "$STAGE/dataset-metadata.json" <<JSON
{"title": "camo-code", "id": "${KAGGLE_USERNAME}/camo-code", "licenses": [{"name": "other"}]}
JSON
  if kaggle datasets status "${KAGGLE_USERNAME}/camo-code" >/dev/null 2>&1; then
    kaggle datasets version -p "$STAGE" -r zip -m "update $(date +%F_%T)"
  else
    kaggle datasets create -p "$STAGE" -r zip
  fi
fi
