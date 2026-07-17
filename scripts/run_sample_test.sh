#!/usr/bin/env bash
set -e

# Génère le PDF de test si reportlab est disponible.
if python3 - <<'PY' >/dev/null 2>&1
import reportlab
PY
then
  python3 scripts/generate_sample_pdf.py
else
  echo "reportlab n'est pas installé, installe-le avec : pip install reportlab" >&2
  exit 1
fi

python3 graph.py --cv samples/example_cv.pdf --jd samples/example_offer.txt
