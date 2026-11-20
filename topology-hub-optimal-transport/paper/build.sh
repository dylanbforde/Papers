#!/bin/sh
set -eu
cd "$(dirname "$0")"

if command -v tectonic >/dev/null 2>&1; then
  tectonic topology_hub_ot.tex
  exit 0
fi

command -v pdflatex >/dev/null 2>&1 || {
  echo "tectonic or pdfLaTeX is required." >&2
  exit 1
}
command -v bibtex >/dev/null 2>&1 || {
  echo "BibTeX is required when building with pdfLaTeX." >&2
  exit 1
}

pdflatex -interaction=nonstopmode -halt-on-error topology_hub_ot.tex
bibtex topology_hub_ot
pdflatex -interaction=nonstopmode -halt-on-error topology_hub_ot.tex
pdflatex -interaction=nonstopmode -halt-on-error topology_hub_ot.tex

