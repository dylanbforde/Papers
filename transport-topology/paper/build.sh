#!/bin/sh
set -eu
cd "$(dirname "$0")"
command -v pdflatex >/dev/null 2>&1 || {
  echo "pdfLaTeX is required." >&2
  exit 1
}
command -v bibtex >/dev/null 2>&1 || {
  echo "BibTeX is required by this script; the included main.bbl also permits direct pdfLaTeX builds." >&2
  exit 1
}
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
