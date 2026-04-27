# NeurIPS 2026 Submission Readiness

Official sources:
- `https://neurips.cc/Conferences/2026/CallForPapers`
- `https://neurips.cc/Conferences/2026/MainTrackHandbook`
- `https://docs.openreview.net/getting-started/creating-an-openreview-profile/expediting-profile-activation`

## Deadlines

- Abstract submission deadline: `May 4, 2026 (AoE)`
- Full paper submission deadline, including supplementary material: `May 6, 2026 (AoE)`
- Submission portal opened: `April 15, 2026`

## OpenReview constraints

- NeurIPS states that all authors must have an OpenReview profile when submitting the abstract.
- The handbook also states that all authors need an up-to-date OpenReview profile by the full paper deadline.
- New profiles created without an institutional email can take up to two weeks to activate.
- If the profile is pending, the fastest official fix is to add and confirm an institutional email on the same pending account rather than creating a second account.

## Formatting constraints

- Use the official `neurips_2026.sty`.
- Submission must be a single PDF containing:
  1. paper content
  2. references
  3. appendices
  4. NeurIPS paper checklist
- Main text limit: `9` content pages, including figures and tables.
- References, appendices, and the checklist do not count toward the 9-page content limit.
- Maximum PDF size: `50MB`.
- Optional anonymized code/data can be uploaded separately as one ZIP file up to `100MB`.

## Repo files

- Anonymous submission source: `paper_neurips/neurips2026_submission.tex`
- Anonymous submission PDF target: `paper_neurips/neurips2026_submission.pdf`
- Non-anonymous preprint source: `paper_neurips/neurips2026_preprint.tex`
- Preprint PDF target: `paper_neurips/neurips2026_preprint.pdf`
- Shared manuscript body: `paper_neurips/neurips_body.tex`
- Filled checklist: `paper_neurips/neurips_checklist.tex`
- Official style file: `paper_neurips/neurips_2026.sty`
- Anonymized supplementary code ZIP: `paper_neurips/neurips2026_package/neurips2026_supplementary_code.zip`
- Final submission package folder: `paper_neurips/neurips2026_package/`

## Recommended OpenReview metadata

- Track: Main Track
- Contribution type: `Theory`
- Secondary reasonable fallback: `General`

## Compile commands

From repo root:

```bash
nix-shell -p texlive.combined.scheme-full --run '
  cd paper_neurips &&
  pdflatex -interaction=nonstopmode neurips2026_submission.tex &&
  bibtex neurips2026_submission &&
  pdflatex -interaction=nonstopmode neurips2026_submission.tex &&
  pdflatex -interaction=nonstopmode neurips2026_submission.tex
'
```

```bash
nix-shell -p texlive.combined.scheme-full --run '
  cd paper_neurips &&
  pdflatex -interaction=nonstopmode neurips2026_preprint.tex &&
  bibtex neurips2026_preprint &&
  pdflatex -interaction=nonstopmode neurips2026_preprint.tex &&
  pdflatex -interaction=nonstopmode neurips2026_preprint.tex
'
```
