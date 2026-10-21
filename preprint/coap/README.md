# Revised manuscript for Computational Optimization and Applications

- `sn-preprint-theorie.tex`: revised manuscript, in the Springer Nature template. The file is
  encoded in Windows-1252 with CRLF line endings: do not convert it to UTF-8. The green paragraph of Section 2.4 is an addition that remains to be supported by numerical tests.
- `sn-preprint-theorie.pdf`: compiled manuscript.
- `sn-preprint-theorie-diff.pdf`: changes with respect to `sn-preprint-ojmo.tex`, the version
  submitted to OJMO, produced by
  `latexdiff --encoding=cp1252 --math-markup=whole sn-preprint-ojmo.tex sn-preprint-theorie.tex`.
- `refs-theorie.bib`: bibliography of the revised manuscript.

The numerical results of Section 4 come from the timed campaign of 2026-10-21, run with the
solver code of commit `3f0744f` on branch `align-paper` (the campaign was launched at `3b97a8c`,
which only adds results), whose files are in `benchmark/results`.
The figures `Fig1.pdf` to `Fig15.pdf` are those of `preprint/figures` on that branch, produced by
`benchmark/make_figures.jl` and `benchmark/residual_cost_analysis.jl`.
The results of Section 4.3 (Table 2) are in `benchmark/results/linear_constraints.csv`,
produced by `benchmark/linear_constraints/run_linear_constraints.jl` at commit `3f0744f`.

Build with `latexmk -pdf sn-preprint-theorie.tex`.
