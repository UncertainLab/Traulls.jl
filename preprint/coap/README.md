# Revised manuscript for Computational Optimization and Applications

- `sn-preprint-theorie.tex`: revised manuscript, in the Springer Nature template. The file is
  encoded in Windows-1252 with CRLF line endings: do not convert it to UTF-8.
- `sn-preprint-theorie.pdf`: compiled manuscript.
- `sn-preprint-theorie-diff.pdf`: changes with respect to `sn-preprint-ojmo.tex`, the version
  submitted to OJMO, produced by
  `latexdiff --encoding=cp1252 --math-markup=whole --append-textcmd=abstract sn-preprint-ojmo.tex sn-preprint-theorie.tex`
  (the last option marks the changes of the abstract). `CHANGEMENTS.md` lists the main changes
  section by section, with the pages of this PDF.
- `sn-preprint-theorie-diff-592160d.pdf`: changes with respect to the version of commit `592160d`,
  whose content is that of `../main.tex` before its synchronization with this file, produced by the
  same `latexdiff` command applied to `git show 592160d:preprint/coap/sn-preprint-theorie.tex` and
  `sn-preprint-theorie.tex`. latexdiff leaves an empty line inside a deleted equation carrying a
  `\label` (the old `eq:refixing`), which was removed from its output before compiling.
- `refs-theorie.bib`: bibliography of the revised manuscript.

The code, results and scripts of the paper are those of the tag `coap-2026` of the repository,
whose content is also on `main`. The scripts are described in `docs/src/benchmarks.md`, and
`benchmark/paper_numbers.jl` prints the aggregated numbers of Sections 2.4, 4.2 and 4.3.
The numerical results of Section 4 come from the timed campaign of 2026-10-21, run with the
solver code of commit `3f0744f` on `main` (the campaign was launched at `3b97a8c`,
which only adds results), whose files are in `benchmark/results`.
The figures `Fig1.pdf` to `Fig15.pdf` are those of `preprint/figures` on that branch, produced by
`benchmark/make_figures.jl` and `benchmark/residual_cost_analysis.jl`.
The results of Section 4.3 (Table 2) are in `benchmark/results/linear_constraints.csv`,
produced by `benchmark/linear_constraints/run_linear_constraints.jl` at commit `3f0744f`.
The comparison of the structured and standard secant equations of Section 2.4 is in
`benchmark/results/secant_comparison.csv` and `benchmark/results/secant_comparison_mu.csv`,
produced by `benchmark/secant_comparison/run_secant_comparison.jl` with the solver code of `3f0744f`.

Build with `latexmk -pdf sn-preprint-theorie.tex`.
