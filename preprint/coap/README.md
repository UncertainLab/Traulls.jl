# Revised manuscript for Computational Optimization and Applications

- `sn-preprint-theorie.tex`: revised manuscript, in the Springer Nature template. The file is
  encoded in Windows-1252 with CRLF line endings: do not convert it to UTF-8. The red boxes and
  the bracketed red remarks are revision notes, and the green paragraph of Section 2.4 is an
  addition that remains to be supported by numerical tests.
- `sn-preprint-theorie.pdf`: compiled manuscript.
- `sn-preprint-theorie-diff.pdf`: changes with respect to `sn-preprint-ojmo.tex`, the version
  submitted to OJMO, produced by
  `latexdiff --encoding=cp1252 --math-markup=whole sn-preprint-ojmo.tex sn-preprint-theorie.tex`.
- `refs-theorie.bib`: bibliography of the revised manuscript.

The numerical results of Section 4 still come from commit `c7bc71b`, as stated in the red note
at the start of that section, and have to be regenerated with `benchmark/run_experiments.jl`.

Build with `latexmk -pdf sn-preprint-theorie.tex`.
