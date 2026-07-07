#!/usr/bin/env bash
#
# make_arxiv.sh — Build a self-contained source archive of the paper for arXiv.
#
# What it does:
#   1. Compiles the document cleanly (pdflatex + bibtex) to regenerate the
#      bibliography (draft.bbl) and resolve all cross-references.
#   2. Stages the minimal set of files arXiv needs:
#        - the main .tex, with "\pdfoutput=1" injected (the figures are PNG, so
#          arXiv must use pdfLaTeX rather than latex+dvips);
#        - the compiled bibliography draft.bbl (arXiv does not run BibTeX when a
#          .bbl is provided and no .bib is shipped);
#        - only the figures actually referenced by \includegraphics.
#   3. Packs everything into a timestamped .tar.gz.
#   4. Re-compiles the staged archive in an isolated temp directory, WITHOUT
#      refs.bib, to prove the archive is self-contained (this is the check that
#      catches the classic "forgot to include the .bbl" arXiv failure).
#
# Usage:
#   ./make_arxiv.sh [output_dir]           # default output dir: ./arxiv_build
#   INCLUDE_BIB=1 ./make_arxiv.sh          # also bundle refs.bib (optional)
#
# Requirements: TeX Live (latexmk, pdflatex, bibtex), tar, gzip.

set -euo pipefail

# --- Locate the paper directory (this script lives next to draft.tex) --------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MAIN="draft"                                  # main file basename -> draft.tex
STAMP="$(date +%Y%m%d)"
OUT_DIR="${1:-$SCRIPT_DIR/arxiv_build}"
STAGE="$OUT_DIR/pkg"
ARCHIVE="$OUT_DIR/traulls_arxiv_${STAMP}.tar.gz"
INCLUDE_BIB="${INCLUDE_BIB:-0}"               # set to 1 to also ship refs.bib

RED=$'\e[31m'; GRN=$'\e[32m'; YLW=$'\e[33m'; RST=$'\e[0m'
info(){ printf '%s==>%s %s\n' "$GRN" "$RST" "$*"; }
warn(){ printf '%s !!%s %s\n' "$YLW" "$RST" "$*"; }
die(){  printf '%s xx%s %s\n' "$RED" "$RST" "$*" >&2; exit 1; }

command -v latexmk  >/dev/null || die "latexmk not found (install TeX Live)."
command -v pdflatex >/dev/null || die "pdflatex not found."
[[ -f "$MAIN.tex" ]] || die "$MAIN.tex not found in $SCRIPT_DIR."

# --- 1. Clean compile so the .bbl and cross-references are up to date ---------
info "Compiling $MAIN.tex (pdflatex + bibtex)…"
latexmk -pdf -bibtex -interaction=nonstopmode -halt-on-error "$MAIN.tex" >/dev/null \
    || die "Compilation failed. Run 'latexmk -pdf $MAIN.tex' to inspect the errors."
[[ -s "$MAIN.bbl" ]] || die "No non-empty $MAIN.bbl was produced — bibliography missing."

if grep -qE 'Citation .* undefined|Reference .* undefined|There were undefined references' "$MAIN.log"; then
    warn "The log reports undefined references/citations — check $MAIN.log before submitting."
fi

# --- 2. Stage the minimal file set -------------------------------------------
info "Staging files in $STAGE …"
mkdir -p "$OUT_DIR"
rm -rf "$STAGE"
mkdir -p "$STAGE"

# 2a. Main .tex, forcing pdfLaTeX via \pdfoutput=1 (kept out of the working copy).
if grep -qE '^[[:space:]]*\\pdfoutput[[:space:]]*=' "$MAIN.tex"; then
    cp "$MAIN.tex" "$STAGE/$MAIN.tex"
else
    awk '1; /\\documentclass/ && !ins { print "\\pdfoutput=1"; ins=1 }' \
        "$MAIN.tex" > "$STAGE/$MAIN.tex"
fi

# 2b. Compiled bibliography (arXiv uses this; no .bib shipped unless requested).
cp "$MAIN.bbl" "$STAGE/$MAIN.bbl"
if [[ "$INCLUDE_BIB" == "1" && -f refs.bib ]]; then
    cp refs.bib "$STAGE/"; info "bundled refs.bib (INCLUDE_BIB=1)."
fi

# 2c. Only the figures actually referenced, resolving the file extension.
missing=0
while IFS= read -r ref; do
    [[ -z "$ref" ]] && continue
    found=""
    for cand in "$ref" "$ref".pdf "$ref".png "$ref".jpg "$ref".jpeg "$ref".eps; do
        [[ -f "$cand" ]] && { found="$cand"; break; }
    done
    [[ -z "$found" ]] && { warn "figure not found for reference: $ref"; missing=1; continue; }
    mkdir -p "$STAGE/$(dirname "$found")"
    cp "$found" "$STAGE/$found"
done < <(grep -oE '\\includegraphics(\[[^]]*\])?\{[^}]*\}' "$MAIN.tex" \
             | sed -E 's/.*\{([^}]*)\}/\1/' | sort -u)
[[ "$missing" -eq 0 ]] || die "Some figures could not be located (see warnings above)."

# 2d. Bundle any custom .sty/.cls that live next to the source (none expected).
shopt -s nullglob
for sty in ./*.sty ./*.cls; do
    cp "$sty" "$STAGE/"; info "bundled local class/style file: $sty"
done
shopt -u nullglob

# --- 3. Pack the archive -----------------------------------------------------
info "Creating archive…"
tar -czf "$ARCHIVE" -C "$STAGE" .

# --- 4. Isolated rebuild test (mimic arXiv: only the shipped files) ----------
info "Test-compiling the staged archive in isolation (no refs.bib)…"
TESTDIR="$(mktemp -d)"
trap 'rm -rf "$TESTDIR"' EXIT
tar -xzf "$ARCHIVE" -C "$TESTDIR"
if ! ( cd "$TESTDIR"
       for pass in 1 2 3; do
           pdflatex -interaction=nonstopmode -halt-on-error "$MAIN.tex" >"pass$pass.log" 2>&1
       done ); then
    die "Isolated compile FAILED — archive is not self-contained. Logs in $TESTDIR."
fi
[[ -s "$TESTDIR/$MAIN.pdf" ]] || die "Isolated compile produced no PDF."
pages="$(pdfinfo "$TESTDIR/$MAIN.pdf" 2>/dev/null | awk '/^Pages/{print $2}')"

# --- 5. Summary --------------------------------------------------------------
info "Archive ready: $ARCHIVE"
printf '    size    : %s\n' "$(du -h "$ARCHIVE" | cut -f1)"
printf '    pages   : %s (isolated pdfLaTeX build OK)\n' "${pages:-unknown}"
echo   "    contents:"
tar -tzf "$ARCHIVE" | sed 's/^/      /'
cat <<EOF

Next steps:
  * Upload $ARCHIVE at https://arxiv.org/submit
  * arXiv will run pdfLaTeX (forced by \\pdfoutput=1) and use the bundled .bbl.
  * refs.bib is omitted by design; re-run this script after any change to the
    text or to refs.bib so the shipped .bbl stays in sync.
EOF
