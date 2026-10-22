#!/usr/bin/env bash
#
# make_arxiv.sh — Build a self-contained source archive of the paper for arXiv.
#
# What it does:
#   1. Compiles the document cleanly (pdflatex + bibtex) to regenerate the
#      bibliography ($MAIN.bbl) and resolve all cross-references.
#   2. Stages the minimal set of files arXiv needs:
#        - the main .tex, with "\pdfoutput=1" added on its first line, where
#          arXiv looks for it (the figures are PDF or PNG, so arXiv must use
#          pdfLaTeX rather than latex+dvips);
#        - the compiled bibliography $MAIN.bbl (arXiv does not run BibTeX when a
#          .bbl is provided and no .bib is shipped);
#        - only the figures actually referenced by \includegraphics.
#   3. Packs everything into a timestamped .tar.gz.
#   4. Re-compiles the staged archive in an isolated temp directory, WITHOUT
#      the .bib files, to prove the archive is self-contained (this is the check
#      that catches the classic "forgot to include the .bbl" arXiv failure).
#
# Usage:
#   ./make_arxiv.sh [output_dir]           # default output dir: ./arxiv_build
#   INCLUDE_BIB=1 ./make_arxiv.sh          # also bundle the .bib files (optional)
#   MAIN=sn-preprint-theorie PAPER_DIR=../preprint/coap ./make_arxiv.sh
#                                          # another paper: main file basename and
#                                          # directory (defaults: main, this one)
#
# Requirements: TeX Live (latexmk, pdflatex, bibtex), poppler (pdfinfo), tar, gzip.

set -euo pipefail
# Byte-wise text processing, whatever the encoding of the source (e.g. Windows-1252)
export LC_ALL=C

# --- Locate the paper directory (by default, the one of this script) ---------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="$(cd -- "${PAPER_DIR:-$SCRIPT_DIR}" && pwd)"
cd "$PAPER_DIR"

MAIN="${MAIN:-main}"                          # main file basename -> main.tex
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
[[ -f "$MAIN.tex" ]] || die "$MAIN.tex not found in $PAPER_DIR."

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
# arXiv only detects it in the first lines, and the class may load the graphics
# driver, so it goes before everything, with the line ending of the file.
if grep -qE '^[[:space:]]*\\pdfoutput[[:space:]]*=' "$MAIN.tex"; then
    cp "$MAIN.tex" "$STAGE/$MAIN.tex"
else
    eol=$'\n'; head -n 1 "$MAIN.tex" | grep -q $'\r$' && eol=$'\r\n'
    { printf '\\pdfoutput=1%s' "$eol"; cat "$MAIN.tex"; } > "$STAGE/$MAIN.tex"
fi

# 2b. Compiled bibliography (arXiv uses this; no .bib shipped unless requested).
cp "$MAIN.bbl" "$STAGE/$MAIN.bbl"
if [[ "$INCLUDE_BIB" == "1" ]]; then
    for bib in $(grep -vE '^[[:space:]]*%' "$MAIN.tex" \
                     | grep -oE '\\bibliography\{[^}]*\}' | sed -E 's/.*\{([^}]*)\}/\1/' | tr ',' ' '); do
        [[ -f "$bib.bib" ]] && { cp "$bib.bib" "$STAGE/"; info "bundled $bib.bib (INCLUDE_BIB=1)."; }
    done
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
done < <(grep -vE '^[[:space:]]*%' "$MAIN.tex" \
             | grep -oE '\\includegraphics(\[[^]]*\])?\{[^}]*\}' \
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
info "Test-compiling the staged archive in isolation (no .bib files)…"
TESTDIR="$(mktemp -d)"
trap 'rm -rf "$TESTDIR"' EXIT
tar -xzf "$ARCHIVE" -C "$TESTDIR"
# The temp directory is removed on exit, so the logs of a failure are kept in OUT_DIR
if ! ( cd "$TESTDIR"
       for pass in 1 2 3; do
           pdflatex -interaction=nonstopmode -halt-on-error "$MAIN.tex" >"pass$pass.log" 2>&1
       done ); then
    cp "$TESTDIR"/pass*.log "$OUT_DIR/"
    die "Isolated compile FAILED — archive is not self-contained. Logs in $OUT_DIR."
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
  * The .bib files are omitted by design; re-run this script after any change to
    the text or to the bibliography so the shipped .bbl stays in sync.
EOF
