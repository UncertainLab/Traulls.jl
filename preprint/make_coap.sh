#!/usr/bin/env bash
#
# make_coap.sh — Build the source archive of the paper for submission to Computational
# Optimization and Applications (Springer Nature, Editorial Manager).
#
# What it does:
#   1. Stages the files of the submission:
#        - the main .tex, without the review marks: the definition of the macro \rev,
#          with the comment lines just above it, is removed, and every \rev{...} is
#          replaced by its content (the working copy keeps its marks);
#        - the .bib files of \bibliography, and the class and style files that live next
#          to the source (sn-jnl.cls, sn-mathphys-num.bst);
#        - only the figures actually referenced by \includegraphics.
#   2. Compiles the staged copy (pdflatex + bibtex), and keeps the compiled bibliography
#      ($MAIN.bbl) and the PDF next to the sources.
#   3. Packs everything into a timestamped .zip.
#   4. Re-compiles the archive in an isolated temp directory, WITHOUT the .bib files and
#      the PDF, to prove that it is self-contained.
#
# Usage:
#   ./make_coap.sh [output_dir]            # default output dir: ./coap_build
#   MAIN=other PAPER_DIR=path ./make_coap.sh
#                                          # main file basename and directory
#                                          # (defaults: sn-preprint-theorie, ./coap)
#
# Requirements: TeX Live (latexmk, pdflatex, bibtex), perl, poppler (pdfinfo), zip, unzip.

set -euo pipefail
# Byte-wise text processing, whatever the encoding of the source (e.g. Windows-1252)
export LC_ALL=C

RED=$'\e[31m'; GRN=$'\e[32m'; YLW=$'\e[33m'; RST=$'\e[0m'
info(){ printf '%s==>%s %s\n' "$GRN" "$RST" "$*"; }
warn(){ printf '%s !!%s %s\n' "$YLW" "$RST" "$*"; }
die(){  printf '%s xx%s %s\n' "$RED" "$RST" "$*" >&2; exit 1; }

# --- Locate the directories (output resolved before leaving the caller's directory) ---
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-$SCRIPT_DIR/coap_build}"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd -- "$OUT_DIR" && pwd)"
PAPER_DIR="$(cd -- "${PAPER_DIR:-$SCRIPT_DIR/coap}" && pwd)"
cd "$PAPER_DIR"

MAIN="${MAIN:-sn-preprint-theorie}"           # main file basename -> sn-preprint-theorie.tex
STAMP="$(date +%Y%m%d)"
STAGE="$OUT_DIR/pkg"
ARCHIVE="$OUT_DIR/traulls_coap_${STAMP}.zip"

for cmd in latexmk pdflatex bibtex perl zip unzip; do
    command -v "$cmd" >/dev/null || die "$cmd not found."
done
[[ -f "$MAIN.tex" ]] || die "$MAIN.tex not found in $PAPER_DIR."

# --- 1. Stage the files of the submission ------------------------------------
info "Staging files in $STAGE …"
rm -rf "$STAGE"
mkdir -p "$STAGE"

# 1a. Main .tex without the review marks. The first substitution removes the definition
# of \rev and the comment lines just above it, the second one replaces every \rev{...}
# by its content, the braces being matched recursively.
nmarks="$(grep -o '\\rev{' "$MAIN.tex" | wc -l)"
perl -0777 -pe '
    s/(?:^%[^\n]*\n)*^[ \t]*\\(?:re)?newcommand\{?\\rev\}?[^\n]*\n//mg;
    s/\\rev(\{((?:[^{}\\]++|\\.|(?1))*)\})/$2/gs;
' "$MAIN.tex" > "$STAGE/$MAIN.tex"
if grep -q '\\rev\b' "$STAGE/$MAIN.tex"; then
    die "Review marks left in $STAGE/$MAIN.tex: check the uses of \\rev in $MAIN.tex."
fi
info "removed $nmarks review mark(s) \\rev{...}."

# 1b. Bibliography databases (the compiled .bbl is added after the compilation).
for bib in $(grep -vE '^[[:space:]]*%' "$MAIN.tex" \
                 | grep -oE '\\bibliography\{[^}]*\}' | sed -E 's/.*\{([^}]*)\}/\1/' | tr ',' ' '); do
    [[ -f "$bib.bib" ]] || die "bibliography database $bib.bib not found."
    cp "$bib.bib" "$STAGE/"
done

# 1c. Class and style files that live next to the source.
shopt -s nullglob
for sty in ./*.cls ./*.bst ./*.sty; do
    cp "$sty" "$STAGE/"
done
shopt -u nullglob

# 1d. Only the figures actually referenced, resolving the file extension.
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

# --- 2. Compile the staged copy -----------------------------------------------
info "Compiling the staged $MAIN.tex (pdflatex + bibtex)…"
( cd "$STAGE" && latexmk -pdf -bibtex -interaction=nonstopmode -halt-on-error "$MAIN.tex" ) \
    >"$OUT_DIR/build.log" 2>&1 \
    || die "Compilation failed. See $OUT_DIR/build.log and $STAGE/$MAIN.log."
[[ -s "$STAGE/$MAIN.bbl" ]] || die "No non-empty $MAIN.bbl was produced — bibliography missing."

# The log is kept in the output directory, the other auxiliary files are removed
mv "$STAGE/$MAIN.log" "$OUT_DIR/$MAIN.log"
if grep -qE 'Citation .* undefined|Reference .* undefined|There were undefined references' \
        "$OUT_DIR/$MAIN.log"; then
    warn "The log reports undefined references/citations — check $OUT_DIR/$MAIN.log before submitting."
fi
( cd "$STAGE" && rm -f -- "$MAIN".aux "$MAIN".out "$MAIN".blg "$MAIN".fls \
                          "$MAIN".fdb_latexmk "$MAIN".synctex.gz )

# --- 3. Pack the archive -------------------------------------------------------
info "Creating archive…"
rm -f "$ARCHIVE"
( cd "$STAGE" && zip -q -X -r "$ARCHIVE" -- * )

# --- 4. Isolated rebuild test (only the shipped files, without the .bib) ------
info "Test-compiling the archive in isolation (no .bib file)…"
TESTDIR="$(mktemp -d)"
trap 'rm -rf "$TESTDIR"' EXIT
unzip -q "$ARCHIVE" -d "$TESTDIR"
rm -f -- "$TESTDIR"/*.bib "$TESTDIR/$MAIN.pdf"
# The temp directory is removed on exit, so the logs of a failure are kept in OUT_DIR
if ! ( cd "$TESTDIR"
       for pass in 1 2 3; do
           pdflatex -interaction=nonstopmode -halt-on-error "$MAIN.tex" >"pass$pass.log" 2>&1
       done ); then
    cp "$TESTDIR"/pass*.log "$OUT_DIR/"
    die "Isolated compile FAILED — the archive is not self-contained. Logs in $OUT_DIR."
fi
[[ -s "$TESTDIR/$MAIN.pdf" ]] || die "Isolated compile produced no PDF."
if grep -qE 'Citation .* undefined|Reference .* undefined' "$TESTDIR/$MAIN.log"; then
    warn "Undefined references/citations in the isolated build."
fi
pages="$(pdfinfo "$TESTDIR/$MAIN.pdf" 2>/dev/null | awk '/^Pages/{print $2}')"

# --- 5. Summary -----------------------------------------------------------------
info "Archive ready: $ARCHIVE"
printf '    size    : %s\n' "$(du -h "$ARCHIVE" | cut -f1)"
printf '    pages   : %s (isolated pdfLaTeX build OK)\n' "${pages:-unknown}"
echo   "    contents:"
unzip -Z1 "$ARCHIVE" | sed 's/^/      /'
cat <<EOF

Next steps:
  * Submit through Editorial Manager, with the archive or the files of $STAGE.
  * $STAGE/$MAIN.pdf is the compiled manuscript, without the review marks.
  * Re-run this script after any change to the text, the figures or the bibliography.
EOF
