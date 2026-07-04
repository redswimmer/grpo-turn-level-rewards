#!/usr/bin/env bash
# Downloads the wiki-18 BM25 index, checks whether it embeds raw documents (in which case the
# separate 5.12GB corpus download can be skipped entirely), and only downloads+extracts the
# corpus if actually needed. See CLAUDE.md's "Why this design (retrieval backend choice)" for
# background on why these exact artifacts were chosen.
#
# Prerequisite: JDK 21 + this project's `pyserini`/`huggingface_hub` deps already installed
# (this script does not install them — see docs/phase-1-retrieval-infra.md task list).
#
# Usage: bash scripts/setup_retrieval.sh [data_dir]   (default: data/wiki18)
set -euo pipefail

DATA_DIR="${1:-data/wiki18}"
mkdir -p "$DATA_DIR"

echo "== Downloading wiki-18 BM25 index (~2.3GB) =="
uv run --with huggingface_hub hf download PeterJinGo/wiki-18-bm25-index \
    --repo-type dataset --local-dir "$DATA_DIR/bm25-repo"

INDEX_PATH="$DATA_DIR/bm25-repo/bm25"
if [ ! -d "$INDEX_PATH" ]; then
    echo "ERROR: expected Lucene index directory not found at $INDEX_PATH" >&2
    exit 1
fi

echo "== Checking whether the index embeds raw documents (contains_doc) =="
CONTAINS_DOC=$(uv run python3 -c "
from pyserini.search.lucene import LuceneSearcher
s = LuceneSearcher('$INDEX_PATH')
print('yes' if s.doc(0).raw() is not None else 'no')
")

if [ "$CONTAINS_DOC" = "yes" ]; then
    echo "Index embeds raw documents -- the 5.12GB corpus download is NOT needed."
    echo ""
    echo "Launch with:"
    echo "  uv run python scripts/retrieval_server.py --index_path $INDEX_PATH --port 8000"
else
    echo "Index does not embed raw documents -- downloading the corpus as a fallback lookup source."
    echo "== Downloading wiki-18 corpus (~5.12GB) =="
    uv run --with huggingface_hub hf download PeterJinGo/wiki-18-corpus \
        --repo-type dataset --local-dir "$DATA_DIR/corpus-repo"

    echo "== Extracting tar-wrapped corpus (see CLAUDE.md's tar-archive gotcha) =="
    tar -xzf "$DATA_DIR/corpus-repo/wiki-18.jsonl.gz" -C "$DATA_DIR"
    CORPUS_PATH=$(find "$DATA_DIR" -name wiki_dump.jsonl | head -1)
    echo "Extracted corpus at: $CORPUS_PATH"
    echo ""
    echo "Launch with:"
    echo "  uv run python scripts/retrieval_server.py --index_path $INDEX_PATH --corpus_path $CORPUS_PATH --port 8000"
fi

echo ""
echo "== After launching the server in another terminal/background process, run: =="
echo "  uv run python scripts/verify_retrieval.py"
