#!/usr/bin/env python3
"""Minimal BM25 retrieval server over the wiki-18 corpus.

Adapted from Search-R1's `search_r1/search/retrieval_server.py` (BM25-only path — the
dense/FAISS path is intentionally omitted; see CLAUDE.md's "Why this design" section for why
BM25-only was chosen). Confirmed against the actual downloaded wiki-18 corpus/index — see
CLAUDE.md for the verified corpus schema and the tar-archive gotcha.

Usage:
    python scripts/retrieval_server.py --index_path data/wiki18/bm25 --port 8000

If the Lucene index does not embed raw documents (checked automatically at startup via
`searcher.doc(0).raw()`), pass --corpus_path pointing at the extracted wiki_dump.jsonl as a
fallback lookup source. The startup log states which mode was used — check it before assuming
the corpus download was necessary.

API contract (matches Search-R1's reference server, documented in CLAUDE.md):
    POST /retrieve  {"queries": [str, ...], "topk": int, "return_scores": bool}
      -> {"result": [[{"document": {"title": str, "text": str, "contents": str}, "score": float}, ...], ...]}
    GET /health -> {"status": "ok", "contains_doc": bool}
"""

import argparse
import json

import datasets
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from pyserini.search.lucene import LuceneSearcher


def parse_title_text(contents: str) -> dict:
    """Split a wiki-18 'contents' field into {title, text, contents}.

    Confirmed schema (CLAUDE.md): contents == '"<Title>"\\n<passage text>' — there is no
    separate title field in the corpus. This matches Search-R1's own BM25Retriever parsing
    (`content.split("\\n")[0].strip('"')`) exactly, so titles line up with how alignment against
    HotpotQA's supporting_facts.title was verified.
    """
    lines = contents.split("\n")
    return {
        "title": lines[0].strip('"'),
        "text": "\n".join(lines[1:]),
        "contents": contents,
    }


class QueryRequest(BaseModel):
    queries: list[str]
    topk: int | None = None
    return_scores: bool = False


def _try_raw(searcher: LuceneSearcher, docid) -> str | None:
    """Best-effort fetch of a document's raw JSON string by docid, returning None if the
    index has no document (or no raw content) at that id -- used only to probe whether the
    index embeds raw documents at all, so a missing/empty result here is a normal outcome."""
    doc = searcher.doc(docid)
    if doc is None:
        return None
    return doc.raw()


def _fetch_raw(searcher: LuceneSearcher, docid) -> str:
    """Fetch a document's raw JSON string by docid, raising loudly if missing -- a docid
    returned by a search on this same searcher should always resolve once contains_doc=True."""
    raw = _try_raw(searcher, docid)
    if raw is None:
        raise RuntimeError(f"Lucene index returned no raw content for docid {docid!r}")
    return raw


def build_app(index_path: str, corpus_path: str | None, default_topk: int) -> FastAPI:
    searcher = LuceneSearcher(index_path)
    contains_doc = _try_raw(searcher, 0) is not None
    print(
        f"[retrieval_server] contains_doc={contains_doc} "
        f"({'index embeds raw documents, corpus_path unused' if contains_doc else 'falling back to corpus_path for document lookup'})"
    )

    corpus = None
    if not contains_doc:
        if not corpus_path:
            raise ValueError(
                "Lucene index does not embed raw documents (contains_doc=False) and no "
                "--corpus_path was given — pass the extracted wiki_dump.jsonl path."
            )
        corpus = datasets.load_dataset("json", data_files=corpus_path, split="train")

    app = FastAPI()

    def search_one(query: str, topk: int, return_scores: bool):
        hits = searcher.search(query, topk)
        if not hits:
            return ([], []) if return_scores else []
        scores = [hit.score for hit in hits]
        if contains_doc:
            docs = [
                parse_title_text(json.loads(_fetch_raw(searcher, hit.docid))["contents"])
                for hit in hits
            ]
        else:
            if corpus is None:
                raise RuntimeError("corpus not loaded despite contains_doc=False")
            docs = [parse_title_text(corpus[int(hit.docid)]["contents"]) for hit in hits]
        return (docs, scores) if return_scores else docs

    @app.post("/retrieve")
    def retrieve(request: QueryRequest):
        topk = request.topk or default_topk
        resp = []
        for query in request.queries:
            if request.return_scores:
                docs, scores = search_one(query, topk, True)
                resp.append(
                    [{"document": d, "score": s} for d, s in zip(docs, scores, strict=True)]
                )
            else:
                resp.append(search_one(query, topk, False))
        return {"result": resp}

    @app.get("/health")
    def health():
        return {"status": "ok", "contains_doc": contains_doc}

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal BM25 retrieval server for wiki-18.")
    parser.add_argument(
        "--index_path",
        type=str,
        required=True,
        help="Path to the extracted wiki-18 BM25 Lucene index directory "
        "(the 'bm25/' folder inside PeterJinGo/wiki-18-bm25-index).",
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default=None,
        help="Path to the extracted wiki_dump.jsonl. Only needed if the server logs "
        "contains_doc=False at startup.",
    )
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = build_app(args.index_path, args.corpus_path, args.topk)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
