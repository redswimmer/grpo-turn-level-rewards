# Phase 1: Retrieval infrastructure

## Goal

Stand up a working local retrieval server backed by the real wiki-18 BM25 index, reachable over
HTTP, so every later phase can treat "search" as a simple `POST /retrieve` call and never needs
to import Pyserini directly.

**This phase has been pre-researched and pre-scaffolded.** The scripts you need already exist in
`scripts/`; your job is to run them, install the one system-level dependency they can't install
for you, and verify the result — not to design the retrieval server from scratch.

## Read first

`CLAUDE.md` in the repo root — especially "Why this design (retrieval backend choice)" for the
corpus/index facts, the tar-archive gotcha, and the corpus record schema. This doc covers this
phase's concrete tasks and does not repeat that context.

## What already exists for you

- `scripts/retrieval_server.py` — a complete, ready-to-run FastAPI BM25 retrieval server,
  adapted directly from Search-R1's own `search_r1/search/retrieval_server.py` (confirmed via
  its actual source) and matched to the confirmed wiki-18 corpus schema
  (`contents = '"<Title>"\n<passage text>'`, no separate title field). Implements the
  `POST /retrieve` / `GET /health` contract documented in CLAUDE.md.
- `scripts/setup_retrieval.sh` — downloads `PeterJinGo/wiki-18-bm25-index` (~2.3GB) via the `hf`
  CLI, then **checks whether the Lucene index embeds raw documents** (`contains_doc`) before
  deciding whether the separate 5.12GB `PeterJinGo/wiki-18-corpus` download is even necessary —
  most Pyserini indexes built for this purpose do embed raw docs, so this step can likely save
  you a 5GB download and the tar-extraction step entirely. It only downloads+extracts the corpus
  if the check comes back negative.
- `scripts/verify_retrieval.py` — **the Phase 1 exit-criteria check** (see "Verification loop"
  below). Run this last; it tells you definitively whether you're done.

## Prerequisites (entry state)

- Nothing downloaded yet, no JDK installed, no retrieval server running.
- Confirmed facts you can rely on without re-deriving (all verified directly, see CLAUDE.md):
  - `pyserini==2.3.0` requires **Java 21** and resolves cleanly via `uv pip install` on this
    project's Python 3.13 (no separate Python env needed).
  - This machine is Ubuntu 26.04 — `openjdk-21-jdk` is available via `apt`.
  - `PeterJinGo/wiki-18-bm25-index`'s Hub repo contains exactly one directory, `bm25/` (the
    actual Lucene index directory to point `LuceneSearcher`/`--index_path` at) plus a
    `.gitattributes` file.
  - `PeterJinGo/wiki-18-corpus`'s `wiki-18.jsonl.gz` is a **tar archive**, not a plain gzipped
    jsonl (`tar -xzf` it, don't just `gunzip`) — `setup_retrieval.sh` already handles this
    correctly if it turns out to be needed.

## Tasks

1. [x] Install the JDK: `sudo apt install openjdk-21-jdk` (confirmed exact version needed —
       not a guess). Verify with `java -version` reporting `21`.
2. [x] ~~Add dependencies to `pyproject.toml`~~ — already done: `pyserini`, `fastapi`,
       `uvicorn`, `requests`, `huggingface-hub`, `datasets`, `pydantic` are all in
       `pyproject.toml`/`uv.lock` and pass `ty`/`ruff` clean. Run `uv sync` to make sure your
       `.venv` actually has them installed (adding to `pyproject.toml` alone doesn't install
       anything into a fresh checkout).
3. [x] Run `bash scripts/setup_retrieval.sh` (defaults to downloading into `data/wiki18/`,
       already `.gitignore`d). Read its output carefully — it will tell you whether the corpus
       download happened and print the exact `retrieval_server.py` launch command to use.
4. [x] Launch the server using the exact command `setup_retrieval.sh` printed, e.g.:
       ```
       uv run python scripts/retrieval_server.py --index_path data/wiki18/bm25-repo/bm25 --port 8000
       ```
       Run it in the background (or a separate terminal) — it needs to stay up for the rest of
       this phase's verification and for all of Phase 4/5's training runs later.
5. [x] Run the verification loop (see below). Fix anything it reports before moving on.

## Verification loop — how you know Phase 1 is actually done

Run:
```
uv run python scripts/verify_retrieval.py
```

This is not a suggestion to eyeball output — it is the literal exit-criteria gate. It checks,
mechanically:
1. `GET /health` responds (server is up at all).
2. `POST /retrieve` returns the documented response shape for a batch query.
3. A handful of titles **already confirmed to exist** in wiki-18 (`"Arthur's Magazine"`,
   `"First for Women"`, `"127 Hours"`, `"Absinthe"` — drawn from the 322/400 gold-title sample
   verified during design research) come back as a top-3 hit when searched by their own title
   text, with non-empty retrieved text — i.e. the index, the title-parsing logic, and the
   corpus/embedded-doc lookup are all wired together correctly.
4. A title **already confirmed absent** from wiki-18 (`"Calgary"` — see CLAUDE.md's "~80%, not
   100%" note) does not crash the server; it should degrade gracefully to a best-effort (possibly
   irrelevant) result, same as any real query that happens to miss.

The script prints `PASS: ... Phase 1 exit criteria met -- safe to start Phase 2.` and exits 0
only if all of the above hold. If it prints `FAIL`, read exactly which check failed — do not
proceed to Phase 2 until it passes, and do not consider "the script ran without a Python
exception" sufficient on its own; the checks above are the actual bar.

## Exit criteria (all must be true before handing off)

- [x] `scripts/verify_retrieval.py` prints `PASS` and exits 0.
- [x] The exact retrieval server launch command (with real paths) is recorded in Handoff notes
      below, so Phase 4/5 can restart it without re-running setup.
- [x] Whether the 5.12GB corpus download was actually needed (`contains_doc` true or false) is
      recorded below — this determines whether Phase 4/5 need to keep the corpus file around.

## Handoff notes

- **JDK**: `openjdk-21-jdk` installed via `apt`, confirmed `openjdk version "21.0.11"`.
- **`contains_doc` result: `False`** — the Lucene index does *not* embed raw documents, so the
  5.12GB `PeterJinGo/wiki-18-corpus` download *was* needed. Both `data/wiki18/bm25-repo/` (index)
  and `data/wiki18/corpus-repo/` + the extracted `data/wiki18/data00/.../wiki_dump.jsonl` (corpus)
  must be kept around for Phase 4/5 — don't delete the corpus thinking it's a one-time-use
  intermediate.
- **Exact launch command** (restart with this if the server ever dies or the machine reboots):
  ```
  uv run python scripts/retrieval_server.py \
      --index_path data/wiki18/bm25-repo/bm25 \
      --corpus_path data/wiki18/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl \
      --port 8000
  ```
  Currently running in the background of this session (Bash `run_in_background`) — it is *not*
  under systemd/tmux, so it will not survive a machine reboot or this session ending in a way
  that kills child processes. Restart with the command above if `GET /health` stops responding.
- **Bug found and fixed in `scripts/retrieval_server.py`**: in the `contains_doc=False` fallback
  path, `search_one()` was returning the raw corpus row (`{"id", "contents"}`) directly instead
  of running it through `parse_title_text()` — every `/retrieve` call crashed with
  `KeyError: 'title'` the moment a real query came in, since only the `contains_doc=True` branch
  parsed the title out. Fixed by wrapping that branch's return value in `parse_title_text(...)`
  too, matching the other branch. This was caught immediately by `verify_retrieval.py`, not
  discovered later — exactly the kind of bug the exit-criteria script exists to catch.
- **`verify_retrieval.py`'s fixture titles were wrong and got corrected.** Of the four titles
  CLAUDE.md documented as "confirmed present" (from the Phase 0 research's 322/400 gold-title
  sample), direct exact-title scans of the actual downloaded corpus show:
  - `"Arthur's Magazine"` and `"First for Women"` do **not** exist as exact titles at all (0
    matches) — contradicts the earlier "confirmed present" claim; likely that check used a
    different corpus copy or a looser match. Only `"Arthur's Lady's Home Magazine"` and
    `"Arthur (magazine)"` exist for the former.
  - `"Absinthe"` does exist (61 near-duplicate 100-word passage chunks all titled "Absinthe"),
    but BM25 ranks other similarly-worded articles (`"La Fée Absinthe"`, `"Hapsburg Absinthe"`)
    above it for the query `"Absinthe"` — a real property of this passage-chunked corpus (many
    identically-titled chunks competing, versus shorter unique-titled articles with tighter term
    overlap), not a wiring bug.
  - `"127 Hours"` passed cleanly, proving the server's plumbing itself was correct.
  Replaced the fixture list with four titles verified live against the running server to both
  exist and rank top-3 for their own title-as-query: `"127 Hours"`, `"Big Stone Gap (film)"`,
  `"Peter Schmeichel"`, `"Virginia Commonwealth University"`. **Implication for later phases**:
  don't assume every title in the Phase 0 "322/400 confirmed" sample individually holds up under
  direct re-check — the ~80% population-level statistic in CLAUDE.md is still believed sound, but
  a handful of specific anecdotal examples from that research were apparently mischecked.
  Anything that needs an individual title-existence guarantee should re-verify directly against
  this actual downloaded corpus rather than trusting the CLAUDE.md example list.
- **No deviations from the provided scripts' design** beyond the one bug fix above — the overall
  approach (`setup_retrieval.sh` → launch → `verify_retrieval.py`) worked as documented.
