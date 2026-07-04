#!/usr/bin/env python3
"""Phase 1 exit-criteria check.

Answers the question "how does the agent know retrieval infra is set up correctly and ready
for Phase 2?" mechanically: run this against a live retrieval_server.py instance. It exits 0
and prints PASS only if every check below passes; otherwise it prints exactly which check
failed and exits 1. Do not mark Phase 1 done, and do not start Phase 2, until this passes.

Usage: uv run python scripts/verify_retrieval.py [--url http://localhost:8000]
"""

import argparse
import sys

import requests

# A few titles CLAUDE.md already confirmed exist in wiki-18 (from the 322/400 gold-title sample
# verified in Phase 0 research), and one confirmed-absent title ("Calgary" — see CLAUDE.md's
# "Confirmed ... title alignment is ~80%, not 100%" note) to check the server degrades
# gracefully on a real corpus gap instead of erroring.
KNOWN_PRESENT_TITLES = ["Arthur's Magazine", "First for Women", "127 Hours", "Absinthe"]
KNOWN_ABSENT_TITLE = "Calgary"


def check(url: str) -> list[str]:
    failures = []

    # 1. Health check.
    try:
        r = requests.get(f"{url}/health", timeout=10)
        r.raise_for_status()
    except Exception as e:
        failures.append(f"GET /health failed: {e}")
        return failures  # no point continuing if the server isn't even up

    # 2. Basic /retrieve call, well-formed response shape.
    try:
        r = requests.post(
            f"{url}/retrieve",
            json={"queries": KNOWN_PRESENT_TITLES, "topk": 3, "return_scores": True},
            timeout=30,
        )
    except Exception as e:
        failures.append(f"POST /retrieve raised {e}")
        return failures
    if r.status_code != 200:
        failures.append(f"POST /retrieve returned {r.status_code}: {r.text[:300]}")
        return failures
    body = r.json()
    if "result" not in body or len(body["result"]) != len(KNOWN_PRESENT_TITLES):
        failures.append(f"Unexpected /retrieve response shape: {body}")
        return failures

    # 3. Each known-present title should come back as a top-3 hit for its own title-as-query,
    #    and the returned document must parse to a non-empty, sensible {title, text}. This is
    #    a plumbing check (index + title-parsing wired correctly), not a semantic-quality check.
    for query, hits in zip(KNOWN_PRESENT_TITLES, body["result"], strict=True):
        if not hits:
            failures.append(f"No hits at all for known-present title query {query!r}")
            continue
        titles_returned = [h["document"]["title"] for h in hits]
        if query not in titles_returned:
            failures.append(
                f"Known-present title {query!r} not found in top-3 results: {titles_returned}"
            )
        top_doc = hits[0]["document"]
        if not top_doc.get("text", "").strip():
            failures.append(f"Top hit for {query!r} has empty text: {top_doc}")

    # 4. A known-absent title must not crash the server -- it should return a graceful
    #    best-effort (possibly irrelevant) result, matching the confirmed ~80% corpus ceiling.
    try:
        r = requests.post(
            f"{url}/retrieve",
            json={"queries": [KNOWN_ABSENT_TITLE], "topk": 3, "return_scores": True},
            timeout=30,
        )
    except Exception as e:
        failures.append(f"POST /retrieve for known-absent title raised {e}")
    else:
        if r.status_code != 200:
            failures.append(
                f"Known-absent title {KNOWN_ABSENT_TITLE!r} caused an error instead of a "
                f"graceful best-effort response: {r.status_code} {r.text[:300]}"
            )

    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    failures = check(args.url)
    if failures:
        print("FAIL -- Phase 1 is not done yet:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("PASS: retrieval server is up, wired correctly, and returns real documents.")
    print("Phase 1 exit criteria met -- safe to start Phase 2.")
    sys.exit(0)
