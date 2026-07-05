"""SearchEnv: a TRL environment_factory-compatible multi-turn search environment.

Instances are reused from a pool across episodes (see CLAUDE.md's "TRL mechanics being relied
on") -- reset() must fully reinitialize all mutable state, with zero leftover state from a
prior episode.
"""

from collections.abc import Callable

import requests

DocList = list[dict[str, str]]


def _default_retrieve_fn(query: str, topk: int, base_url: str) -> DocList:
    """POST a single query to the real retrieval server (scripts/retrieval_server.py).

    The server already splits each document's title/text out of its `contents` field
    server-side (see its `parse_title_text`) -- this trusts that and does no re-parsing.
    """
    response = requests.post(
        f"{base_url}/retrieve",
        json={"queries": [query], "topk": topk, "return_scores": False},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["result"][0]


class SearchEnv:
    """Multi-turn search environment exposing `search` as a tool for GRPO's environment_factory.

    Tracks the fraction of gold supporting-fact titles surfaced by search() calls during an
    episode -- this is the turn-level reward signal (CLAUDE.md's "Reward design" section).
    """

    def __init__(
        self,
        retrieve_fn: Callable[[str, int], DocList] | None = None,
        topk: int = 3,
        base_url: str = "http://localhost:8000",
    ) -> None:
        self._retrieve_fn = retrieve_fn or (
            lambda query, k: _default_retrieve_fn(query, k, base_url)
        )
        self._topk = topk
        self._gold_titles: frozenset[str] = frozenset()
        self._hit_titles: set[str] = set()

    def reset(self, metadata: dict, **kwargs: object) -> None:
        """Reset all per-episode state for a newly sampled row.

        Args:
            metadata (dict): The row's metadata; metadata["supporting_facts"]["title"] is the
                list of gold paragraph titles for this question.
        """
        self._gold_titles = frozenset(metadata["supporting_facts"]["title"])
        self._hit_titles = set()

    def search(self, query: str) -> str:
        """Search the wiki-18 corpus for passages relevant to a query.

        Args:
            query: The search query text.

        Returns:
            str: The top retrieved passages formatted as numbered documents with title and
                text, or a message indicating no results were found.
        """
        docs = self._retrieve_fn(query, self._topk)
        for doc in docs:
            if doc["title"] in self._gold_titles:
                self._hit_titles.add(doc["title"])
        if not docs:
            return "No results found."
        return "\n".join(
            f'Doc {i} (Title: "{doc["title"]}"): {doc["text"]}' for i, doc in enumerate(docs, 1)
        )

    @property
    def retrieval_fraction(self) -> float:
        """Fraction of gold supporting-fact titles surfaced by search() this episode."""
        if not self._gold_titles:
            return 0.0
        return len(self._hit_titles) / len(self._gold_titles)
