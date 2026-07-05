from turn_level_rewards.env import SearchEnv

HOTPOT_ROW_1 = {
    "metadata": {
        "supporting_facts": {"title": ["127 Hours", "Peter Schmeichel"], "sent_id": [0, 0]},
    },
}

HOTPOT_ROW_2 = {
    "metadata": {
        "supporting_facts": {
            "title": ["Big Stone Gap (film)", "Virginia Commonwealth University"],
            "sent_id": [0, 0],
        },
    },
}


def _fake_retriever(docs_by_query):
    def retrieve(query, topk):
        return docs_by_query.get(query, [])

    return retrieve


def test_search_hitting_gold_title_updates_retrieval_fraction():
    retrieve = _fake_retriever(
        {
            "127 hours survival": [
                {
                    "title": "127 Hours",
                    "text": "A 2010 survival drama film.",
                    "contents": '"127 Hours"\nA 2010 survival drama film.',
                }
            ]
        }
    )
    env = SearchEnv(retrieve_fn=retrieve)
    env.reset(**HOTPOT_ROW_1)

    env.search("127 hours survival")

    assert env.retrieval_fraction == 0.5


def test_search_hitting_distractor_does_not_update_retrieval_fraction():
    retrieve = _fake_retriever(
        {
            "danish goalkeepers": [
                {
                    "title": "Football in Denmark",
                    "text": "Association football is the most popular sport in Denmark.",
                    "contents": '"Football in Denmark"\nAssociation football is the most popular sport in Denmark.',
                }
            ]
        }
    )
    env = SearchEnv(retrieve_fn=retrieve)
    env.reset(**HOTPOT_ROW_1)

    env.search("danish goalkeepers")

    assert env.retrieval_fraction == 0.0


def test_reset_twice_in_a_row_clears_prior_episode_state():
    retrieve = _fake_retriever(
        {
            "127 hours survival": [
                {
                    "title": "127 Hours",
                    "text": "A 2010 survival drama film.",
                    "contents": '"127 Hours"\nA 2010 survival drama film.',
                }
            ]
        }
    )
    env = SearchEnv(retrieve_fn=retrieve)
    env.reset(**HOTPOT_ROW_1)
    env.search("127 hours survival")
    assert env.retrieval_fraction == 0.5

    env.reset(**HOTPOT_ROW_2)

    assert env.retrieval_fraction == 0.0


def test_retrieval_fraction_caps_at_one_on_duplicate_hit():
    retrieve = _fake_retriever(
        {
            "127 hours": [{"title": "127 Hours", "text": "...", "contents": '"127 Hours"\n...'}],
            "peter schmeichel": [
                {
                    "title": "Peter Schmeichel",
                    "text": "...",
                    "contents": '"Peter Schmeichel"\n...',
                }
            ],
        }
    )
    env = SearchEnv(retrieve_fn=retrieve)
    env.reset(**HOTPOT_ROW_1)

    env.search("127 hours")
    env.search("127 hours")  # duplicate hit, must not double-count
    env.search("peter schmeichel")

    assert env.retrieval_fraction == 1.0


def test_search_returns_readable_string_with_title_and_text():
    retrieve = _fake_retriever(
        {
            "127 hours": [
                {
                    "title": "127 Hours",
                    "text": "A 2010 survival drama film.",
                    "contents": '"127 Hours"\nA 2010 survival drama film.',
                }
            ]
        }
    )
    env = SearchEnv(retrieve_fn=retrieve)
    env.reset(**HOTPOT_ROW_1)

    result = env.search("127 hours")

    assert "127 Hours" in result
    assert "A 2010 survival drama film." in result


def test_search_with_no_results_returns_message_not_error():
    retrieve = _fake_retriever({})
    env = SearchEnv(retrieve_fn=retrieve)
    env.reset(**HOTPOT_ROW_1)

    result = env.search("nonexistent query")

    assert result == "No results found."
