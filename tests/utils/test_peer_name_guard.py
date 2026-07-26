"""Tests for the deterministic peer-name guard.

Run without the session-scoped DB fixture:

    uv run pytest tests/utils/test_peer_name_guard.py -q --noconftest
"""

import pytest

from src.utils.peer_name_guard import normalize_observation_subject

# Every corrupted spelling of `chris` observed in live deriver output.
OBSERVED_CORRUPTIONS = [
    "chrs",
    "chri",
    "chrsi",
    "chres",
    "chrid",
    "chrfis",
    "chrus",
    "chrsis",
    "chrirs",
    "chiris",
    "chis",
    "chhis",
    "chras",
    "chuis",
]

# Corrupted spellings observed for the hyphenated peer id `user-chris`.
OBSERVED_PREFIXED_CORRUPTIONS = [
    "user-chrus",
    "user-chiris",
    "user-christ",
]

# Live peer ids, used as the "other real peers" set.
LIVE_PEERS = [
    "chris",
    "claude",
    "user-chris",
    "opencode",
    "bagface",
    "pi",
    "alice",
    "assistant",
]


def guard(content: str, peer_id: str = "chris", peers: list[str] | None = None):
    return normalize_observation_subject(content, peer_id, peers or LIVE_PEERS)


class TestExactMatchIsNoOp:
    @pytest.mark.parametrize(
        "content",
        [
            "chris prefers dark mode in his editor.",
            "On May 22, 2026, chris confirmed the fix was applied.",
            "**Chris** told **opencode** that he already has a config.yml file.",
            "chris has a friend named Sarah, and chris is planning to meet sarah.",
        ],
    )
    def test_correct_spelling_untouched(self, content: str):
        out, corrections = guard(content)
        assert out == content
        assert corrections == []

    def test_unicode_hyphen_counts_as_correct(self):
        # U+2011 non-breaking hyphen occurs in real stored observations.
        content = "user‑chris will check service runtime logs."
        out, corrections = guard(content, "user-chris")
        assert out == content
        assert corrections == []

    def test_correct_spelling_elsewhere_suppresses_correction(self):
        # Under-correction is deliberate: if the model got it right once, a
        # similar-looking token is more likely a different entity.
        content = "chris asked about chrs and the deploy."
        out, corrections = guard(content)
        assert out == content
        assert corrections == []


class TestObservedCorruptions:
    @pytest.mark.parametrize("variant", OBSERVED_CORRUPTIONS)
    def test_leading_subject_is_repaired(self, variant: str):
        out, corrections = guard(f"{variant} prefers dark mode in his editor.")
        assert out == "chris prefers dark mode in his editor."
        assert len(corrections) == 1
        assert corrections[0].variant == variant

    @pytest.mark.parametrize("variant", OBSERVED_CORRUPTIONS)
    def test_repaired_after_a_date_prefix(self, variant: str):
        # Most real observations lead with a timestamp, not the subject.
        out, _ = guard(f"On May 22, 2026, {variant} confirmed the fix was applied.")
        assert out == "On May 22, 2026, chris confirmed the fix was applied."

    @pytest.mark.parametrize("variant", OBSERVED_PREFIXED_CORRUPTIONS)
    def test_hyphenated_peer_id_is_repaired(self, variant: str):
        out, corrections = guard(f"{variant} restarted the container.", "user-chris")
        assert out == "user-chris restarted the container."
        assert len(corrections) == 1

    def test_capitalisation_is_preserved(self):
        out, _ = guard("Chrs prefers dark mode.")
        assert out == "Chris prefers dark mode."

    def test_markdown_emphasis_is_preserved(self):
        out, _ = guard("At 19:04, **Chrs** told **opencode** about the plan.")
        assert out == "At 19:04, **Chris** told **opencode** about the plan."

    def test_all_occurrences_repaired(self):
        out, corrections = guard("chrs has a friend named Sarah, and chrs met sarah.")
        assert out == "chris has a friend named Sarah, and chris met sarah."
        assert corrections[0].count == 2

    def test_possessive_survives(self):
        out, _ = guard("chrs's laptop is a MacBook.")
        assert out == "chris's laptop is a MacBook."


class TestLegitimateWordsAreNotRewritten:
    @pytest.mark.parametrize(
        "content",
        [
            "The UI uses chips to display selected filters.",
            "chars is a keyword used in MEMORY.md.",
            "The OpenObserve dashboard renders charts for host metrics.",
            "The contact address is cheri@example.com.",
            "The plan mentions chars, chips and charts on the same page.",
        ],
    )
    def test_near_radius_words_untouched(self, content: str):
        out, corrections = guard(content)
        assert out == content
        assert corrections == []

    @pytest.mark.parametrize(
        "content",
        [
            "The path is /Users/christopher/dev/app on the second machine.",
            "christopher is the macOS account name on the other laptop.",
            "The temp dir is /private/tmp/claude-501/-Users-chr and is truncated.",
            "Run `docker compose logs chrs` to inspect the container.",
            "The file lives at ~/dev/chrs/notes.md on disk.",
        ],
    )
    def test_paths_identifiers_and_code_spans_untouched(self, content: str):
        out, corrections = guard(content)
        assert out == content
        assert corrections == []


class TestOtherPeersAndPeopleAreNotRewritten:
    def test_other_real_peer_untouched(self):
        # An observation by claude about chris is valid in both directions.
        content = "claude summarised the deploy for the team."
        out, corrections = guard(content, "chris")
        assert out == content
        assert corrections == []

    def test_near_miss_peer_id_that_is_itself_a_real_peer(self):
        peers = ["chris", "chres", "claude"]
        content = "chres reviewed the pull request."
        out, corrections = normalize_observation_subject(content, "chris", peers)
        assert out == content
        assert corrections == []

    def test_third_party_name_untouched(self):
        content = "Sarah asked about the meeting, and Chloe agreed to join."
        out, corrections = guard(content)
        assert out == content
        assert corrections == []

    def test_hyphenated_peer_not_confused_with_bare_peer(self):
        # `chris` and `user-chris` are distinct live peers.
        content = "chris merged the branch."
        out, corrections = guard(content, "user-chris")
        assert out == content
        assert corrections == []


class TestGuardScope:
    @pytest.mark.parametrize("peer_id", ["pi", "ab", "abcd"])
    def test_short_peer_ids_are_never_guarded(self, peer_id: str):
        content = "pa did something and abce did something else."
        out, corrections = normalize_observation_subject(content, peer_id, LIVE_PEERS)
        assert out == content
        assert corrections == []

    @pytest.mark.parametrize("peer_id", ["opencode", "bagface", "alice", "assistant"])
    def test_generic_over_peer_ids(self, peer_id: str):
        variant = peer_id[:2] + peer_id[3:]  # single deletion
        out, corrections = normalize_observation_subject(
            f"{variant} ran the compliance pass.", peer_id, LIVE_PEERS
        )
        assert out == f"{peer_id} ran the compliance pass."
        assert len(corrections) == 1

    @pytest.mark.parametrize("content", ["", "   "])
    def test_empty_content(self, content: str):
        out, corrections = guard(content)
        assert out == content
        assert corrections == []

    def test_two_edits_away_is_not_corrected(self):
        # Bias is toward under-correcting; distance 2 is out of scope.
        content = "chrus is fine but chrrus is two edits away."
        out, _ = guard(content)
        assert "chris is fine" in out
        assert "chrrus" in out
