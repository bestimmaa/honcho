"""Deterministic guard against near-miss misspellings of a peer id.

The minimal deriver asks the LLM to write the observed peer's id verbatim as the
subject of every observation it emits. Smaller/local models frequently emit a
near-miss instead (``chrs``/``chrsi``/``chrus`` for ``chris``). That is a silent
recall failure: the peer name is a high-weight token in the observation
embedding, so a mangled subject drifts away from any query naming that peer and
falls outside the caller's distance gate. The observation is written, stored,
and never retrieved -- nothing errors.

Prompt-level anchoring was tried and did not eliminate it, so this module
provides a post-processing guard applied before observations reach storage.

Design bias: **under-correct**. A missed corruption costs one unrecalled
observation. A false correction silently rewrites a true statement about
somebody else. Every rule here is deliberately narrow:

* Only a *single* edit away (Damerau-Levenshtein <= 1, i.e. one substitution,
  insertion, deletion, or adjacent transposition). Every corruption observed in
  production is distance 1, while the near-miss words that legitimately occur in
  the corpus (``chips``, ``chars``, ``charts``, ``cheri``) are all distance >= 2.
* Only when the first character matches.
* Only for peer ids of at least :data:`MIN_PEER_ID_LENGTH` characters, so short
  ids like ``pi`` never trigger anything.
* Never after the first correct spelling of the peer id in the same
  observation. The subject leads, so a near-miss ahead of the first correct
  spelling is still a mangled subject and is repaired; a near-miss after it is
  a later mention that the sentence may well have introduced as somebody else.
* Never for a token that is another real peer in the same workspace -- an
  observation by ``claude`` about ``chris`` is valid.
* Never for a token that is a known English word.
* Never for a token embedded in a path, URL, email, or code span -- the corpus
  contains real truncated paths such as ``/private/tmp/claude-501/-Users-chr``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

__all__ = [
    "MIN_PEER_ID_LENGTH",
    "PeerNameCorrection",
    "normalize_observation_subject",
]

# Peer ids shorter than this are left alone entirely: a single edit away from a
# 2-4 character id sweeps in far too much ordinary text.
MIN_PEER_ID_LENGTH = 5

# Unicode dash variants that models substitute for a plain hyphen. These are
# folded before comparison so that ``user‑chris`` is recognised as a
# correct spelling of ``user-chris`` rather than a corruption of it.
_DASHES = "‐‑‒–—―−"
_DASH_TRANSLATION = {ord(d): "-" for d in _DASHES}

# A token is a letter-initial run of letters/digits/hyphens/underscores. Hyphens
# are included so that a hyphenated peer id such as ``user-chris`` is matched as
# one unit and ``user-chrus`` is one edit away from it.
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_" + _DASHES + r"-]*")

# Inline code spans are masked out wholesale: everything inside them is a
# literal (command lines, file names, config keys) and must survive verbatim.
_CODE_SPAN_RE = re.compile(r"`[^`]*`")

# Characters that may sit immediately before/after a token. Anything else (``/``
# ``\`` ``@`` ``.`` ``:`` ``=`` ``#`` ...) means the token is part of a larger
# structured literal -- a path, URL, email, or dotted identifier -- and is left
# alone.
_ALLOWED_BEFORE = frozenset(" \t\n\r\f\v([{\"'*_~<“‘«")
_ALLOWED_AFTER = frozenset(" \t\n\r\f\v)]},.;:!?\"'*_~>”’»")

# Last-line-of-defence word list. The distance-1 rule already excludes every
# near-miss word seen in the live corpus; this is a deterministic backstop so
# that an ordinary English word is never rewritten into a peer id. Kept
# hardcoded on purpose -- reading a system word list would make behaviour differ
# between a developer machine and the container.
_COMMON_WORDS_SOURCE = """
    about above after again against all also always among and another any are around
    because been before being below between both bring build built call came can cannot
    change changed chars chart charts chat check checked chips choice choose chose class
    clean clear close code come common config could count create created data date days
    debug default delete described detail did does done down during each early edit either
    else end enough error even ever every example except expect failed field file files
    find first fix fixed following for form found from full further gave general get give
    given goes going good got great group had half hand has have help her here high him
    his hold home host how however http image implement important index info inside instead
    into issue item items just keep kept key kind knew know known large last later latest
    least leave left less let level like line link list little local log logs long look
    made main make man many match may mean means meet member mention merge message method
    might mode model more most move much must name named need needed never new next night
    none nor not note nothing now number object off often old once one only open option
    order other our out over own page pair part path pattern people per place plan point
    port possible power prefer present press previous problem process program project
    provide public put query question quite ran range rather read real really reason
    record reference remove removed report request require result return review right room
    round rule run running said same save saw say says school second see seem seen self
    send sent server service session set setting several shall she short should show side
    similar since single small some sort sound source space start state status step still
    stop store string strong style such support sure system table take taken talk task
    tell test text than that the their them then there these they thing think this those
    though three through time tool took top total toward track true try turn two type
    under until upon usage use used user using usual valid value version very view want
    was watch water way week well went were what when where which while white who whole
    why will wish with within without word work world would write written wrong year yes
    yet you your
    """

_COMMON_WORDS = frozenset(_COMMON_WORDS_SOURCE.split())


@dataclass(frozen=True, slots=True)
class PeerNameCorrection:
    """A single rewrite the guard performed, for logging."""

    variant: str
    """The misspelling as the model emitted it."""

    count: int
    """How many times it was replaced in this observation."""


def _fold(token: str) -> str:
    """Lowercase a token and fold unicode dash variants to a plain hyphen."""
    return token.lower().translate(_DASH_TRANSLATION)


def _within_one_edit(a: str, b: str) -> bool:
    """Return True if ``a`` and ``b`` differ by exactly one Damerau edit.

    One substitution, one insertion, one deletion, or one adjacent
    transposition. Identical strings return False -- callers handle equality
    separately.
    """
    len_a, len_b = len(a), len(b)
    if abs(len_a - len_b) > 1:
        return False
    if a == b:
        return False

    if len_a == len_b:
        diffs = [i for i in range(len_a) if a[i] != b[i]]
        if len(diffs) == 1:
            return True
        if len(diffs) == 2 and diffs[1] == diffs[0] + 1:
            # adjacent transposition
            return a[diffs[0]] == b[diffs[1]] and a[diffs[1]] == b[diffs[0]]
        return False

    longer, shorter = (a, b) if len_a > len_b else (b, a)
    i = j = 0
    skipped = False
    while i < len(longer) and j < len(shorter):
        if longer[i] == shorter[j]:
            i += 1
            j += 1
            continue
        if skipped:
            return False
        skipped = True
        i += 1
    return True


def _match_ok(content: str, start: int, end: int) -> bool:
    """Reject tokens glued to a path/URL/email/identifier separator."""
    if start > 0 and content[start - 1] not in _ALLOWED_BEFORE:
        return False
    return not (end < len(content) and content[end] not in _ALLOWED_AFTER)


def _restore_case(variant: str, peer_id: str) -> str:
    """Apply the variant's capitalisation style to the canonical peer id."""
    if variant.isupper() and not variant.islower():
        return peer_id.upper()
    if variant[:1].isupper():
        return peer_id[:1].upper() + peer_id[1:]
    return peer_id


def normalize_observation_subject(
    content: str,
    peer_id: str,
    known_peer_ids: Iterable[str] | None = None,
) -> tuple[str, list[PeerNameCorrection]]:
    """Repair near-miss misspellings of ``peer_id`` in an observation.

    Args:
        content: The observation text as emitted by the model.
        peer_id: The observed peer's id, which the model was told to copy
            verbatim.
        known_peer_ids: Other real peer ids in scope (workspace/batch). Tokens
            matching one of these are never rewritten, so an observation naming
            a co-participant survives untouched.

    Returns:
        ``(content, corrections)``. ``corrections`` is empty when nothing was
        changed, in which case ``content`` is returned unmodified.
    """
    if not content or not peer_id or len(peer_id) < MIN_PEER_ID_LENGTH:
        return content, []

    target = _fold(peer_id)
    others = {_fold(p) for p in known_peer_ids or ()} - {target}

    # Mask inline code spans so their contents are never considered.
    masked = _CODE_SPAN_RE.sub(lambda m: " " * len(m.group(0)), content)

    candidates: list[tuple[int, int, str]] = []
    for match in _TOKEN_RE.finditer(masked):
        start, end = match.span()
        token = match.group(0)
        folded = _fold(token)

        if not _match_ok(masked, start, end):
            # Part of a path, URL, email, or dotted identifier -- not a subject.
            continue

        if folded == target:
            # The subject leads the observation, so a near-miss *before* the
            # first correct spelling is a mangled subject and still gets
            # repaired. Anything after it is a later mention -- more likely a
            # different entity the sentence went on to introduce -- and is left
            # alone.
            break

        if folded in others or folded in _COMMON_WORDS:
            continue
        if folded[0] != target[0]:
            continue
        if not _within_one_edit(folded, target):
            continue

        candidates.append((start, end, token))

    if not candidates:
        return content, []

    counts: dict[str, int] = {}
    pieces: list[str] = []
    cursor = 0
    for start, end, token in candidates:
        pieces.append(content[cursor:start])
        pieces.append(_restore_case(token, peer_id))
        cursor = end
        counts[token] = counts.get(token, 0) + 1
    pieces.append(content[cursor:])

    corrections = [PeerNameCorrection(v, c) for v, c in counts.items()]
    return "".join(pieces), corrections
