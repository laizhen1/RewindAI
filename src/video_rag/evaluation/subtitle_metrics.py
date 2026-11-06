"""Token error counts for explicitly reviewed, time-aligned subtitle cues."""
from __future__ import annotations

import re
import unicodedata

# CJK ideographs are separate tokens; other Unicode words and apostrophes form words.
_TOKEN = re.compile(r"[\u3400-\u9fff]|[^\W_\u3400-\u9fff]+(?:['’][^\W_\u3400-\u9fff]+)?", re.UNICODE)
MAX_TOKENS_PER_CUE = 200
MAX_REVIEWED_CUES = 200


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(unicodedata.normalize('NFKC', text).casefold())


def edit_counts(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    """Minimum-edit substitutions, deletions and insertions, using linear memory."""
    if max(len(reference), len(hypothesis)) > MAX_TOKENS_PER_CUE:
        raise ValueError('A reviewed subtitle cue has too many tokens')
    # Entries are (distance, substitutions, deletions, insertions).
    prev = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, ref in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, hyp in enumerate(hypothesis, 1):
            d, sub, deletion, insertion = prev[j - 1]
            substitution = (d + int(ref != hyp), sub + int(ref != hyp), deletion, insertion)
            d, sub, deletion, insertion = prev[j]
            delete = (d + 1, sub, deletion + 1, insertion)
            d, sub, deletion, insertion = current[j - 1]
            insert = (d + 1, sub, deletion, insertion + 1)
            current.append(min((substitution, delete, insert), key=lambda item: item[0]))
        prev = current
    return prev[-1][1:]


def reviewed_token_error(cues, edits: dict[int, str], reviewed: list[int]) -> dict:
    if (not isinstance(reviewed, list) or not 1 <= len(reviewed) <= MAX_REVIEWED_CUES
            or any(type(i) is not int or not 0 <= i < len(cues) for i in reviewed)
            or len(set(reviewed)) != len(reviewed)):
        raise ValueError('Choose 1 to 200 distinct subtitle cues to review')
    substitutions = deletions = insertions = reference_tokens = 0
    for i in reviewed:
        ref = tokens(edits.get(i, cues[i].text))
        hyp = tokens(cues[i].text)
        if not ref:
            raise ValueError('A reviewed cue has no reference tokens')
        s, d, insert = edit_counts(ref, hyp)
        substitutions += s
        deletions += d
        insertions += insert
        reference_tokens += len(ref)
    errors = substitutions + deletions + insertions
    return {'reviewed_cues': len(reviewed), 'reference_tokens': reference_tokens,
            'substitutions': substitutions, 'deletions': deletions, 'insertions': insertions,
            'token_errors': errors, 'token_error_rate': errors / reference_tokens}
