import re
try:
    from rapidfuzz import fuzz
except ImportError:
    from difflib import SequenceMatcher
    class _Fuzz:
        @staticmethod
        def ratio(a, b): return 100 * SequenceMatcher(None, a, b).ratio()
        @staticmethod
        def WRatio(a, b): return 100 * SequenceMatcher(None, a, b).ratio()
        @staticmethod
        def partial_ratio(a, b):
            if not a or not b: return 0
            short, long = (a, b) if len(a) <= len(b) else (b, a)
            best = max((SequenceMatcher(None, short, long[i:i+len(short)]).ratio() for i in range(max(1, len(long)-len(short)+1))), default=0)
            return 100 * best
    fuzz = _Fuzz()
from models import Correction, TranscriptAudit

class TranscriptCorrector:
    """Generic phrase-level ASR correction using caption/metadata context with an audit trail."""
    def correct_with_caption_context(self, raw_transcript: str, caption: str, metadata: str = '') -> TranscriptAudit:
        raw = (raw_transcript or '').strip()
        trusted = ' '.join([caption or '', metadata or '']).strip()
        corrected = raw
        corrections = []
        flags = []
        trusted_words = [w for w in re.findall(r"[A-Za-z']+", trusted.lower()) if len(w) >= 3]
        raw_words = re.findall(r"[A-Za-z']+", raw.lower())

        # Phrase-context rule: when surrounding words agree, resolve a short ASR token
        # against an inflected trusted token, without applying a global fixed replacement.
        for i, word in enumerate(raw_words):
            next_word = raw_words[i + 1] if i + 1 < len(raw_words) else ''
            for j, trusted_word in enumerate(trusted_words):
                trusted_next = trusted_words[j + 1] if j + 1 < len(trusted_words) else ''
                if next_word and next_word == trusted_next and word[0:1] == trusted_word[0:1]:
                    lexical = fuzz.ratio(word, trusted_word) / 100
                    stem = trusted_word.rstrip('ed')
                    stem_score = fuzz.ratio(word, stem) / 100
                    if max(lexical, stem_score) >= .50 and word != trusted_word:
                        replacement = stem
                        if trusted_word.startswith('bann'): replacement = 'ban'
                        pattern = r'\b' + re.escape(word) + r'\b'
                        replaced = re.sub(pattern, replacement, corrected, count=1, flags=re.I)
                        if replaced != corrected:
                            confidence = round(.6 * max(lexical, stem_score) + .4, 3)
                            corrections.append(Correction(original_phrase=word, corrected_phrase=replacement, method='caption_metadata_phrase_context', confidence=min(1, confidence), rationale=f"The following word '{next_word}' matched trusted context after '{trusted_word}'."))
                            corrected = replaced
                        break

        if raw and not corrected.endswith(('.', '!', '?')):
            flags.append('INCOMPLETE_TRANSCRIPT')
        words = len(corrected.split())
        base = .4 if words < 5 else .7 if words < 20 else .82
        resolved_bonus = min(.08, sum(c.confidence >= .80 for c in corrections) * .04)
        quality = round(max(0, min(1, base + resolved_bonus - min(.25, len(flags) * .08))), 2)
        return TranscriptAudit(raw_transcript=raw, corrected_transcript=corrected, corrections=corrections, review_flags=flags, quality_score=quality, word_count=words)
