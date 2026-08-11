"""Deterministic Statement Preclassification Engine

Pattern-based classification using regex and keyword rules.
NO LLM calls. Pure deterministic logic.

Classification confidence threshold:
- >= 0.90: Use deterministic classification
- 0.65-0.89: Use embedding similarity against prototypes  
- < 0.65: Escalate to LLM

This is the FIRST step in statement classification.
"""

import re
from typing import Tuple, Optional
from models_v2 import StatementType

# ==============================================================================
# PATTERN DEFINITIONS
# ==============================================================================

# Opinion indicators
OPINION_PATTERNS = [
    r'\bi\s+think\b',
    r'\bi\s+believe\b',
    r'\bin\s+my\s+opinion\b',
    r'\bi\s+feel\s+(like|that)\b',
    r'\bseems\s+to\s+me\b',
    r'\bpersonally\b',
    r'\bimo\b',  # "in my opinion" abbreviation
]

OPINION_KEYWORDS = {
    'best', 'worst', 'greatest', 'terrible', 'beautiful', 'ugly',
    'unfair', 'wrong', 'right', 'should', 'shouldnt', 'ought',
    'amazing', 'awful', 'fantastic', 'horrible'
}

# Feeling indicators
FEELING_PATTERNS = [
    r'\bi\s+feel\b',
    r'\bi\s+felt\b',
    r'\bmade\s+me\s+feel\b',
    r'\bi\s+was\s+(worried|scared|happy|sad|angry|excited)\b',
    r'\bi\s+(enjoyed|hated|loved|despised)\b',
]

# Prediction indicators  
PREDICTION_PATTERNS = [
    r'\bwill\s+(be|have|become|reach)\b',
    r'\bgoing\s+to\b',
    r'\blikely\s+to\b',
    r'\bexpected\s+to\b',
    r'\bcould\s+happen\b',
    r'\bmight\s+(be|become|reach)\b',
    r'\bby\s+(next\s+year|\d{4}|next\s+month)\b',
    r'\bin\s+the\s+(future|coming\s+(years?|months?))\b',
]

# Promotional indicators
PROMOTIONAL_PATTERNS = [
    r'\buse\s+(my\s+)?code\b',
    r'\blink\s+in\s+(bio|description)\b',
    r'\blimited\s+time\b',
    r'\bbuy\s+now\b',
    r'\b(get|save)\s+\d+%\b',
    r'\b(sponsored|ad|affiliate)\b',
    r'\bdiscount\b',
    r'\bguaranteed\b',
    r'\border\s+now\b',
    r'\bcheck\s+out\b',
    r'\bswipe\s+up\b',
]

PROMOTIONAL_KEYWORDS = {
    'shop', 'buy', 'purchase', 'sale', 'deal', 'offer',
    'promo', 'coupon', 'discount', 'limited', 'exclusive'
}

# Statistical indicators (numbers, percentages, rates)
STATISTICAL_PATTERNS = [
    r'\b\d+(\.\d+)?%',  # Percentages
    r'\$\d+',  # Currency
    r'\b\d+(\.\d+)?\s*(million|billion|trillion)\b',  # Large numbers
    r'\b\d+(\.\d+)?x\s+(more|less|higher|lower)\b',  # Ratios
    r'\brate\s+of\s+\d+',  # Rates
    r'\brank(ed)?\s+#?\d+\b',  # Rankings
]

# Causal indicators
CAUSAL_PATTERNS = [
    r'\bcauses?\b',
    r'\bleads?\s+to\b',
    r'\bresults?\s+in\b',
    r'\bbecause\s+of\b',
    r'\bdue\s+to\b',
    r'\bprevents?\b',
    r'\bcures?\b',
    r'\bdrives?\b',
    r'\btriggers?\b',
    r'\bcreates?\b',
]

# Logical argument indicators
LOGICAL_PATTERNS = [
    r'\btherefore\b',
    r'\bthus\b',
    r'\bhence\b',
    r'\bso\s+(that)?\b',
    r'\bwhich\s+means\b',
    r'\bif\s+.+\s+then\b',
    r'\bgiven\s+that\b',
    r'\bas\s+a\s+result\b',
]

# Engagement bait
ENGAGEMENT_PATTERNS = [
    r'\bcomment\s+(below|if)\b',
    r'\blet\s+me\s+know\b',
    r'\btag\s+someone\b',
    r'\bshare\s+this\b',
    r'\bfollow\s+for\s+more\b',
    r'\blike\s+if\b',
    r'\bcheck\s+(this|out)\b',
]

# Health-specific patterns
HEALTH_PATTERNS = [
    r'\b(cure|treat|heal|fix)s?\b',
    r'\bsupplement\b',
    r'\bdrug\b',
    r'\bmedication\b',
    r'\bdiagnosis\b',
    r'\bdisease\b',
    r'\bsymptoms?\b',
]

# Financial patterns
FINANCIAL_PATTERNS = [
    r'\bstock\b',
    r'\bcrypto(currency)?\b',
    r'\binvest(ment)?\b',
    r'\bmarket\b',
    r'\breturn\b',
    r'\bportfolio\b',
    r'\basset\b',
]

# Policy/legal patterns
POLICY_PATTERNS = [
    r'\blaw\b',
    r'\bregulation\b',
    r'\bpolicy\b',
    r'\bgovernment\b',
    r'\bbann?(ed|ing)?\b',
    r'\brestrict(ed|ion)?\b',
    r'\brequire(d|ment)?\b',
    r'\ballows?\b',
]

# ==============================================================================
# CLASSIFICATION FUNCTIONS
# ==============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for pattern matching"""
    return text.lower().strip()

def count_pattern_matches(text: str, patterns: list[str]) -> int:
    """Count how many patterns match in the text"""
    text_lower = normalize_text(text)
    count = 0
    for pattern in patterns:
        if re.search(pattern, text_lower):
            count += 1
    return count

def has_keyword(text: str, keywords: set[str]) -> bool:
    """Check if text contains any keyword"""
    text_lower = normalize_text(text)
    words = set(re.findall(r'\b\w+\b', text_lower))
    return bool(words & keywords)

def classify_opinion(text: str) -> Tuple[Optional[StatementType], float]:
    """Detect opinion statements"""
    matches = count_pattern_matches(text, OPINION_PATTERNS)
    has_kw = has_keyword(text, OPINION_KEYWORDS)
    
    # Strong opinion indicators
    if matches >= 2:
        return StatementType.OPINION, 0.95
    if matches == 1 and has_kw:
        return StatementType.OPINION, 0.92
    if has_kw and len(text.split()) < 20:  # Short evaluative statement
        return StatementType.OPINION, 0.88
    if matches == 1:
        return StatementType.OPINION, 0.85
        
    return None, 0.0

def classify_feeling(text: str) -> Tuple[Optional[StatementType], float]:
    """Detect feeling/emotional statements"""
    matches = count_pattern_matches(text, FEELING_PATTERNS)
    
    if matches >= 2:
        return StatementType.FEELING, 0.96
    if matches == 1:
        return StatementType.FEELING, 0.91
        
    return None, 0.0

def classify_prediction(text: str) -> Tuple[Optional[StatementType], float]:
    """Detect prediction statements"""
    matches = count_pattern_matches(text, PREDICTION_PATTERNS)
    
    # Check for temporal future indicators
    has_future_tense = bool(re.search(r'\bwill\b', normalize_text(text)))
    has_time_horizon = bool(re.search(r'\b(by|in|next)\s+(\d{4}|year|month|decade)\b', normalize_text(text)))
    
    if matches >= 2:
        return StatementType.PREDICTION, 0.94
    if matches == 1 and (has_future_tense or has_time_horizon):
        return StatementType.PREDICTION, 0.90
    if matches == 1:
        return StatementType.PREDICTION, 0.85
        
    return None, 0.0

def classify_promotional(text: str) -> Tuple[Optional[StatementType], float]:
    """Detect promotional content"""
    matches = count_pattern_matches(text, PROMOTIONAL_PATTERNS)
    has_kw = has_keyword(text, PROMOTIONAL_KEYWORDS)
    
    if matches >= 3:
        return StatementType.PROMOTIONAL_STATEMENT, 0.98
    if matches >= 2:
        return StatementType.PROMOTIONAL_STATEMENT, 0.95
    if matches == 1 and has_kw:
        return StatementType.PROMOTIONAL_STATEMENT, 0.92
    if matches == 1:
        return StatementType.PROMOTIONAL_STATEMENT, 0.87
        
    return None, 0.0

def classify_engagement_bait(text: str) -> Tuple[Optional[StatementType], float]:
    """Detect engagement bait"""
    matches = count_pattern_matches(text, ENGAGEMENT_PATTERNS)
    
    if matches >= 2:
        return StatementType.ENGAGEMENT_BAIT, 0.97
    if matches == 1:
        return StatementType.ENGAGEMENT_BAIT, 0.92
        
    return None, 0.0

def classify_statistical(text: str) -> Tuple[Optional[StatementType], float]:
    """Detect statistical claims"""
    matches = count_pattern_matches(text, STATISTICAL_PATTERNS)
    
    if matches >= 3:
        return StatementType.STATISTICAL_CLAIM, 0.95
    if matches >= 2:
        return StatementType.STATISTICAL_CLAIM, 0.90
    if matches == 1:
        return StatementType.STATISTICAL_CLAIM, 0.82
        
    return None, 0.0

def classify_causal(text: str) -> Tuple[Optional[StatementType], float]:
    """Detect causal claims"""
    matches = count_pattern_matches(text, CAUSAL_PATTERNS)
    
    if matches >= 2:
        return StatementType.CAUSAL_CLAIM, 0.93
    if matches == 1:
        return StatementType.CAUSAL_CLAIM, 0.87
        
    return None, 0.0

def classify_logical_argument(text: str) -> Tuple[Optional[StatementType], float]:
    """Detect logical arguments"""
    matches = count_pattern_matches(text, LOGICAL_PATTERNS)
    
    if matches >= 2:
        return StatementType.LOGICAL_ARGUMENT, 0.92
    if matches == 1:
        return StatementType.LOGICAL_ARGUMENT, 0.85
        
    return None, 0.0

def classify_by_domain(text: str) -> Tuple[Optional[StatementType], float]:
    """Classify by domain-specific patterns"""
    health_matches = count_pattern_matches(text, HEALTH_PATTERNS)
    financial_matches = count_pattern_matches(text, FINANCIAL_PATTERNS)
    policy_matches = count_pattern_matches(text, POLICY_PATTERNS)
    
    if health_matches >= 2:
        return StatementType.HEALTH_CLAIM, 0.88
    if financial_matches >= 2:
        return StatementType.FINANCIAL_CLAIM, 0.88
    if policy_matches >= 2:
        return StatementType.POLICY_CLAIM, 0.88
        
    return None, 0.0

# ==============================================================================
# MAIN CLASSIFICATION FUNCTION
# ==============================================================================

def classify_statement_deterministic(text: str) -> Tuple[StatementType, float, bool]:
    """
    Classify a statement using only deterministic rules.
    
    Returns:
        (statement_type, confidence, needs_llm)
        
    Confidence thresholds:
        >= 0.90: Use deterministic (no LLM)
        0.65-0.89: Use embedding similarity
        < 0.65: Use LLM
    """
    if not text or len(text.strip()) < 5:
        return StatementType.FILLER, 1.0, False
    
    # Try all classifiers in priority order
    classifiers = [
        classify_engagement_bait,  # Highest priority - most obvious
        classify_promotional,
        classify_feeling,
        classify_opinion,
        classify_prediction,
        classify_causal,
        classify_statistical,
        classify_logical_argument,
        classify_by_domain,
    ]
    
    best_type = None
    best_confidence = 0.0
    
    for classifier in classifiers:
        stmt_type, confidence = classifier(text)
        if stmt_type and confidence > best_confidence:
            best_type = stmt_type
            best_confidence = confidence
    
    # Decision logic
    if best_confidence >= 0.90:
        # High confidence: use deterministic classification
        return best_type, best_confidence, False
    elif best_confidence >= 0.65:
        # Medium confidence: suggest embedding similarity check
        return best_type, best_confidence, False
    else:
        # Low confidence: needs LLM
        return StatementType.UNCLASSIFIABLE, best_confidence, True

def classify_batch_deterministic(statements: list[str]) -> list[dict]:
    """
    Classify multiple statements efficiently.
    
    Returns list of:
        {
            "text": str,
            "type": StatementType,
            "confidence": float,
            "needs_llm": bool
        }
    """
    results = []
    for text in statements:
        stmt_type, confidence, needs_llm = classify_statement_deterministic(text)
        results.append({
            "text": text,
            "type": stmt_type,
            "confidence": confidence,
            "needs_llm": needs_llm
        })
    return results

def get_classification_stats(results: list[dict]) -> dict:
    """Get statistics on classification batch"""
    total = len(results)
    high_confidence = sum(1 for r in results if r["confidence"] >= 0.90)
    medium_confidence = sum(1 for r in results if 0.65 <= r["confidence"] < 0.90)
    needs_llm = sum(1 for r in results if r["needs_llm"])
    
    return {
        "total_statements": total,
        "high_confidence_deterministic": high_confidence,
        "medium_confidence_embedding": medium_confidence,
        "low_confidence_llm_needed": needs_llm,
        "llm_avoidance_rate": (total - needs_llm) / total if total > 0 else 0.0
    }
