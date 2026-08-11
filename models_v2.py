"""ReelCheckAI V2 Data Models

Comprehensive schema for deterministic-first multimodal trust intelligence.
Designed to minimize API calls while maximizing analytical depth.

Core principles:
- Track provenance of every piece of content
- Quality score everything
- Statement taxonomy (30+ types)
- API budget enforcement
- Cache everything reusable
- Never fabricate unavailable data
"""

from datetime import datetime, timezone, timedelta
from typing import Literal, Optional
from pydantic import BaseModel, Field
from enum import Enum
import hashlib

# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================

PROCESSOR_VERSION = "4.0.0"
POLICY_VERSION = "1.0.0"

# Post type classification
class PostType(str, Enum):
    REEL = "REEL"
    VIDEO_POST = "VIDEO_POST"
    IMAGE_POST = "IMAGE_POST"
    CAROUSEL_IMAGE_ONLY = "CAROUSEL_IMAGE_ONLY"
    CAROUSEL_VIDEO_ONLY = "CAROUSEL_VIDEO_ONLY"
    CAROUSEL_MIXED = "CAROUSEL_MIXED"
    CAPTION_ONLY = "CAPTION_ONLY"
    STORY_EXPORT = "STORY_EXPORT"
    SCREENSHOT_UPLOAD = "SCREENSHOT_UPLOAD"
    VIDEO_UPLOAD = "VIDEO_UPLOAD"
    AUDIO_UPLOAD = "AUDIO_UPLOAD"
    UNAVAILABLE = "UNAVAILABLE"
    UNKNOWN = "UNKNOWN"

# Processing modes
class AnalysisMode(str, Enum):
    FAST = "FAST"  # Low-latency consumer analysis
    STANDARD = "STANDARD"  # Default balanced mode
    DEEP = "DEEP"  # High-impact detailed analysis

# API budget limits per mode
API_BUDGETS = {
    AnalysisMode.FAST: {
        "llm_calls": 2,
        "search_calls": 3,
        "evidence_sources": 9,
        "vision_calls": 0,
        "max_claims": 3
    },
    AnalysisMode.STANDARD: {
        "llm_calls": 4,
        "search_calls": 8,
        "evidence_sources": 20,
        "vision_calls": 3,
        "max_claims": 5
    },
    AnalysisMode.DEEP: {
        "llm_calls": 8,
        "search_calls": 16,
        "evidence_sources": 40,
        "vision_calls": 8,
        "max_claims": 10
    }
}

# Evidence authority registry - domain specific
AUTHORITY_REGISTRY = {
    "Policy": [
        "whitehouse.gov", "congress.gov", "federalregister.gov",
        "commerce.gov", "state.gov", "treasury.gov"
    ],
    "Finance": [
        "sec.gov", "federalreserve.gov", "fdic.gov",
        "bloomberg.com", "ft.com", "wsj.com"
    ],
    "Health": [
        "fda.gov", "cdc.gov", "nih.gov", "who.int",
        "nejm.org", "thelancet.com", "bmj.com"
    ],
    "Science": [
        "nature.com", "science.org", "pnas.org",
        "sciencedirect.com", "arxiv.org"
    ],
    "Legal": [
        "supremecourt.gov", "uscourts.gov", "justia.com",
        "law.cornell.edu"
    ]
}

# Search cache freshness by domain (in hours)
FRESHNESS_PERIODS = {
    "breaking_news": 6,
    "policy": 24,
    "finance": 6,
    "health": 168,  # 7 days
    "science": 168,
    "historical": 720,  # 30 days
    "definitions": 2160  # 90 days
}

# ==============================================================================
# STATEMENT TAXONOMY (30+ types)
# ==============================================================================

class StatementType(str, Enum):
    # Factual claims
    FACTUAL_CLAIM = "FACTUAL_CLAIM"
    STATISTICAL_CLAIM = "STATISTICAL_CLAIM"
    CAUSAL_CLAIM = "CAUSAL_CLAIM"
    COMPARATIVE_CLAIM = "COMPARATIVE_CLAIM"
    ATTRIBUTION_CLAIM = "ATTRIBUTION_CLAIM"
    QUOTE_CLAIM = "QUOTE_CLAIM"
    EVENT_CLAIM = "EVENT_CLAIM"
    POLICY_CLAIM = "POLICY_CLAIM"
    LEGAL_CLAIM = "LEGAL_CLAIM"
    SCIENTIFIC_CLAIM = "SCIENTIFIC_CLAIM"
    HEALTH_CLAIM = "HEALTH_CLAIM"
    FINANCIAL_CLAIM = "FINANCIAL_CLAIM"
    PRODUCT_CLAIM = "PRODUCT_CLAIM"
    HISTORICAL_CLAIM = "HISTORICAL_CLAIM"
    
    # Non-factual content
    PREDICTION = "PREDICTION"
    RECOMMENDATION = "RECOMMENDATION"
    VIEWPOINT = "VIEWPOINT"
    OPINION = "OPINION"
    FEELING = "FEELING"
    PERSONAL_EXPERIENCE = "PERSONAL_EXPERIENCE"
    MORAL_JUDGMENT = "MORAL_JUDGMENT"
    
    # Logical/rhetorical
    LOGICAL_ARGUMENT = "LOGICAL_ARGUMENT"
    ANALOGY = "ANALOGY"
    RHETORICAL_QUESTION = "RHETORICAL_QUESTION"
    
    # Promotional
    PROMOTIONAL_STATEMENT = "PROMOTIONAL_STATEMENT"
    ENGAGEMENT_BAIT = "ENGAGEMENT_BAIT"
    HYPE = "HYPE"
    
    # Other
    SATIRE = "SATIRE"
    HUMOR = "HUMOR"
    INSTRUCTION = "INSTRUCTION"
    WARNING = "WARNING"
    DEFINITION = "DEFINITION"
    CONTEXT = "CONTEXT"
    FILLER = "FILLER"
    UNCLASSIFIABLE = "UNCLASSIFIABLE"

# ==============================================================================
# CONTENT PROVENANCE
# ==============================================================================

class SourceType(str, Enum):
    CAPTION = "CAPTION"
    METADATA_TITLE = "METADATA_TITLE"
    METADATA_DESCRIPTION = "METADATA_DESCRIPTION"
    TRANSCRIPT_SEGMENT = "TRANSCRIPT_SEGMENT"
    IMAGE_OCR = "IMAGE_OCR"
    VIDEO_FRAME_OCR = "VIDEO_FRAME_OCR"
    VISUAL_SUMMARY = "VISUAL_SUMMARY"
    CHART_EXTRACTION = "CHART_EXTRACTION"
    CAROUSEL_SUMMARY = "CAROUSEL_SUMMARY"
    MANUAL_CAPTION = "MANUAL_CAPTION"
    MANUAL_TRANSCRIPT = "MANUAL_TRANSCRIPT"

class ContentProvenance(BaseModel):
    """Track origin of every piece of text"""
    content_id: str = Field(default_factory=lambda: f"content-{int(datetime.now(timezone.utc).timestamp() * 1000)}")
    source_type: SourceType
    asset_id: Optional[str] = None
    raw_text: str
    normalized_text: str
    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None
    quality_score: float = Field(default=0.0, ge=0, le=1)
    corrections: list = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

# ==============================================================================
# ASSET MODEL (for carousel/multi-media posts)
# ==============================================================================

class AssetType(str, Enum):
    IMAGE = "IMAGE"
    VIDEO = "VIDEO"
    AUDIO = "AUDIO"
    DOCUMENT = "DOCUMENT"

class Asset(BaseModel):
    """Individual media item in a post"""
    asset_id: str = Field(default_factory=lambda: f"asset-{int(datetime.now(timezone.utc).timestamp() * 1000)}")
    asset_index: int  # Position in carousel
    asset_type: AssetType
    local_path: Optional[str] = None
    raw_ocr: str = ""
    clean_ocr: str = ""
    visual_summary: str = ""
    transcript: Optional[str] = None
    quality_score: float = Field(default=0.0, ge=0, le=1)
    extracted_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

# ==============================================================================
# SUBMISSION TRACKING
# ==============================================================================

class SubmissionStatus(str, Enum):
    SUBMITTED = "SUBMITTED"
    EXTRACTING = "EXTRACTING"
    CLASSIFYING = "CLASSIFYING"
    SEARCHING = "SEARCHING"
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"

class Submission(BaseModel):
    """Track analysis session with idempotency"""
    submission_id: str = Field(default_factory=lambda: f"sub-{int(datetime.now(timezone.utc).timestamp() * 1000)}")
    original_url: str
    normalized_url: str
    shortcode: str
    requested_mode: AnalysisMode = AnalysisMode.STANDARD
    submitted_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    user_session_id: Optional[str] = None
    processor_version: str = PROCESSOR_VERSION
    policy_version: str = POLICY_VERSION
    status: SubmissionStatus = SubmissionStatus.SUBMITTED
    
    @property
    def analysis_key(self) -> str:
        """Generate idempotency key"""
        key_str = f"{self.normalized_url}{self.processor_version}{self.policy_version}{self.requested_mode.value}"
        return hashlib.sha256(key_str.encode()).hexdigest()

# ==============================================================================
# TRANSCRIPT MODELS (existing, kept for compatibility)
# ==============================================================================

class Correction(BaseModel):
    original_phrase: str
    corrected_phrase: str
    method: str
    confidence: float = Field(ge=0, le=1)
    rationale: str

class TranscriptAudit(BaseModel):
    raw_transcript: str = ''
    corrected_transcript: str = ''
    corrections: list[Correction] = Field(default_factory=list)
    review_flags: list[str] = Field(default_factory=list)
    quality_score: float = Field(default=0, ge=0, le=1)
    word_count: int = 0

# ==============================================================================
# STATEMENT MODEL (replaces simple Claim)
# ==============================================================================

class Statement(BaseModel):
    """Complete statement with 30+ type taxonomy"""
    statement_id: str = Field(default_factory=lambda: f"stmt-{int(datetime.now(timezone.utc).timestamp() * 1000)}")
    text: str
    primary_type: StatementType
    secondary_types: list[StatementType] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    classification_confidence: float = Field(default=0.0, ge=0, le=1)
    externally_verifiable: bool
    requires_logical_analysis: bool
    requires_evidence_search: bool
    risk_domain: Optional[str] = None
    quality_score: float = Field(default=0.0, ge=0, le=1)
    
    # Claim refinement fields
    canonical_form: Optional[str] = None
    actor: Optional[str] = None
    action: Optional[str] = None
    object: Optional[str] = None
    temporal_context: Optional[str] = None
    jurisdiction: Optional[str] = None
    
    # Routing metadata
    workflow_assigned: Optional[str] = None
    search_queries: list[str] = Field(default_factory=list)

# ==============================================================================
# EVIDENCE MODEL (expanded)
# ==============================================================================

class Relationship(str, Enum):
    DIRECT_SUPPORT = "DIRECT_SUPPORT"
    PARTIAL_SUPPORT = "PARTIAL_SUPPORT"
    RELATED_CONTEXT = "RELATED_CONTEXT"
    CONTRADICTION = "CONTRADICTION"
    WEAK_EVIDENCE = "WEAK_EVIDENCE"
    NOISE_SOURCE = "NOISE_SOURCE"

class Evidence(BaseModel):
    evidence_id: str = Field(default_factory=lambda: f"ev-{int(datetime.now(timezone.utc).timestamp() * 1000)}")
    statement_id: str
    title: str
    url: str
    domain: str
    snippet: str = ''
    relationship: Relationship = Relationship.RELATED_CONTEXT
    authority_tier: Literal['HIGH','MEDIUM','LOW','UNKNOWN'] = 'UNKNOWN'
    
    # Expanded scoring dimensions
    search_score: float = Field(default=0, ge=0, le=1)
    directness_score: float = Field(default=0, ge=0, le=1)
    entity_alignment_score: float = Field(default=0, ge=0, le=1)
    temporal_alignment_score: float = Field(default=0, ge=0, le=1)
    jurisdiction_alignment_score: float = Field(default=0, ge=0, le=1)
    independence_score: float = Field(default=0, ge=0, le=1)
    
    # Final deterministic quality
    quality_score: float = Field(default=0, ge=0, le=1)
    published_date: Optional[str] = None
    retrieved_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

# ==============================================================================
# VERDICT MODEL (expanded for workflow outputs)
# ==============================================================================

class VerdictLabel(str, Enum):
    # Factual verdicts
    SUPPORTED = "Supported"
    MOSTLY_SUPPORTED = "Mostly Supported"
    PARTIALLY_SUPPORTED = "Partially Supported"
    MIXED_EVIDENCE = "Mixed Evidence"
    UNSUPPORTED = "Unsupported"
    INSUFFICIENT_EVIDENCE = "Insufficient Evidence"
    MISLEADING_FRAMING = "Misleading Framing"
    
    # Non-factual outputs
    OPINION = "Opinion"
    PERSONAL_EXPERIENCE = "Personal Experience"
    PREDICTION = "Prediction"
    RECOMMENDATION = "Recommendation"
    PROMOTIONAL = "Promotional"
    SATIRE = "Satire/Humor"
    LOGICAL_ISSUE = "Logical Issue"
    
    # Process outputs
    NEEDS_REVIEW = "Needs Human Review"
    NEEDS_CONTEXT = "Needs Context"

class StatementVerdict(BaseModel):
    """Verdict for any statement type (not just factual claims)"""
    statement_id: str
    statement_text: str
    statement_type: StatementType
    verdict: VerdictLabel
    confidence: float = Field(ge=0, le=1)
    
    # Workflow-specific outputs
    factual_premises: list[str] = Field(default_factory=list)  # For opinions
    assumptions: list[str] = Field(default_factory=list)  # For predictions/logic
    risk_factors: list[str] = Field(default_factory=list)  # For recommendations
    commercial_interests: list[str] = Field(default_factory=list)  # For promotions
    logical_issues: list[str] = Field(default_factory=list)  # For arguments
    
    # Evidence (only for verifiable claims)
    evidence: list[Evidence] = Field(default_factory=list)
    supporting_evidence_count: int = 0
    contradicting_evidence_count: int = 0
    evidence_quality_score: float = Field(default=0, ge=0, le=1)
    
    # Explanation
    reasoning: str
    evidence_gap: str = ''
    user_takeaway: str
    
    # Scores
    risk_score: int = Field(default=50, ge=0, le=100)

# ==============================================================================
# API BUDGET TRACKING
# ==============================================================================

class APIBudgetUsage(BaseModel):
    """Track and enforce API call limits"""
    mode: AnalysisMode
    llm_calls_used: int = 0
    search_calls_used: int = 0
    vision_calls_used: int = 0
    evidence_sources_retrieved: int = 0
    embedding_batches_used: int = 0
    
    @property
    def llm_budget_remaining(self) -> int:
        return API_BUDGETS[self.mode]["llm_calls"] - self.llm_calls_used
    
    @property
    def search_budget_remaining(self) -> int:
        return API_BUDGETS[self.mode]["search_calls"] - self.search_calls_used
    
    @property
    def can_call_llm(self) -> bool:
        return self.llm_calls_used < API_BUDGETS[self.mode]["llm_calls"]
    
    @property
    def can_call_search(self) -> bool:
        return self.search_calls_used < API_BUDGETS[self.mode]["search_calls"]

# ==============================================================================
# SEARCH CACHE
# ==============================================================================

class SearchCacheEntry(BaseModel):
    """Cache search results with freshness tracking"""
    query_hash: str
    normalized_claim_hash: str
    domain: str  # For freshness period lookup
    result_json: str
    retrieved_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at: str
    
    @property
    def is_fresh(self) -> bool:
        expires = datetime.fromisoformat(self.expires_at.replace('Z', '+00:00'))
        return datetime.now(timezone.utc) < expires

# ==============================================================================
# PROCESSING METRICS (expanded)
# ==============================================================================

class ProcessingMetrics(BaseModel):
    total_duration_sec: float = 0
    extraction_duration_sec: float = 0
    transcription_duration_sec: float = 0
    classification_duration_sec: float = 0
    evidence_duration_sec: float = 0
    
    stages_completed: list[str] = Field(default_factory=list)
    stages_failed: list[str] = Field(default_factory=list)
    fallbacks_used: list[str] = Field(default_factory=list)
    
    # API usage
    api_budget: Optional[APIBudgetUsage] = None
    
    processor_version: str = PROCESSOR_VERSION
    policy_version: str = POLICY_VERSION
    analyzed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

# ==============================================================================
# FINAL ANALYSIS RESULT (6-level output)
# ==============================================================================

class AnalysisResult(BaseModel):
    """Complete multimodal trust intelligence output"""
    # Identification
    analysis_id: str = Field(default_factory=lambda: f"analysis-{int(datetime.now(timezone.utc).timestamp() * 1000)}")
    submission_id: str
    url: str
    shortcode: str
    
    # Post metadata
    post_type: PostType
    asset_count: int = 0
    assets: list[Asset] = Field(default_factory=list)
    
    # Extracted content
    caption: str = ''
    visual_text: str = ''
    transcript: TranscriptAudit = Field(default_factory=TranscriptAudit)
    content_chunks: list[ContentProvenance] = Field(default_factory=list)
    
    # Classification
    statements: list[Statement] = Field(default_factory=list)
    verdicts: list[StatementVerdict] = Field(default_factory=list)
    
    # === LEVEL 1: Immediate answer ===
    overall_verdict: VerdictLabel = VerdictLabel.INSUFFICIENT_EVIDENCE
    one_sentence_takeaway: str = ""
    
    # === LEVEL 2: What matters ===
    supported_information: list[str] = Field(default_factory=list)
    important_context: list[str] = Field(default_factory=list)
    
    # === LEVEL 3: What needs caution ===
    unverified_claims: list[str] = Field(default_factory=list)
    predictions_noted: list[str] = Field(default_factory=list)
    promotional_framing: list[str] = Field(default_factory=list)
    missing_context: list[str] = Field(default_factory=list)
    
    # === LEVEL 4: Statement breakdown ===
    facts_summary: str = ""
    opinions_summary: str = ""
    feelings_summary: str = ""
    predictions_summary: str = ""
    recommendations_summary: str = ""
    
    # === LEVEL 5: Evidence ===
    all_evidence: list[Evidence] = Field(default_factory=list)
    evidence_quality_summary: str = ""
    contradictions_found: list[str] = Field(default_factory=list)
    
    # === LEVEL 6: Trust Audit ===
    extraction_quality_report: str = ""
    corrections_applied_count: int = 0
    components_unavailable: list[str] = Field(default_factory=list)
    fallbacks_used: list[str] = Field(default_factory=list)
    uncertainty_areas: list[str] = Field(default_factory=list)
    
    # Scoring
    reliability_score: int = Field(default=0, ge=0, le=100)
    risk_score: int = Field(default=50, ge=0, le=100)
    content_value_score: int = Field(default=0, ge=0, le=100)
    
    # Legacy fields (backward compatibility)
    core_claim: str = ''
    category: str = 'General'
    viewpoints: list[str] = Field(default_factory=list)
    what_matters: list[str] = Field(default_factory=list)
    cautions: list[str] = Field(default_factory=list)
    promotional_noise: list[str] = Field(default_factory=list)
    recommended_action: str = ''
    
    # Metrics
    metrics: ProcessingMetrics = Field(default_factory=ProcessingMetrics)
