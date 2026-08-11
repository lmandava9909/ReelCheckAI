from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field, field_validator

RiskDomain = Literal['Policy','Finance','Health','Legal','Safety','Science','Technology','Consumer','General']
VerdictLabel = Literal['Supported','Mostly Supported','Partially Supported','Mixed Evidence','Unsupported','Insufficient Evidence','Misleading Framing','Needs Human Review']
Relationship = Literal['DIRECT_SUPPORT','PARTIAL_SUPPORT','RELATED_CONTEXT','CONTRADICTION','WEAK_EVIDENCE','NOISE_SOURCE']

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

class Claim(BaseModel):
    claim_id: str
    text: str = Field(min_length=10, max_length=700)
    claim_type: str = 'general_factual_claim'
    risk_domain: RiskDomain = 'General'
    priority: Literal['high','medium','low'] = 'medium'
    source_excerpt: str = ''
    quality_score: float = Field(default=.7, ge=0, le=1)
    needs_review: bool = False
    review_flags: list[str] = Field(default_factory=list)
    @field_validator('text')
    @classmethod
    def substantive(cls, v):
        if v.lower().strip() in {'check this out','follow for more','link in bio'}:
            raise ValueError('Non-factual promotional text')
        return v.strip()

class Evidence(BaseModel):
    evidence_id: str
    claim_id: str
    title: str
    url: str
    domain: str
    snippet: str = ''
    relationship: Relationship = 'RELATED_CONTEXT'
    authority_tier: Literal['HIGH','MEDIUM','LOW','UNKNOWN'] = 'UNKNOWN'
    search_score: float = Field(default=0, ge=0, le=1)
    quality_score: float = Field(default=0, ge=0, le=1)
    published_date: str | None = None

class ClaimVerdict(BaseModel):
    claim_id: str
    claim_text: str
    verdict: VerdictLabel
    confidence: float = Field(ge=0, le=1)
    risk_score: int = Field(ge=0, le=100)
    evidence_quality_score: float = Field(ge=0, le=1)
    supporting_evidence_count: int = 0
    contradicting_evidence_count: int = 0
    reasoning: str
    evidence_gap: str = ''
    user_takeaway: str
    evidence: list[Evidence] = Field(default_factory=list)

class ProcessingMetrics(BaseModel):
    total_duration_sec: float = 0
    extraction_duration_sec: float = 0
    transcription_duration_sec: float = 0
    claim_duration_sec: float = 0
    evidence_duration_sec: float = 0
    stages_completed: list[str] = Field(default_factory=list)
    stages_failed: list[str] = Field(default_factory=list)
    fallbacks_used: list[str] = Field(default_factory=list)
    llm_calls: int = 0
    search_api_calls: int = 0
    processor_version: str = '3.0.0'
    analyzed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class AnalysisResult(BaseModel):
    analysis_id: str
    url: str
    shortcode: str
    post_type: str
    caption: str = ''
    asset_count: int = 0
    visual_text: str = ''
    transcript: TranscriptAudit = Field(default_factory=TranscriptAudit)
    category: str = 'General'
    viewpoints: list[str] = Field(default_factory=list)
    core_claim: str = ''
    claims: list[Claim] = Field(default_factory=list)
    verdicts: list[ClaimVerdict] = Field(default_factory=list)
    overall_verdict: VerdictLabel = 'Insufficient Evidence'
    reliability_score: int = Field(default=0, ge=0, le=100)
    risk_score: int = Field(default=50, ge=0, le=100)
    content_value_score: int = Field(default=0, ge=0, le=100)
    what_matters: list[str] = Field(default_factory=list)
    cautions: list[str] = Field(default_factory=list)
    promotional_noise: list[str] = Field(default_factory=list)
    recommended_action: str = ''
    metrics: ProcessingMetrics = Field(default_factory=ProcessingMetrics)
