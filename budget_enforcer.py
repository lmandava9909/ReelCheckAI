"""API Budget Enforcement System

Hard limits on API calls per analysis mode:
- FAST: 2 LLM calls, 3 search calls, 9 evidence sources
- STANDARD: 4 LLM calls, 8 search calls, 20 evidence sources
- DEEP: 8 LLM calls, 16 search calls, 40 evidence sources

Never exceed these limits. Early exit if budget exhausted.
Track all API usage. Batch calls when possible.
"""

from typing import Optional, Tuple
from models_v2 import AnalysisMode, APIBudgetUsage, API_BUDGETS

class BudgetExhaustedException(Exception):
    """Raised when API budget is exceeded"""
    pass

class BudgetEnforcer:
    """Enforce API call budgets and track usage"""
    
    def __init__(self, mode: AnalysisMode):
        self.mode = mode
        self.usage = APIBudgetUsage(mode=mode)
        self.budget = API_BUDGETS[mode]
        
    def can_call_llm(self, calls_needed: int = 1) -> Tuple[bool, str]:
        """Check if LLM calls are available"""
        available = self.usage.llm_budget_remaining
        if calls_needed > available:
            return False, f"LLM budget exhausted. Used {self.usage.llm_calls_used}/{self.budget['llm_calls']}, need {calls_needed} more"
        return True, ""
    
    def can_call_search(self, calls_needed: int = 1) -> Tuple[bool, str]:
        """Check if search calls are available"""
        available = self.usage.search_budget_remaining
        if calls_needed > available:
            return False, f"Search budget exhausted. Used {self.usage.search_calls_used}/{self.budget['search_calls']}, need {calls_needed} more"
        return True, ""
    
    def can_retrieve_evidence(self, sources_needed: int) -> Tuple[bool, str]:
        """Check if evidence retrieval is within budget"""
        current = self.usage.evidence_sources_retrieved
        max_allowed = self.budget['evidence_sources']
        if current + sources_needed > max_allowed:
            return False, f"Evidence budget exceeded. Have {current}/{max_allowed}, need {sources_needed} more"
        return True, ""
    
    def record_llm_call(self, calls: int = 1):
        """Record LLM API usage"""
        self.usage.llm_calls_used += calls
        if self.usage.llm_calls_used > self.budget['llm_calls']:
            raise BudgetExhaustedException(
                f"LLM budget exceeded: {self.usage.llm_calls_used}/{self.budget['llm_calls']}"
            )
    
    def record_search_call(self, calls: int = 1):
        """Record search API usage"""
        self.usage.search_calls_used += calls
        if self.usage.search_calls_used > self.budget['search_calls']:
            raise BudgetExhaustedException(
                f"Search budget exceeded: {self.usage.search_calls_used}/{self.budget['search_calls']}"
            )
    
    def record_vision_call(self, calls: int = 1):
        """Record vision API usage"""
        self.usage.vision_calls_used += calls
    
    def record_evidence_retrieval(self, sources: int):
        """Record evidence source retrieval"""
        self.usage.evidence_sources_retrieved += sources
        if self.usage.evidence_sources_retrieved > self.budget['evidence_sources']:
            raise BudgetExhaustedException(
                f"Evidence budget exceeded: {self.usage.evidence_sources_retrieved}/{self.budget['evidence_sources']}"
            )
    
    def record_embedding_batch(self, batches: int = 1):
        """Record embedding API usage"""
        self.usage.embedding_batches_used += batches
    
    def get_remaining_budget(self) -> dict:
        """Get all remaining budgets"""
        return {
            "llm_calls": self.usage.llm_budget_remaining,
            "search_calls": self.usage.search_budget_remaining,
            "evidence_sources": self.budget['evidence_sources'] - self.usage.evidence_sources_retrieved,
            "vision_calls_used": self.usage.vision_calls_used,
            "embedding_batches_used": self.usage.embedding_batches_used,
        }
    
    def get_usage_report(self) -> str:
        """Get human-readable usage report"""
        llm_pct = (self.usage.llm_calls_used / self.budget['llm_calls'] * 100) if self.budget['llm_calls'] > 0 else 0
        search_pct = (self.usage.search_calls_used / self.budget['search_calls'] * 100) if self.budget['search_calls'] > 0 else 0
        evidence_pct = (self.usage.evidence_sources_retrieved / self.budget['evidence_sources'] * 100) if self.budget['evidence_sources'] > 0 else 0
        
        return f"""API Budget Usage ({self.mode.value} mode):
  LLM calls: {self.usage.llm_calls_used}/{self.budget['llm_calls']} ({llm_pct:.0f}%)
  Search calls: {self.usage.search_calls_used}/{self.budget['search_calls']} ({search_pct:.0f}%)
  Evidence sources: {self.usage.evidence_sources_retrieved}/{self.budget['evidence_sources']} ({evidence_pct:.0f}%)
  Vision calls: {self.usage.vision_calls_used}
  Embedding batches: {self.usage.embedding_batches_used}"""
    
    def should_early_exit(self, statements_remaining: int) -> bool:
        """
        Decide if we should early exit based on remaining budget.
        
        Early exit if:
        - LLM budget < 1 AND statements remain that might need LLM
        - Search budget < 1 AND factual claims remain
        """
        if self.usage.llm_budget_remaining < 1 and statements_remaining > 0:
            return True
        if self.usage.search_budget_remaining < 1 and statements_remaining > 0:
            return True
        return False

# ==============================================================================
# BUDGET-AWARE BATCHING
# ==============================================================================

def batch_llm_calls(items: list, budget: BudgetEnforcer, max_per_batch: int = 10) -> list[list]:
    """
    Batch items for LLM processing while respecting budget.
    
    Returns list of batches, each respecting the remaining LLM budget.
    """
    remaining_calls = budget.usage.llm_budget_remaining
    if remaining_calls == 0:
        return []
    
    # Calculate how many batches we can afford
    num_batches = min(remaining_calls, (len(items) + max_per_batch - 1) // max_per_batch)
    
    batches = []
    items_per_batch = len(items) // num_batches if num_batches > 0 else len(items)
    
    for i in range(0, len(items), items_per_batch):
        batch = items[i:i + items_per_batch]
        if batch:
            batches.append(batch)
            if len(batches) >= remaining_calls:
                break
    
    return batches

def prioritize_evidence_search(statements: list, budget: BudgetEnforcer) -> list:
    """
    Prioritize statements for evidence search based on:
    1. Risk domain (Health/Financial first)
    2. Statement type (Factual claims prioritized)
    3. Quality score
    
    Returns reordered list fitting within search budget.
    """
    # Categorize by priority
    high_priority = []
    medium_priority = []
    low_priority = []
    
    for stmt in statements:
        # Check if it's a health/financial claim (high risk)
        is_high_risk = any(
            domain in (stmt.risk_domain or "").lower()
            for domain in ["health", "medical", "financial", "investment"]
        )
        
        if is_high_risk:
            high_priority.append(stmt)
        elif stmt.requires_evidence_search and stmt.externally_verifiable:
            medium_priority.append(stmt)
        else:
            low_priority.append(stmt)
    
    # Sort each tier by quality
    high_priority.sort(key=lambda s: s.quality_score, reverse=True)
    medium_priority.sort(key=lambda s: s.quality_score, reverse=True)
    low_priority.sort(key=lambda s: s.quality_score, reverse=True)
    
    # Combine and limit to search budget
    prioritized = high_priority + medium_priority + low_priority
    search_limit = budget.usage.search_budget_remaining
    
    return prioritized[:search_limit]

# ==============================================================================
# MODE-SPECIFIC STRATEGY
# ==============================================================================

def get_mode_strategy(mode: AnalysisMode) -> dict:
    """
    Get processing strategy for each mode.
    
    FAST: Skip all non-essential processing
    STANDARD: Balanced approach
    DEEP: Full analysis
    """
    strategies = {
        AnalysisMode.FAST: {
            "skip_opinion_premise_extraction": True,
            "skip_logical_analysis": True,
            "skip_prediction_evaluation": True,
            "max_claims_to_process": 3,
            "evidence_per_claim": 3,
            "use_cache_aggressively": True,
            "batch_size": 5,
        },
        AnalysisMode.STANDARD: {
            "skip_opinion_premise_extraction": False,
            "skip_logical_analysis": False,
            "skip_prediction_evaluation": False,
            "max_claims_to_process": 5,
            "evidence_per_claim": 4,
            "use_cache_aggressively": True,
            "batch_size": 10,
        },
        AnalysisMode.DEEP: {
            "skip_opinion_premise_extraction": False,
            "skip_logical_analysis": False,
            "skip_prediction_evaluation": False,
            "max_claims_to_process": 10,
            "evidence_per_claim": 4,
            "use_cache_aggressively": False,
            "batch_size": 20,
        }
    }
    return strategies[mode]
