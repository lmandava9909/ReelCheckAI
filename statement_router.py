"""Statement Routing System

Routes classified statements to specialized analysis workflows.
Each workflow has different evidence requirements and output contracts.

9 Workflow Types:
1. FACTUAL - Evidence search and verification
2. OPINION - Premise extraction and implicit claims
3. FEELING - Generalization check and overgeneralization detection
4. PREDICTION - Scenario evaluation and assumption identification
5. RECOMMENDATION - Risk routing (Health/Financial get extra scrutiny)
6. PROMOTIONAL - Commercial framing analysis
7. LOGICAL - Argument parsing and fallacy detection
8. SATIRE - Literal interpretation avoidance
9. INSTRUCTION - Risk classification

Each workflow returns a specialized verdict structure.
"""

from typing import Optional
from enum import Enum
from models_v2 import StatementType, Statement, VerdictLabel

class WorkflowType(str, Enum):
    FACTUAL = "FACTUAL"
    OPINION = "OPINION"
    FEELING = "FEELING"
    PREDICTION = "PREDICTION"
    RECOMMENDATION = "RECOMMENDATION"
    PROMOTIONAL = "PROMOTIONAL"
    LOGICAL = "LOGICAL"
    SATIRE = "SATIRE"
    INSTRUCTION = "INSTRUCTION"
    SKIP = "SKIP"  # For filler, context, etc.

# ==============================================================================
# ROUTING LOGIC
# ==============================================================================

def route_statement(statement: Statement) -> WorkflowType:
    """
    Route a classified statement to the appropriate analysis workflow.
    
    Returns the workflow type that should process this statement.
    """
    stmt_type = statement.primary_type
    
    # Factual claims - need evidence search
    if stmt_type in {
        StatementType.FACTUAL_CLAIM,
        StatementType.STATISTICAL_CLAIM,
        StatementType.CAUSAL_CLAIM,
        StatementType.COMPARATIVE_CLAIM,
        StatementType.ATTRIBUTION_CLAIM,
        StatementType.QUOTE_CLAIM,
        StatementType.EVENT_CLAIM,
        StatementType.POLICY_CLAIM,
        StatementType.LEGAL_CLAIM,
        StatementType.SCIENTIFIC_CLAIM,
        StatementType.HEALTH_CLAIM,
        StatementType.FINANCIAL_CLAIM,
        StatementType.PRODUCT_CLAIM,
        StatementType.HISTORICAL_CLAIM,
    }:
        return WorkflowType.FACTUAL
    
    # Opinions - extract factual premises
    if stmt_type == StatementType.OPINION or stmt_type == StatementType.VIEWPOINT:
        return WorkflowType.OPINION
    
    # Moral judgments (subtype of opinion)
    if stmt_type == StatementType.MORAL_JUDGMENT:
        return WorkflowType.OPINION
    
    # Feelings - check for overgeneralization
    if stmt_type == StatementType.FEELING or stmt_type == StatementType.PERSONAL_EXPERIENCE:
        return WorkflowType.FEELING
    
    # Predictions - evaluate assumptions
    if stmt_type == StatementType.PREDICTION:
        return WorkflowType.PREDICTION
    
    # Recommendations - risk routing
    if stmt_type == StatementType.RECOMMENDATION:
        return WorkflowType.RECOMMENDATION
    
    # Promotional content - framing analysis
    if stmt_type in {StatementType.PROMOTIONAL_STATEMENT, StatementType.ENGAGEMENT_BAIT, StatementType.HYPE}:
        return WorkflowType.PROMOTIONAL
    
    # Logical arguments - argument parsing
    if stmt_type in {StatementType.LOGICAL_ARGUMENT, StatementType.ANALOGY}:
        return WorkflowType.LOGICAL
    
    # Satire - avoid literal interpretation
    if stmt_type in {StatementType.SATIRE, StatementType.HUMOR}:
        return WorkflowType.SATIRE
    
    # Instructions/warnings - risk classification
    if stmt_type in {StatementType.INSTRUCTION, StatementType.WARNING}:
        return WorkflowType.INSTRUCTION
    
    # Skip processing for filler, rhetorical questions, context
    if stmt_type in {
        StatementType.FILLER,
        StatementType.CONTEXT,
        StatementType.DEFINITION,
        StatementType.RHETORICAL_QUESTION,
        StatementType.UNCLASSIFIABLE
    }:
        return WorkflowType.SKIP
    
    # Default: factual workflow for unknown types
    return WorkflowType.FACTUAL

# ==============================================================================
# WORKFLOW REQUIREMENTS
# ==============================================================================

def get_workflow_requirements(workflow: WorkflowType) -> dict:
    """
    Get the requirements for each workflow type.
    
    Returns dict with:
        - needs_evidence_search: bool
        - needs_premise_extraction: bool
        - needs_risk_assessment: bool
        - needs_logical_analysis: bool
        - max_evidence_sources: int
        - min_evidence_quality: float
        - output_template: str
    """
    requirements = {
        WorkflowType.FACTUAL: {
            "needs_evidence_search": True,
            "needs_premise_extraction": False,
            "needs_risk_assessment": False,
            "needs_logical_analysis": False,
            "max_evidence_sources": 20,
            "min_evidence_quality": 0.7,
            "verdict_options": [
                VerdictLabel.SUPPORTED,
                VerdictLabel.MOSTLY_SUPPORTED,
                VerdictLabel.PARTIALLY_SUPPORTED,
                VerdictLabel.MIXED_EVIDENCE,
                VerdictLabel.UNSUPPORTED,
                VerdictLabel.INSUFFICIENT_EVIDENCE,
                VerdictLabel.MISLEADING_FRAMING,
            ]
        },
        WorkflowType.OPINION: {
            "needs_evidence_search": False,  # Unless premises need checking
            "needs_premise_extraction": True,
            "needs_risk_assessment": False,
            "needs_logical_analysis": True,
            "max_evidence_sources": 10,
            "min_evidence_quality": 0.6,
            "verdict_options": [VerdictLabel.OPINION]
        },
        WorkflowType.FEELING: {
            "needs_evidence_search": False,
            "needs_premise_extraction": False,
            "needs_risk_assessment": False,
            "needs_logical_analysis": False,
            "max_evidence_sources": 0,
            "min_evidence_quality": 0.0,
            "verdict_options": [VerdictLabel.PERSONAL_EXPERIENCE]
        },
        WorkflowType.PREDICTION: {
            "needs_evidence_search": True,  # For assumptions
            "needs_premise_extraction": True,
            "needs_risk_assessment": True,
            "needs_logical_analysis": True,
            "max_evidence_sources": 15,
            "min_evidence_quality": 0.65,
            "verdict_options": [VerdictLabel.PREDICTION]
        },
        WorkflowType.RECOMMENDATION: {
            "needs_evidence_search": True,
            "needs_premise_extraction": True,
            "needs_risk_assessment": True,
            "needs_logical_analysis": False,
            "max_evidence_sources": 20,  # Extra for health/financial
            "min_evidence_quality": 0.75,  # Higher bar for recommendations
            "verdict_options": [VerdictLabel.RECOMMENDATION]
        },
        WorkflowType.PROMOTIONAL: {
            "needs_evidence_search": False,
            "needs_premise_extraction": False,
            "needs_risk_assessment": False,
            "needs_logical_analysis": False,
            "max_evidence_sources": 5,  # Only for specific claims
            "min_evidence_quality": 0.6,
            "verdict_options": [VerdictLabel.PROMOTIONAL]
        },
        WorkflowType.LOGICAL: {
            "needs_evidence_search": True,  # For premises
            "needs_premise_extraction": True,
            "needs_risk_assessment": False,
            "needs_logical_analysis": True,
            "max_evidence_sources": 15,
            "min_evidence_quality": 0.7,
            "verdict_options": [VerdictLabel.LOGICAL_ISSUE]
        },
        WorkflowType.SATIRE: {
            "needs_evidence_search": False,
            "needs_premise_extraction": False,
            "needs_risk_assessment": False,
            "needs_logical_analysis": False,
            "max_evidence_sources": 0,
            "min_evidence_quality": 0.0,
            "verdict_options": [VerdictLabel.SATIRE]
        },
        WorkflowType.INSTRUCTION: {
            "needs_evidence_search": False,
            "needs_premise_extraction": False,
            "needs_risk_assessment": True,
            "needs_logical_analysis": False,
            "max_evidence_sources": 5,
            "min_evidence_quality": 0.6,
            "verdict_options": [VerdictLabel.NEEDS_REVIEW]
        },
        WorkflowType.SKIP: {
            "needs_evidence_search": False,
            "needs_premise_extraction": False,
            "needs_risk_assessment": False,
            "needs_logical_analysis": False,
            "max_evidence_sources": 0,
            "min_evidence_quality": 0.0,
            "verdict_options": []
        }
    }
    
    return requirements.get(workflow, requirements[WorkflowType.SKIP])

# ==============================================================================
# BATCH ROUTING
# ==============================================================================

def route_statements_batch(statements: list[Statement]) -> dict:
    """
    Route multiple statements and organize by workflow.
    
    Returns dict mapping workflow type to list of statements.
    """
    routing_map = {workflow: [] for workflow in WorkflowType}
    
    for statement in statements:
        workflow = route_statement(statement)
        routing_map[workflow].append(statement)
        
        # Update statement metadata
        statement.workflow_assigned = workflow.value
    
    return routing_map

def get_routing_stats(routing_map: dict) -> dict:
    """Get statistics on routing decisions"""
    total = sum(len(stmts) for stmts in routing_map.values())
    
    factual_count = len(routing_map.get(WorkflowType.FACTUAL, []))
    opinion_count = len(routing_map.get(WorkflowType.OPINION, []))
    prediction_count = len(routing_map.get(WorkflowType.PREDICTION, []))
    promotional_count = len(routing_map.get(WorkflowType.PROMOTIONAL, []))
    skip_count = len(routing_map.get(WorkflowType.SKIP, []))
    
    # Calculate evidence search percentage
    evidence_needed = factual_count + prediction_count + len(routing_map.get(WorkflowType.RECOMMENDATION, []))
    evidence_search_rate = evidence_needed / total if total > 0 else 0.0
    
    return {
        "total_statements": total,
        "factual": factual_count,
        "opinion": opinion_count,
        "prediction": prediction_count,
        "promotional": promotional_count,
        "skip": skip_count,
        "evidence_search_needed": evidence_needed,
        "evidence_search_rate": evidence_search_rate,
        "cost_optimization_rate": 1.0 - evidence_search_rate  # How many avoid search
    }

# ==============================================================================
# WORKFLOW PRIORITY
# ==============================================================================

def get_workflow_priority(workflow: WorkflowType) -> int:
    """
    Get processing priority for workflows.
    Lower number = higher priority.
    
    Process high-risk items (recommendations, health claims) first.
    """
    priority_map = {
        WorkflowType.RECOMMENDATION: 1,  # Highest priority (health/financial risk)
        WorkflowType.FACTUAL: 2,
        WorkflowType.PREDICTION: 3,
        WorkflowType.LOGICAL: 4,
        WorkflowType.OPINION: 5,
        WorkflowType.PROMOTIONAL: 6,
        WorkflowType.FEELING: 7,
        WorkflowType.SATIRE: 8,
        WorkflowType.INSTRUCTION: 9,
        WorkflowType.SKIP: 10,  # Lowest priority
    }
    return priority_map.get(workflow, 999)

def sort_workflows_by_priority(routing_map: dict) -> list[tuple[WorkflowType, list[Statement]]]:
    """
    Sort workflows by processing priority.
    Returns list of (workflow, statements) tuples in priority order.
    """
    workflows_with_statements = [
        (workflow, statements)
        for workflow, statements in routing_map.items()
        if statements  # Only include workflows with statements
    ]
    
    return sorted(workflows_with_statements, key=lambda x: get_workflow_priority(x[0]))
