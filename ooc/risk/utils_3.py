from rfp_agent import RFPAnalysisAgent
import os
# Initialize agent
rfp_agent = RFPAnalysisAgent()

def analyze_clause_bias(text: str) -> dict:
    """Analyze clause using self-learning agent"""
    api_key = os.getenv("GROQ_API_KEY")  # Get from environment variable
    return rfp_agent.analyze_clause(text, api_key)

def suggest_balanced_clause(finding: dict) -> str:
    """Get AI-generated balanced alternative"""
    api_key = os.getenv("GROQ_API_KEY")
    return rfp_agent.suggest_balanced_terms(finding['original_text'], api_key)