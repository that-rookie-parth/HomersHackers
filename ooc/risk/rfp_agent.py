import os
from typing import Dict, List
import json
from datetime import datetime
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
# import openai  # or import groq for Groq API
import groq

class RFPAnalysisAgent:
    def __init__(self, model_path: str = "models/rfp_patterns.json"):
        self.model_path = model_path
        self.patterns_db = self._load_patterns()
        self.historical_analyses = []
        self.vectorizer = TfidfVectorizer(max_features=1000)
        
    def _load_patterns(self) -> Dict:
        """Load existing patterns or create new DB"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'r') as f:
                return json.load(f)
        return {
            'risk_patterns': {},
            'balanced_solutions': {},
            'learned_examples': []
        }
    
    def analyze_clause(self, clause_text: str, api_key: str) -> Dict:
        """Analyze clause using both patterns and AI"""
        # First check against known patterns
        risks = self._check_known_patterns(clause_text)
        
        # Then use AI to identify new patterns
        ai_analysis = self._ai_risk_analysis(clause_text, api_key)
        
        # Combine and learn from results
        combined_analysis = self._combine_analyses(risks, ai_analysis)
        self._learn_from_analysis(clause_text, combined_analysis)
        
        return combined_analysis
    
    def _ai_risk_analysis(self, text: str, api_key: str) -> Dict:
        """Use LLM to analyze risks and suggest improvements"""
        prompt = f"""
        Analyze this RFP clause for ConsultAdd (IT services company). Identify:
        1. Any terms that could disadvantage ConsultAdd
        2. Risk level (High/Medium/Low)
        3. Specific concerns
        4. Suggested balanced alternative
        
        Clause:
        {text}
        
        Respond in JSON format:
        {{
            "risks": [
                {{
                    "type": "risk_type",
                    "level": "risk_level",
                    "concern": "specific_concern",
                    "suggestion": "balanced_alternative"
                }}
            ],
            "patterns_identified": ["pattern1", "pattern2"]
        }}
        """
        
        response = self._get_llm_response(prompt, api_key)
        return json.loads(response)
    
    def _learn_from_analysis(self, text: str, analysis: Dict):
        """Learn new patterns from analysis"""
        # Store example
        self.patterns_db['learned_examples'].append({
            'text': text,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        })
        
        # Cluster similar clauses to identify patterns
        if len(self.patterns_db['learned_examples']) > 10:
            self._update_patterns()
        
        # Save updated patterns
        self._save_patterns()
    
    def _update_patterns(self):
        """Use clustering to identify new patterns"""
        texts = [ex['text'] for ex in self.patterns_db['learned_examples']]
        vectors = self.vectorizer.fit_transform(texts)
        
        # Cluster similar clauses
        clusters = DBSCAN(eps=0.3, min_samples=2).fit(vectors)
        
        # Extract patterns from clusters
        for cluster_id in set(clusters.labels_):
            if cluster_id != -1:  # Skip noise
                cluster_texts = [t for i, t in enumerate(texts) 
                               if clusters.labels_[i] == cluster_id]
                self._extract_pattern_from_cluster(cluster_texts)
    
    def suggest_balanced_terms(self, clause_text: str, api_key: str) -> str:
        """Generate balanced alternative using AI and learned patterns"""
        # Get historical similar clauses
        similar_clauses = self._find_similar_clauses(clause_text)
        
        prompt = f"""
        Given this RFP clause and similar historical examples, suggest a balanced version that:
        1. Protects ConsultAdd's interests
        2. Maintains professional relationship
        3. Is fair to both parties
        
        Original clause:
        {clause_text}
        
        Similar historical examples:
        {json.dumps(similar_clauses, indent=2)}
        
        Provide a balanced alternative clause:
        """
        
        return self._get_llm_response(prompt, api_key)
    
    def _save_patterns(self):
        """Save learned patterns to disk"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'w') as f:
            json.dump(self.patterns_db, f, indent=2)


