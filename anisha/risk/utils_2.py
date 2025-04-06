import re
import logging
from typing import Dict, List, Tuple
import fitz
from transformers import pipeline

def extract_rfp_sections(text: str) -> Dict[str, str]:
    """Extract common RFP sections using standard RFP structure"""
    rfp_sections = {
        'overview': '',
        'scope_of_work': '',
        'requirements': [],
        'evaluation_criteria': '',
        'timeline': '',
        'submission_requirements': '',
        'terms_and_conditions': []
    }
    
    # Common RFP section headers
    section_patterns = {
        'overview': r'(?i)(introduction|overview|background|purpose).*?\n',
        'scope_of_work': r'(?i)(scope\s+of\s+work|statement\s+of\s+work|services\s+required).*?\n',
        'requirements': r'(?i)(technical\s+requirements|functional\s+requirements|specifications).*?\n',
        'evaluation_criteria': r'(?i)(evaluation\s+criteria|selection\s+process).*?\n',
        'timeline': r'(?i)(timeline|schedule|important\s+dates).*?\n',
        'submission_requirements': r'(?i)(submission|proposal\s+requirements|response\s+format).*?\n',
        'terms_and_conditions': r'(?i)(terms\s+and\s+conditions|general\s+conditions).*?\n'
    }
    
    current_section = None
    lines = text.split('\n')
    
    for line in lines:
        for section, pattern in section_patterns.items():
            if re.match(pattern, line):
                current_section = section
                continue
        
        if current_section:
            if isinstance(rfp_sections[current_section], list):
                rfp_sections[current_section].append(line)
            else:
                rfp_sections[current_section] += line + '\n'
    
    return rfp_sections

def extract_requirements(text: str) -> List[Dict[str, str]]:
    """Extract specific requirements from RFP text"""
    requirements = []
    
    # Common requirement indicators
    requirement_patterns = [
        r'(?:^|\n)(?P<id>\d+\.[\d\.])\s(?P<text>.*?)(?=\n\d+\.|\Z)',  # Numbered requirements
        r'(?:^|\n)(?P<id>[A-Z]\.[\d\.])\s(?P<text>.*?)(?=\n[A-Z]\.|\Z)',  # Letter requirements
        r'(?:^|\n)(?P<id>R\d+)\s*(?P<text>.*?)(?=\nR\d+|\Z)',  # R-prefixed requirements
        r'(?:^|\n)(?:Requirement|REQ):\s*(?P<text>.*?)(?=\n(?:Requirement|REQ):|\Z)'  # Explicit requirements
    ]
    
    for pattern in requirement_patterns:
        matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
        for match in matches:
            requirement = {
                'id': match.group('id') if 'id' in match.groupdict() else 'N/A',
                'text': match.group('text').strip(),
                'type': classify_requirement_type(match.group('text')),
                'mandatory': is_mandatory_requirement(match.group('text'))
            }
            requirements.append(requirement)
    
    return requirements

def classify_requirement_type(text: str) -> str:
    """Classify the type of requirement"""
    type_indicators = {
        'technical': r'(?i)(technical|system|software|hardware|platform|infrastructure)',
        'functional': r'(?i)(shall|must|will|should|functionality|feature)',
        'operational': r'(?i)(operation|maintain|support|service|availability)',
        'compliance': r'(?i)(comply|standard|regulation|certif|security)',
        'business': r'(?i)(business|process|workflow|organization)'
    }
    
    for req_type, pattern in type_indicators.items():
        if re.search(pattern, text):
            return req_type
    return 'general'

def is_mandatory_requirement(text: str) -> bool:
    """Determine if a requirement is mandatory"""
    mandatory_patterns = [
        r'(?i)(shall|must|required|mandatory|essential)',
        r'(?i)(will\s+be\s+required)',
        r'(?i)(is\s+required)',
        r'(?i)(are\s+to\s+be\s+provided)'
    ]
    
    return any(re.search(pattern, text) for pattern in mandatory_patterns)

def analyze_rfp_document(pdf_path: str) -> Dict:
    """Main function to analyze RFP document"""
    try:
        # Extract text from PDF
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        
        # Extract and analyze sections
        sections = extract_rfp_sections(text)
        requirements = extract_requirements(text)
        
        # Analyze requirements
        analysis = {
            'sections': sections,
            'requirements': requirements,
            'statistics': {
                'total_requirements': len(requirements),
                'mandatory_requirements': sum(1 for r in requirements if r['mandatory']),
                'by_type': {}
            }
        }
        
        # Calculate statistics
        for req in requirements:
            req_type = req['type']
            analysis['statistics']['by_type'][req_type] = analysis['statistics']['by_type'].get(req_type, 0) + 1
        
        return analysis
        
    except Exception as e:
        logging.error(f"Error analyzing RFP document: {str(e)}")
        return None
    
def identify_biased_patterns():
    """Define patterns that could disadvantage ConsultAdd"""
    return {
        'unilateral_termination': {
            'pattern': r'(?i)(terminate|termination).*?(at\s+(\w+\s+){0,3}discretion|without\s+cause|any\s+reason|sole\s+discretion)',
            'risk_level': 'High',
            'suggestion': 'Add mutual termination rights or require notice period of at least 30 days'
        },
        'unlimited_liability': {
            'pattern': r'(?i)(indemnify|indemnification|liable|liability).*?(all|any|whatever|whatsoever|unlimited)',
            'risk_level': 'High',
            'suggestion': 'Cap liability to contract value or 12 months of fees'
        },
        'payment_terms': {
            'pattern': r'(?i)(payment.*?([6-9][0-9]|[1-9][0-9]{2,})\s*days|withhold\s+payment)',
            'risk_level': 'Medium',
            'suggestion': 'Adjust payment terms to net-30 or net-45 days maximum'
        },
        'intellectual_property': {
            'pattern': r'(?i)(ip|intellectual\s+property|work\s+product).?(shall\s+belong|ownership|transfer|assign).?(client|customer)',
            'risk_level': 'High',
            'suggestion': 'Limit IP transfer to project deliverables only, exclude pre-existing IP'
        },
        'non_compete': {
            'pattern': r'(?i)(non-compete|not\s+compete|restrict.?business).?(years?|months?)',
            'risk_level': 'Medium',
            'suggestion': 'Limit non-compete to specific clients/regions and maximum 12 months'
        }
    }

def analyze_clause_bias(text: str) -> dict:
    """Analyze clause for potential bias against ConsultAdd"""
    biased_patterns = identify_biased_patterns()
    findings = []
    
    for bias_type, details in biased_patterns.items():
        if re.search(details['pattern'], text):
            findings.append({
                'type': bias_type,
                'risk_level': details['risk_level'],
                'suggestion': details['suggestion'],
                'original_text': text
            })
    
    return findings

def suggest_balanced_clause(finding: dict) -> str:
    """Generate balanced alternative for biased clause"""
    balanced_templates = {
        'unilateral_termination': """
        Either party may terminate this agreement:
        1. With cause: Upon material breach with 30 days written notice to cure
        2. Without cause: Upon 60 days written notice to the other party
        """,
        'unlimited_liability': """
        Each party's liability shall be limited to:
        1. Direct damages not exceeding 12 months of fees
        2. Exclusion of indirect, special, or consequential damages
        """,
        'payment_terms': """
        Payment terms:
        1. Net-30 days from invoice date
        2. Disputed amounts must be notified within 15 days
        3. Undisputed amounts must be paid per schedule
        """,
        'intellectual_property': """
        1. ConsultAdd retains ownership of pre-existing IP
        2. Client owns project-specific deliverables
        3. ConsultAdd retains right to use general knowledge and experience
        """,
        'non_compete': """
        Non-compete limited to:
        1. Direct client accounts served
        2. 12 months post-termination
        3. Specific geographic region
        """
    }
    
    return balanced_templates.get(finding['type'], "Suggested revision needed")