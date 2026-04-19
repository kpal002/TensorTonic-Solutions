import re

def extract_patterns(text, pattern_type):
    """
    Returns: list[str]
    """
    patterns = {
        "emails": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
        "urls": r'https?://[^\s]+(?<![,.)!?;:])',
        "dates": r'\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b',
        "money": r'\$\d+(?:\.\d{2})?',
        "hashtags": r'#\w+',
    }
    pattern = patterns.get(pattern_type)
    if pattern is None:
        return []
    return re.findall(pattern, text)