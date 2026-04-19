import re

def tokenize(text):
    """
    Returns: list[str]
    """
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return tokens