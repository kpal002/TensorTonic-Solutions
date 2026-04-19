import re

def text_normalize(text, operations):
    """
    Returns: str
    """
    for op in operations:
        if op == "lowercase":
            text = text.lower()
            
        elif op == "remove_punctuation":
            text = re.sub(r'[^\w\s]', '', text)
            
        elif op == "remove_digits":
            text = re.sub(r"\d+", "", text)

        elif op == "collapse_whitespace":
            text = re.sub(r"\s+", " ", text)

        else:
            text = text.strip()

    return text
            
