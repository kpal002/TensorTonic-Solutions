import re

def segment_sentences(text):
    """
    Returns: list[str]
    """
    A = {"Mr", "Mrs", "Dr", "Jr", "Sr", "vs", "etc", "i.e", "e.g", "U.S"}
    parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+', text) if p.strip()]

    result = []
    i = 0

    while i < len(parts):
        current = parts[i].strip()

        while True:
            words = current.split()
            if not words:
                break

            last_word = words[-1].rstrip(".!?")
            if last_word in A and i + 1 < len(parts):
                i += 1
                current = current + " " + parts[i].strip()
            else:
                break

        result.append(current)
        i += 1

    return result