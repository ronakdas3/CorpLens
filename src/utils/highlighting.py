import re


def highlight_relevant_sentence(chunk, query):
    """
    Finds the most relevant sentence in a chunk based on keyword overlap.
    """

    sentences = re.split(r'(?<=[.!?]) +', chunk)

    query_words = set(re.findall(r"\w+", query.lower()))

    best_sentence = ""
    best_score = 0

    for sentence in sentences:
        words = set(re.findall(r"\w+", sentence.lower()))
        score = len(query_words & words)

        if score > best_score:
            best_score = score
            best_sentence = sentence

    return best_sentence if best_sentence else chunk[:200]