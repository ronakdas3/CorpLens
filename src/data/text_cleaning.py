import re


def remove_references(text):

    pattern = r"(references|bibliography)(.*)"

    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)

    if match:
        text = text[:match.start()]

    return text