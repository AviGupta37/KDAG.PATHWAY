import re

def extract_claims(backstory):
    sentences = re.split(r"[.?!]", backstory)
    claims = [s.strip() for s in sentences if len(s.strip()) > 15]
    return claims

