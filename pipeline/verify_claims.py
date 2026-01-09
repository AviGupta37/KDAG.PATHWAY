def verify_claim(claim, evidence_chunks):
    text = " ".join([c["text"] for c in evidence_chunks])

    # Simple heuristic baseline
    if claim.lower() in text.lower():
        return "supported"

    contradiction_words = ["never", "opposite", "denies", "contradicts"]
    if any(w in text.lower() for w in contradiction_words):
        return "contradicted"

    return "unconstrained"

