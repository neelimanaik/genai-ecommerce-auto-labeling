HAIR_KEYWORDS = {
    "shampoo", "conditioner", "hair oil", "hair serum",
    "hair fall", "anti hair", "scalp", "dandruff",
    "hairloss", "hair loss",  "volumizing", "keratin", "hair mask"
}

SKIN_KEYWORDS = {
    "soap", "handwash", "body wash", "skin",
    "lotion", "cream", "scrub", "moisturizer",
    "face wash","lip balm", "cream", "lotion", "scrub"
}

def rule_based_classify(text: str):
    text = text.lower()

    hair_hits = sum(1 for k in HAIR_KEYWORDS if k in text)
    skin_hits = sum(1 for k in SKIN_KEYWORDS if k in text)

    if hair_hits > skin_hits and hair_hits >= 1:
        return "Hair Care", "CONFIDENT"

    if skin_hits > hair_hits and skin_hits >= 1:
        return "Skin Care", "CONFIDENT"

    return None, "AMBIGUOUS"