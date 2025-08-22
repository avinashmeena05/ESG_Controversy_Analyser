# Simple transparent lexicon with per-category weights
LEXICON = {
    'E': {
        'spill': 8, 'pollution': 7, 'wastewater': 6, 'emission': 5,
        'deforestation': 8, 'toxic': 7, 'contamination': 8, 'oil': 5
    },
    'S': {
        'harassment': 8, 'safety': 6, 'child': 9, 'labor': 7,
        'discrimination': 8, 'community': 5, 'strike': 6, 'boycott': 5
    },
    'G': {
        'fraud': 9, 'bribery': 9, 'corruption': 8, 'whistleblower': 6,
        'fine': 5, 'lawsuit': 6, 'governance': 4, 'board': 3, 'money-laundering': 8
    }
}

def score_severity(text: str, category: str) -> int:
    text_l = (text or '').lower()
    score = 0
    for kw, w in LEXICON.get(category, {}).items():
        if kw in text_l:
            score += w
    # Cap and rescale to 0-100
    score = min(score, 25)
    return int( (score / 25) * 100 )
