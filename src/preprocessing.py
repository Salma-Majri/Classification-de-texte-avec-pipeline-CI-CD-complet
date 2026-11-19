import re
import nltk
from nltk.corpus import stopwords

# Chargement des stopwords (une seule fois, au niveau module)
nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """Nettoyage : minuscules, suppression HTML et ponctuation, stopwords NLTK."""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'<.*?>', '', text)          # balises HTML
    text = re.sub(r'[^a-zA-Z\s]', '', text)    # garder lettres et espaces
    text = text.lower()
    words = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)