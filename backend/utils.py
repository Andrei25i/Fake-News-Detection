import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text_general(text: str) -> str:
    """Applies common pre-processing: anti-bias source removal and technical cleaning."""
    text = str(text).lower()
    
    # Clean [video], [image], [img], etc tags (frequent in fake news)
    text = re.sub(r'[\(\[]\s*(video|image|img|photos?|watch).*?[\)\]]', '', text)
    text = re.sub(r'^(watch|image|video)\s?:\s?', '', text)

    # ANTI-BIAS: Remove generic source headers like "CITY (SOURCE) -"
    text = re.sub(r'^.*?\s\(.*?\)\s-\s', '', text)
    # 2. Remove standalone mentions of common agencies
    text = re.sub(r'\b(reuters|cnn|ap|united press international)\b', '', text)
    
    # Technical cleaning (URLs, HTML)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def clean_text_classic(text: str, stop_words: set) -> str:
    """Aggressive cleaning for Logistic Regression."""
    text = clean_text_general(text)

    # Remove all non-letters (punctuation, numbers)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()

    # Remove stopwords and Lemmatize
    cleaned_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    
    return ' '.join(cleaned_words)

def clean_text_bert(text: str) -> str:
    """Minimal cleaning for BERT (keeps context)."""
    return clean_text_general(text)