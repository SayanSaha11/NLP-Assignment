import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    filtered_words = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(filtered_words)

# Test Case
print(preprocess_text("This is a sample text. It contains punctuation and stopwords."))  # Expected Output: "sample text contains"
