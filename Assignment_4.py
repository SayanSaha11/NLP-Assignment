import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

def lemmatize_word(word):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word, pos='v')

# Test Case
print(lemmatize_word("running"))  # Expected Output: "run"
