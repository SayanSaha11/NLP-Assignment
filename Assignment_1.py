import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def tokenize_statement(statement):
    return word_tokenize(statement)

# Test Case
print(tokenize_statement("Hello, how are you?"))
