from nltk.stem import PorterStemmer

def stem_word(word):
    stemmer = PorterStemmer()
    return stemmer.stem(word)

# Test Case
print(stem_word("programming"))  # Expected Output: "program"
