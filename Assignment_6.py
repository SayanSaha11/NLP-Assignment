from sklearn.feature_extraction.text import TfidfVectorizer

def compute_tfidf(documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    return tfidf_matrix.toarray(), vectorizer.get_feature_names_out()

# Test Case
documents = ["This is a sample document.", "Another document with different content."]
tfidf_matrix, feature_names = compute_tfidf(documents)
print(tfidf_matrix)
print(feature_names)
