from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(sent1, sent2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([sent1, sent2])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return similarity[0][0]

# Test Case
print(compute_similarity("This is a sample sentence.", "This sentence is just a sample."))  # Expected around 0.8
