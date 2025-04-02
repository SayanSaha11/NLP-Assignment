import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

def extract_named_entities(sentence):
    words = word_tokenize(sentence)
    pos_tags = pos_tag(words)
    named_entities = ne_chunk(pos_tags)
    entities = []
    for subtree in named_entities:
        if isinstance(subtree, Tree):
            entity_name = " ".join([token for token, tag in subtree.leaves()])
            entity_type = subtree.label()
            entities.append((entity_name, entity_type))
    return entities

# Test Case
print(extract_named_entities("John Smith works at Google."))  # Expected Output: [("John Smith", "PERSON"), ("Google", "ORGANIZATION")]
