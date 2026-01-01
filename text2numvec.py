import nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Download necessary resources for tokenization
nltk.download('punkt')

# 1. Prepare a sample dataset (A list of sentences)
data = [
    "Machine learning is a method of data analysis.",
    "Artificial intelligence is the future of technology.",
    "Neural networks are inspired by the biological brain.",
    "Data science involves statistics and programming.",
    "Word embeddings represent words as numerical vectors."
]

# 2. Preprocessing: Tokenize and lowercase
tokenized_data = [word_tokenize(sentence.lower()) for sentence in data]

# 3. Initialize and Train the Word2Vec Model
# vector_size: dimensionality of the word vectors (e.g., 100)
# window: maximum distance between the current and predicted word
# min_count: ignores all words with total frequency lower than this
# sg: 0 for CBOW, 1 for Skip-gram
model = Word2Vec(sentences=tokenized_data, 
                 vector_size=100, 
                 window=5, 
                 min_count=1, 
                 workers=4)

# 4. Accessing the Vectors
word = "data"
vector = model.wv[word]

print(f"Vector for '{word}' (first 10 elements): \n{vector[:10]}")

# 5. Finding similar words
similar_words = model.wv.most_similar("intelligence", topn=3)
print(f"\nWords most similar to 'intelligence': {similar_words}")
