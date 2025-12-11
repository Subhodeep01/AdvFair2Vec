from gensim.models import KeyedVectors

# Load your custom vectors
# model = KeyedVectors.load_word2vec_format('checkpoints/debiased_vectors.txt', binary=False)
model = KeyedVectors.load_word2vec_format('base_vectors.txt', binary=False)

# Perform vector math
result = model.most_similar(positive=['engineer', 'she'], negative=['he'])
print(result) 
# Ideally, this should NOT return "nurse" anymore, but something like "doctor" or "physician"