import numpy as np

# Position encodings
def sinusoidal_encoding(p):
    return np.array([np.sin(p), np.cos(p)])

# Encodings for p1 and p2
v = sinusoidal_encoding(1)
w = sinusoidal_encoding(4)

# Cosine similarity calculation
cosine_similarity = np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w))
cosine_similarity
print(cosine_similarity)