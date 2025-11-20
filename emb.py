# test run
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode("Smart Campus Guide is working!", convert_to_numpy=True)

print("Embedding shape:", emb.shape)
print("Embedding sample:", emb[:5])  # first 5 values
