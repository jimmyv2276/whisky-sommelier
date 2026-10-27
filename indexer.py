import faiss
import numpy as np

# Load the embeddings from the file
embeddings = np.load('embeddings.npy')

# Get the dimensionality of the embeddings
dimension = embeddings.shape[1]

# Initialize a FAISS index (IndexFlatL2 is a simple index for L2 distance search)
index = faiss.IndexFlatL2(dimension)

# Add the embeddings to the index
index.add(embeddings)

print(f"FAISS index created successfully with {index.ntotal} vectors and dimension {index.d}.")

# Store the index globally for subsequent cells
globals()['faiss_index'] = index