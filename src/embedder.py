print("\nLoading the embedding model 'all-MiniLM-L6-v2'. This may take a moment...")
model = SentenceTransformer('all-MiniLM-L6-v2')
corpus = df_cleaned['embedding_text'].tolist()

print(f"Generating embeddings for {len(corpus)} whisky descriptions...")
embeddings = model.encode(corpus, show_progress_bar=True)

print("\n--- Embedding Generation Complete! ---")
print(f"Shape of the embeddings array: {embeddings.shape}")

# Store the embeddings globally for subsequent cells
globals()['embeddings'] = embeddings

embeddings_file = 'embeddings.npy'
cleaned_data_file = 'cleaned_whiskies.csv'

np.save(embeddings_file, embeddings)
df_cleaned.to_csv(cleaned_data_file, index=False)