import google.generativeai as genai
import numpy as np
import os
import pandas as pd
from tqdm import tqdm  
import faiss

genai.configure(api_key="AIzaSyCrbEDZuPcBoAJ5fSOWYD2JOEhmAYueaOU")
model = genai.GenerativeModel("gemini-2.0-flash")

current_dir = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.join(current_dir, 'database', 'embeddings.csv')

def get_embedding(value):
    response = genai.embed_content(
        model='models/embedding-001',
        content=value,
        task_type="retrieval_query"
    )
    vector = np.array(response['embedding'], dtype=np.float32)
    return vector / np.linalg.norm(vector)

df = pd.read_csv(working_dir)
saved = []

print("Generating embeddings...")
for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding QnAs", ncols=70):
    question = row['Question']
    answer = row['Answer']
    vector = get_embedding(question)
    saved.append(vector)

vector_path = current_dir
os.makedirs(f"{vector_path}/vectorstore", exist_ok=True)
vector_np = np.array(saved, dtype=np.float32)
np.save(f'{vector_path}/vectorstore/vectorembedding.npy', vector_np)

print("Embeddings saved successfully!")

print("Creating FAISS index....")
dimension=vector_np.shape[1]
index=faiss.IndexFlatIP(dimension)
faiss.normalize_L2(vector_np)
index.add(vector_np)

faiss.write_index(index,f"{vector_path}/vectorstore/faiss.index")
print("FAISS index saved successfully!")