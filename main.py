import os
import google.generativeai as genai
import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",  # React/Frontend
    "https://your-domain.com", # production domain
    "*",  # (use "*" only if you want to allow all origins)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

working_dir=os.path.dirname(os.path.abspath(__file__))
database_path=os.path.join(working_dir,'database','embeddings.csv')
embedding_path=os.path.join(working_dir,'vectorstore','vectorembedding.npy')
faiss_index_path=os.path.join(working_dir,'vectorstore','faiss.index')


embedding_vector=np.load(embedding_path)
df=pd.read_csv(database_path)
index=faiss.read_index(faiss_index_path)

def get_embedding(value):
    response=genai.embed_content(
        model='models/embedding-001',
        content=value,
        task_type="retrieval_query"
    )
    vector=np.array(response['embedding'],dtype=np.float32)
    return vector/np.linalg.norm(vector)


def response_gen(query,top_k):
    gemini_api_key_embed=os.getenv('gemini_api_key_embed')
    genai.configure(api_key=gemini_api_key_embed)
    model=genai.GenerativeModel('gemini-1.5-flash')
    query_embedding=get_embedding(query).reshape(1,-1)
    distances,indices=index.search(query_embedding,top_k)
    top_index=indices[0][0]
    top_row=df.iloc[top_index]
    similarity=distances[0][0]*100
    context=f"You are a chatbot called 'ASKGDG'and you are here to assist people. Talk as if you are customer support executive and send response to the question {query} in max 2 lines. Dont say anything apart from the answer to the question."
    example='''
        Question: What is GDG?
        Answer: GDG is a committee called as GOOGLE DEVELOPERS GROUP.
        Notes: No need to use 'Here is the answer to' or any such similar sentences.
    '''
    remember="You are  not allowed to answer anything that is not related to GOOGLE, THADOMAL SHAHANI ENGINEERING COLLEGE or GDG."
    
    if similarity<90:
        gemini_api_key_resp=os.getenv('gemini_api_key_resp')
        genai.configure(api_key=gemini_api_key_resp)
        model=genai.GenerativeModel('gemini-1.5-flash')
        Question=f'''{context}
        {example}
        {remember}'''
        response=model.generate_content(Question)
        print(response.text)
        return response.text
    else:
        print(top_row['Answer'],similarity)
        return top_row['Answer']

class chatbot(BaseModel):
    query:str

@app.post("/chat")
def main(payload:chatbot):
    answer=response_gen(payload.query,1)
    return answer
