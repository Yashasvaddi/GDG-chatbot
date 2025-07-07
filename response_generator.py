import os
import google.generativeai as genai
import numpy as np
import pandas as pd
import faiss

model_names = ["gemini-2.5-flash","gemini-2.5-pro","gemini-2.0-flash","gemini-1.5-flash","gemini-2.0-flash-exp"]

genai.configure(api_key="AIzaSyC3vNkSnEJl-eFloSm9M4Bw0F_cJv2vusY")
model=genai.GenerativeModel(model_names[i])

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
    
    if similarity<95:
        Question=f'''{context}
        {example}
        {remember}'''
        response=model.generate_content(Question)
        print(response.text)
        return response.text
    else:
        print(top_row['Answer'],similarity)
        return top_row['Answer']

if __name__=='__main__':
    ask="How can i help you?"
    count=0
    try:
        while True:
            if count>=1:
                ask="Is there anything else that i can help u with?"
            query=input(f"{ask}\n")
            response_gen(query,1)
            count=count+1
    except Exception as e:
        if i<len(model_names):
            i=i+1
            print(f"Switching model to {model_names[i]}")
        else:
            i=0