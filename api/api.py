from fastapi import FastAPI,Query
from response_generator import response_gen

app=FastAPI()

@app.get('/query')
def handle_query(question: str = Query(...),top_k: int = 1):
    return {"response":response_gen(question,top_k)}
