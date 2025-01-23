

from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
import os
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Base models for user input
class UserQuery(BaseModel):
    question: str

class MemoryUpdate(BaseModel):
    favorite_ingredients: list
    favorite_cocktails: list

# Initialize LLM and Vector DB
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

embeddings = OpenAIEmbeddings()

def initialize_vector_db():
    # Load and preprocess cocktail data
    cocktail_data = pd.read_csv("final_cocktails.csv")

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = []

    for _, row in cocktail_data.iterrows():
        ingredients = row.get("ingredients", "")
        name = row.get("name", "Unknown Cocktail")
        content = f"Name: {name}\nIngredients: {ingredients}"
        documents.extend(text_splitter.split_text(content))

    # Create vector database
    vector_db = FAISS.from_texts(documents, embeddings)
    return vector_db

vector_db = initialize_vector_db()
retriever = vector_db.as_retriever()
qa_chain = RetrievalQA(llm=llm, retriever=retriever)

# In-memory storage for user-specific preferences
user_memories = {
    "favorite_ingredients": [],
    "favorite_cocktails": []
}

# Endpoint for querying the LLM with RAG
@app.post("/query")
def query_llm(user_query: UserQuery):
    try:
        response = qa_chain.run(user_query.question)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for updating user memories
@app.post("/update_memory")
def update_memory(memory_update: MemoryUpdate):
    global user_memories

    user_memories["favorite_ingredients"].extend(memory_update.favorite_ingredients)
    user_memories["favorite_cocktails"].extend(memory_update.favorite_cocktails)

    # Deduplicate the lists
    user_memories["favorite_ingredients"] = list(set(user_memories["favorite_ingredients"]))
    user_memories["favorite_cocktails"] = list(set(user_memories["favorite_cocktails"]))

    return {"message": "Memories updated successfully!", "current_memory": user_memories}

# Endpoint to retrieve user memories
@app.get("/get_memory")
def get_memory():
    return user_memories

# Helper function to recommend cocktails based on memory
@app.get("/recommend")
def recommend_cocktails():
    if not user_memories["favorite_ingredients"]:
        raise HTTPException(status_code=400, detail="No favorite ingredients found in memory.")

    favorite_ingredients = user_memories["favorite_ingredients"]
    query = f"Recommend 5 cocktails that contain {', '.join(favorite_ingredients)}."

    try:
        response = qa_chain.run(query)
        return {"recommendations": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
