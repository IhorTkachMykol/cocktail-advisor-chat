# cocktail-advisor-chat

Step 1: Setting Up the Project

Use a virtual environment (via conda or venv).
Install necessary libraries:

Step 2: Build the Chat Application

Backend (FastAPI)

Create an API endpoint to interact with the chatbot

Load FAISS index

Define RAG pipeline

Frontend
Option: CLI
Create a simple Python CLI to send questions to the FastAPI endpoint and display answers.

Step 3: Build and Integrate the Vector Database
Index Cocktail Data:

Embed cocktail descriptions and ingredients into a vector database using FAISS:

Embed and store in FAISS

Extract user preferences from chat messages (e.g., favorite ingredients) and update the vector database:

Step 4: Implement RAG Pipeline

Step 5: Test Use Cases

Load Data:

Run Queries:

Test retrieval with questions like:

Ensure responses are relevant and improve the dataset or model as necessary.

Step 6: Finalize and Document
README.md


