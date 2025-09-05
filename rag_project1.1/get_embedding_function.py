from langchain_google_genai import GoogleGenerativeAIEmbeddings

#  API key de Google
GOOGLE_API_KEY = "AIzaSyCw9VzUFhWWadmurQ47Qz-6p4JzaF6zGhU"

# Qdrant Cloud
QDRANT_URL = "https://6b3a9368-b08f-4d67-a6d3-3aa89b0ba118.europe-west3-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.lSzwJpSV-M-rYJFY5OLvwphM3iVln_z74IkawrG1ya8"
COLLECTION_NAME = "rag_documents"

def get_embedding_function():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
