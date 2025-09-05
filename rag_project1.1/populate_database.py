import argparse
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from get_embedding_function import get_embedding_function, QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME

DATA_PATH = "data"

def main():
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_qdrant(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_qdrant(chunks: list[Document]):
    
    chunks_with_ids = calculate_chunk_ids(chunks)
    
   
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
    
  
    embedding_function = get_embedding_function()
    
    try:
       
        collections = client.get_collections()
        collection_exists = any(col.name == COLLECTION_NAME for col in collections.collections)
        
        if not collection_exists:
            print("🔧 Creando nueva colección...")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
            )
        
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embedding_function,
        )
        
        # Verificar documentos existentes (usando metadatos para evitar duplicados)
        existing_points = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=10000,
            with_payload=True
        )[0]
        
        existing_ids = {point.payload.get("id") for point in existing_points if point.payload}
        print(f"Number of existing documents in DB: {len(existing_ids)}")
        
        # Solo agregar documentos nuevos
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)
        
        if len(new_chunks):
            print(f"👉 Adding new documents: {len(new_chunks)}")
            
            # Agregar documentos al vector store
            vector_store.add_documents(new_chunks)
            print("✅ Documentos agregados exitosamente")
        else:
            print("✅ No new documents to add")
            
    except Exception as e:
        print(f"❌ Error connecting to Qdrant: {e}")
        print("💡 Verifica tu QDRANT_URL y QDRANT_API_KEY en get_embedding_function.py")


def calculate_chunk_ids(chunks):
   

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

       
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

       
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

 
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    try:
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        client.delete_collection(collection_name=COLLECTION_NAME)
        print("🗑️ Colección eliminada")
    except Exception as e:
        print(f"⚠️ Error clearing database: {e}")


if __name__ == "__main__":
    main()