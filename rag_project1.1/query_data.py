import warnings
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from get_embedding_function import get_embedding_function, QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME

# Ocultar warnings de deprecación
warnings.filterwarnings("ignore", category=DeprecationWarning)

PROMPT_TEMPLATE = """
Responde en **español natural y claro**, usando únicamente la siguiente información del contexto.
Si el contexto no contiene la respuesta, indica que no aparece en el documento.

Contexto:
{context}

---

Pregunta: {question}
Respuesta en español:
"""

# API key visible SOLO para pruebas
GOOGLE_API_KEY = "AIzaSyAGartUrEQ36t4CkmI1lZVvMragPuxcwkE"


def query_rag(query_text: str):
    try:
        # Conectar a Qdrant
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        
        # Verificar que la colección existe
        collections = client.get_collections()
        collection_exists = any(col.name == COLLECTION_NAME for col in collections.collections)
        
        if not collection_exists:
            print("❌ Error: La colección no existe. Ejecuta primero: python populate_database.py")
            return "Error: Base de datos no encontrada"
        
        # Obtener información de la colección
        collection_info = client.get_collection(COLLECTION_NAME)
        document_count = collection_info.points_count
        print(f"📊 Documentos en la base de datos: {document_count}")
        
        if document_count == 0:
            print("❌ La base de datos está vacía. Ejecuta: python populate_database.py")
            return "Error: Base de datos vacía"

        # Crear vector store
        embedding_function = get_embedding_function()
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embedding_function,
        )

        # Buscar en los documentos
        print(f"🔍 Buscando: {query_text}")
        results = vector_store.similarity_search_with_score(query_text, k=5)
        
        if not results:
            print("❌ No se encontraron resultados relevantes")
            return "No se encontraron documentos relevantes"
        
        print(f"✅ Encontrados {len(results)} documentos relevantes")
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        # Construir el prompt
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        # Generar respuesta con Gemini
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3
        )
        response = model.invoke(prompt)

        
        print(f"\n📄 Respuesta: {response.content.strip()}\n")
        return response.content
        
    except Exception as e:
        print(f"❌ Error conectando a Qdrant: {e}")
        print("💡 Verifica tu QDRANT_URL y QDRANT_API_KEY en get_embedding_function.py")
        return "Error de conexión con la base de datos"


def main():
    print("💬 Sistema RAG con Qdrant Cloud listo. Escribe tu pregunta o 'salir' para terminar.\n")
    while True:
        query_text = input("👉 Pregunta: ")
        if query_text.lower() in ["salir", "exit", "quit"]:
            print("👋 Saliendo del sistema...")
            break
        query_rag(query_text)


if __name__ == "__main__":
    main()
