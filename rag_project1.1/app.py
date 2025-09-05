import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
import tempfile
import asyncio
import time

# 🔧 Fix para Streamlit + asyncio
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
    
    
# 🔑 Configuración de API Keys y Qdrant
GOOGLE_API_KEY = "AIzaSyAGartUrEQ36t4CkmI1lZVvMragPuxcwkE"
QDRANT_URL = "https://6b3a9368-b08f-4d67-a6d3-3aa89b0ba118.europe-west3-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.lSzwJpSV-M-rYJFY5OLvwphM3iVln_z74IkawrG1ya8"

# ⚙️ Configuración de modelos
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)

# 🔌 Conectar a Qdrant
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# 📂 Procesar PDFs y guardarlos en una colección única
def process_pdfs(uploaded_files):
    # Crear nombre de colección único (ej: rag_documents_1699999999) importante porque si se mesclan y puede dar error 
    collection_name = f"rag_documents_{int(time.time())}"
    st.session_state.collection_name = collection_name

    # Crear colección en Qdrant
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
    )

    total_chunks = 0
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            loader = PyPDFLoader(tmp.name)
            documents = loader.load()

        # 🔧 Ajuste para mantener más contexto
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,   # más grande para no perder info
            chunk_overlap=150  # más solapamiento
        )
        chunks = splitter.split_documents(documents)

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )

        vector_store.add_documents(chunks)
        total_chunks += len(chunks)

    return len(uploaded_files), total_chunks


# ❓ Consultar con RAG
def query_rag(query_text):
    if "collection_name" not in st.session_state:
        return "⚠️ Primero sube un PDF para crear la base de datos."

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=st.session_state.collection_name,
        embedding=embeddings,
    )

    results = vector_store.similarity_search(query_text, k=5)
    if not results:
        return "⚠️ No encontré información relevante en los documentos."

    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""
    Responde en español natural y claro usando únicamente la siguiente información:

    {context}

    ---
    Pregunta: {query_text}
    Respuesta:
    """

    response = llm.invoke(prompt)
    return response.content


# 🎨 Interfaz con Streamlit
st.title("📄 RAG con Qdrant + Gemini")
st.write("Sube uno o varios PDFs y haz preguntas sobre su contenido.")

# Subida de múltiples PDFs
uploaded_files = st.file_uploader("Subir PDFs", type="pdf", accept_multiple_files=True)
if uploaded_files:
    num_docs, num_chunks = process_pdfs(uploaded_files)
    st.success(f"✅ {num_docs} documento(s) procesado(s) en {num_chunks} fragmentos.")

# Inicializar historial
if "history" not in st.session_state:
    st.session_state.history = []

# Caja de preguntas
question = st.text_input("👉 Escribe tu pregunta aquí:")

if st.button("Preguntar"):
    if question:
        answer = query_rag(question)
        st.session_state.history.append({"q": question, "a": answer})
    else:
        st.warning("Por favor escribe una pregunta.")

# Mostrar historial
if st.session_state.history:
    st.write("### 📜 Historial de preguntas")
    for i, item in enumerate(st.session_state.history, 1):
        st.write(f"**Q{i}:** {item['q']}")
        st.write(f"📝 {item['a']}")
        st.write("---")

