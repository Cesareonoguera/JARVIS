import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # <-- CORREGIDO
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings # <-- Más estable y gratis

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="JARVIS BIM", page_icon="🤖")
st.title("🤖 JARVIS BIM - DeepSeek Edition")

# 2. GESTIÓN DE LA API KEY (SECRETS)
# Intentamos leer de st.secrets, si falla, mostramos aviso amable
if "DEEPSEEK_API_KEY" in st.secrets:
    api_key = st.secrets["DEEPSEEK_API_KEY"]
else:
    st.error("⚠️ No se ha encontrado la API Key de DeepSeek en los Secrets.")
    st.stop()

# 3. PREPARACIÓN DE LA BIBLIOTECA (CON CACHÉ)
# @st.cache_resource hace que esto corra solo una vez al inicio
@st.cache_resource
def procesar_documentacion():
    with st.spinner("🔄 J.A.R.V.I.S. está indexando la normativa (esto ocurre solo una vez)..."):
        loader = PyPDFDirectoryLoader(".") 
        docs = loader.load()
        
        if not docs:
            st.warning("No se encontraron PDFs en la carpeta.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        
        # Usamos embeddings locales (gratis y rápidos) para no depender de la API para esto
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore

# Inicializar vectorstore
vectorstore = procesar_documentacion()

if vectorstore:
    retriever = vectorstore.as_retriever()
else:
    retriever = None

# 4. CONFIGURACIÓN DEL LLM (DeepSeek)
def get_llm():
    return ChatOpenAI(
        model='deepseek-chat', 
        openai_api_key=api_key, 
        openai_api_base='https://api.deepseek.com',
        temperature=0
    )

# 5. INTERFAZ DE CHAT (LÓGICA STREAMLIT)

# Inicializar historial si no existe
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capturar nueva entrada del usuario
if prompt := st.chat_input("Consulta la normativa (CTE, EHE...)..."):
    # Guardar y mostrar mensaje usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar respuesta
    with st.chat_message("assistant"):
        if not retriever:
            response = "⚠️ Error: La base de datos documental no está activa."
            st.markdown(response)
        else:
            with st.spinner("Analizando normativa..."):
                # Recuperar contexto
                docs = retriever.invoke(prompt)
                contexto = "\n\n".join([d.page_content for d in docs])
                
                # Prompt del sistema
                system_prompt = f"""
                Eres J.A.R.V.I.S., el ingeniero virtual senior de BIM Consulting Solutions SL.
                
                CONTEXTO NORMATIVO:
                {contexto}
                
                PREGUNTA DEL USUARIO:
                {prompt}
                
                INSTRUCCIONES:
                1. Responde basándote estrictamente en el contexto normativo.
                2. Cita siempre la norma y el artículo.
                3. NO INVENTES información. Si no está en el contexto, dilo.
                """
                
                # Llamada al LLM
                llm = get_llm()
                try:
                    full_response = llm.invoke(system_prompt).content
                    st.markdown(full_response)
                    # Guardar respuesta en historial
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"Error al conectar con DeepSeek: {e}")