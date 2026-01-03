import os
#import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Configuración de la página
st.set_page_config(page_title="JARVIS BIM", page_icon="🤖")

# 1. CONFIGURACIÓN DE SEGURIDAD
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("¡ERROR! No se ha encontrado la variable DEEPSEEK_API_KEY en los Secrets.")

# 2. PREPARACIÓN DE LA BIBLIOTECA (RAG)
def procesar_documentacion():
    print("🔄 J.A.R.V.I.S. está indexando la normativa...")
    loader = PyPDFDirectoryLoader(".") 
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    
    # Usamos embeddings de DeepSeek (compatible OpenAI)
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key, 
        openai_api_base="https://api.deepseek.com",
        check_embedding_ctx_length=False
    )
    
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

# Inicializamos la base de datos al arrancar
try:
    vectorstore = procesar_documentacion()
    retriever = vectorstore.as_retriever()
except Exception as e:
    print(f"Error indexando: {e}")
    retriever = None

# ASÍ SE CONECTA A DEEPSEEK USANDO LOS SECRETOS DE STREAMLIT
def get_llm():
    return ChatOpenAI(
        model='deepseek-chat', 
        openai_api_key=st.secrets["DEEPSEEK_API_KEY"], 
        openai_api_base='https://api.deepseek.com', # <--- Esto redirige a DeepSeek
        temperature=0
    )

st.title("🤖 JARVIS BIM - DeepSeek Edition")

# 4. EL SISTEMA DE RESPUESTA
def consultar_jarvis(mensaje, historia):
    if not retriever:
        return "⚠️ Error: No se han podido cargar los documentos PDF. Revisa los logs."

    # Buscamos contexto relevante
    docs = retriever.invoke(mensaje)
    contexto = "\n\n".join([d.page_content for d in docs])

    system_prompt = f"""
    Eres J.A.R.V.I.S., el ingeniero virtual senior de BIM Consulting Solutions SL.
    
    TU MISIÓN:
    Ser un consultor normativo (CTE, Código Estructural, Eurocódigos).
    
    DIRECTIVA DE SEGURIDAD Nº 1:
    NO REALICES CÁLCULOS MATEMÁTICOS NI DIMENSIONAMIENTOS.
    Si el usuario pide un cálculo, explica la normativa aplicable y DERÍVALO a la "Suite de Cálculo de BIM Consulting Solutions".
    
    CONTEXTO NORMATIVO RECUPERADO:
    {contexto}
    
    Instrucciones de respuesta:
    1. Responde basándote estrictamente en el contexto normativo anterior.
    2. Cita siempre la norma y el artículo (Ej: DB-SE-C Art 4.1).
    3. Si el contexto no tiene la respuesta, dilo honestamente.
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": mensaje}
    ]
    
    response = llm.invoke(messages)
    return response.content

# 5. LA INTERFAZ
interfaz = gr.ChatInterface(
    fn=consultar_jarvis,
    title="J.A.R.V.I.S. - Structural Core",
    description="Asistente de Normativa (CTE / CE / Eurocódigos) de BIM Consulting Solutions.",
    theme="soft",
    examples=["Recubrimiento mínimo en ambiente marino según Código Estructural", "Sobrecarga de uso en hospitales CTE DB-SE-AE"],
    submit_btn="Analizar Normativa",
)

if __name__ == "__main__":
    interfaz.launch(server_name="0.0.0.0", server_port=7860)