import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

openai_api_key = 'OPENAI KEY'
with_streaming = True

def extract_splits_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)   
    return splits

def process_pdfs(uploaded_files):
    all_splits = []
    for uploaded_file in uploaded_files:
        with open(os.path.join("documents", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        pdf_path = os.path.join("documents", uploaded_file.name)
        single_doc_splits = extract_splits_from_pdf(pdf_path)
        all_splits.append(single_doc_splits)
    all_splits_flattened = [item for sublist in all_splits for item in sublist]
    return all_splits_flattened

def store_splits_in_faiss(splits, openai_api_key):
    DB_FAISS_PATH = 'vectorstore/db_faiss'
    embedder = OpenAIEmbeddings(api_key=openai_api_key)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embedder)
    vectorstore.save_local(DB_FAISS_PATH)

def load_knowledgeBase(openai_api_key):
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    DB_FAISS_PATH = 'vectorstore/db_faiss'
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db

def load_llm(openai_api_key, with_streaming=True):
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", 
                     temperature=0, max_tokens=500,
                     api_key=openai_api_key,
                     stream=with_streaming)
    return llm

def load_prompt():
    prompt = """ 
    You are an AI assistant with access to a collection of documents. Your task is to answer questions based on the information contained in these documents. 
    Please provide detailed and accurate answers, ensuring that the information is directly derived from the content of the documents. 
    If the answer cannot be found in the documents, let the user know that the information is not available.
    Return the answer in form of Markdown for formatting
    Given below is the context and question of the user.
    context = {context}
    question = {question}
    """
    prompt = ChatPromptTemplate.from_template(prompt)
    return prompt

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

st.sidebar.header("PDF Document Vector Database")
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if st.sidebar.button("Process PDFs") and uploaded_files:
    all_splits = process_pdfs(uploaded_files)
    store_splits_in_faiss(all_splits, openai_api_key)
    st.sidebar.success("PDFs processed and stored in vector database.")
    
    st.sidebar.subheader("Uploaded Files:")
    for uploaded_file in uploaded_files:
        st.sidebar.write(uploaded_file.name)

knowledgeBase = load_knowledgeBase(openai_api_key)
llm = load_llm(openai_api_key)
prompt = load_prompt()

st.title("Document Retrieval Assistant")
query = st.text_input("Enter your question:")
if st.button("Get Answer"):
    if query:
        similar_embeddings_for_references = knowledgeBase.similarity_search(query)
        similar_embeddings = FAISS.from_documents(documents=similar_embeddings_for_references, embedding=OpenAIEmbeddings(api_key=openai_api_key))
        retriever = similar_embeddings.as_retriever()
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        if with_streaming:
            collected_messages = []
            placeholder = st.empty()
            for chunk in rag_chain.stream(query):
                collected_message = chunk
                if collected_message:
                    collected_messages.append(collected_message)
                    current_text = ''.join(collected_messages)
                    placeholder.markdown(current_text)
            answer = ''.join(collected_messages)
        else:
            answer = rag_chain.invoke(query)
            st.markdown(answer)
    
        
        st.markdown("References:")
        all_ref = []
        for doc in similar_embeddings_for_references:
            all_ref.append(doc.metadata['source'])
        
        for ref in set(all_ref):
            st.markdown(f"- {ref}")
    else:
        st.warning("Please enter a question.")