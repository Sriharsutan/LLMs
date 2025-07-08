import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS, Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain



def get_response(user_input):
    llm, retrival_chain = get_context_chain(st.session_state.vector_store)

    conversational_rag_chain = get_conversationa_rag_chain(llm, retrival_chain)

    response = conversational_rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_input
        })
    print(response)
    return response["answer"]

def get_vector_store(url):
    loader = WebBaseLoader(url)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc_chunks = text_splitter.split_documents(document)

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    vector_store = Chroma.from_documents(doc_chunks, embeddings)
    return vector_store


def get_context_chain(vector_store):
    model_name = "Qwen/Qwen1.5-1.8B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    hf_pipe = pipeline("text-generation", tokenizer=tokenizer, model=model, temperature=0.6)

    llm = HuggingFacePipeline(pipeline=hf_pipe)
    
    retriever = vector_store.as_retriever(search_kwargs={"k":4})
    

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up inorder to get information relevant to the conversation")
    ])

    retiever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return llm, retiever_chain

def get_conversationa_rag_chain(llm, retrieval_chain):
    prompt = ChatPromptTemplate.from_messages([
        #("system", "Answer the user's question based on the given context:\n\n {context}"),
        ("system", "You are a helpful assistant. Use the provided context to answer the user's question. Keep your answer concise and avoid copying the context.\nContext: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    stuffed_doc_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retrieval_chain, stuffed_doc_chain)


st.set_page_config(page_title="Web-chat")
st.title("Chat-with-website")
st.write("Enter the URL in the sidebar and ask the related questions below in the textbox")


with st.sidebar:
    st.header("Settings")
    web_page = st.text_input("Website URL")

if web_page is None or web_page=="":
        st.info("Please enter a website URL")
    
else:

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, How can I help you.")
        ]
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vector_store(web_page)

    
    user_input = st.chat_input("Enter your questions here")

    if user_input:
        response = get_response(user_input)
        #st.write(response)
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=response))

        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("ai"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("human"):
                    st.write(message.content)