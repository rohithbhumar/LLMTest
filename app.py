import streamlit as st
from PyPDF2 import PdfReader
import docx2txt
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms import HuggingFaceHub

import os
from openai_key import openai_key, hugging_face_api_key
from htmlTemplate import css, bot_template, user_template


def extract_pdf_text(documents):
    text = ""
    if documents:
        for doc in documents:
            # Extract filename from the UploadedFile object
            file_name = doc.name
            file_extension = os.path.splitext(file_name)[1].lower()
            if file_extension == ".pdf":
                pdf_reader = PdfReader(doc)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            elif file_extension == ".docx":
                docx_doc = docx2txt.process(doc)
                for paragraph in docx_doc:
                    text += paragraph
            else:
                raise ValueError("Unsupported file format")
    else:
        raise ValueError("No documents uploaded")
    return text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text_chunks = text_splitter.split_text(raw_text)
    return text_chunks


def get_vectorstore(text_chunks):
    os.environ['OPENAI_API_KEY'] = openai_key
    # os.environ['HUGGINGFACEHUB_API_TOKEN'] = hugging_face_api_key
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")  # pip install InstructorEmbedding sentence_transformers
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.2, "max_length":512})
    # llm = HuggingFaceHub(repo_id="google/gemma-7b")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_user_input(user_question):
    response = st.session_state.conversation({"question": user_question})
    # st.write(response)
    st.session_state.chat_history = response["chat_history"]
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="AI PDF app", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with your Docs")
    user_question = st.text_input("Ask a question about your PDFs or Word")
    if user_question:
        handle_user_input(user_question)

    # st.write(user_template.replace("{{MSG}}", "hello Human"), unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}", "hello AI"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Read your documents")
        doc_content = st.file_uploader("Upload your PDFs here and click on process", accept_multiple_files=True)
        doc_process_button = st.button("Process")
        if doc_process_button:
            with st.spinner("Processing.."):
                # get pdf text
                raw_text = extract_pdf_text(documents=doc_content)
                # st.write(raw_text)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text=raw_text)
                st.write(text_chunks)

                # create vector store(embeddings)
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()
