
import streamlit as st
from streamlit_modal import Modal

def rag_feedback(student_result):
    import os
    from langchain_community.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.prompts import ChatPromptTemplate
    from langchain_community.chat_models import ChatOpenAI
    DOC_PATH = "knowledge.txt"

# Set up OpenAI API key
    OPENAI_API_KEY = st.secrets["OpenAI_key"]

# ----- Data Indexing Process -----

# load your text file
    loader = TextLoader(DOC_PATH)
    pages = loader.load()

    # split the doc into smaller chunks i.e. chunk_size=500
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = text_splitter.split_documents(pages)

    # get OpenAI Embedding model
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # embed the chunks as vectors and load them into the database
    db_faiss = FAISS.from_documents(chunks, embeddings)

    # retrieve context - top 5 most relevant (closest) chunks to the query vector
    docs_faiss = db_faiss.similarity_search(student_result, k=5)

    # generate an answer based on given user query and retrieved context information
    context_text = "\n\n".join([doc.page_content for doc in docs_faiss])

    # load retrieved context and user query in the prompt template
    PROMPT_TEMPLATE = """
    Please provide feedback on the following music theory result based on the given context:
    {context}
    Music theory result: {student_result}
    Provide a detailed feedback.
    Don't justify your answers.
    Do not say "according to the context" or "mentioned in the context" or similar.
    Feedback:
    """
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, student_result=student_result)

    # call LLM model to generate the feedback based on the given context and student result
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    feedback = model.predict(prompt)
    return feedback
    
pw = st.secrets["Password"]

if "login" not in st.session_state:
    st.session_state["login"] = False
    st.session_state["pw"] = ""

def login_button_clicked():
    if st.session_state["pw"] in pw:
        st.session_state["login"] = True
    else:
        st.error("Wrong password")

if st.session_state["login"] == False:
    with st.popover(label="Login"):
        with st.form(key="login_form"):
            st.session_state["pw"] = st.text_input("Password", key="pwinput", type="password")
            st.form_submit_button("OK", on_click=login_button_clicked)
elif st.session_state["login"]:
    st.write("You are logged in!")

# Streamlit app code
st.title("Music Theory Feedback")
if st.session_state["login"]:
    with st.popover("Chat with AI",use_container_width=True):
        prompt = st.chat_input("Ask me anything you want to know about music theory:")
        if prompt:
            st.write(f"User: {prompt}")
            feedback = rag_feedback(prompt)
            st.write(f"AI: {feedback}")
            time.sleep(5)

