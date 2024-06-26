import time
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings

# Function to provide feedback based on student's music theory results
def rag_feedback(student_result):
    from langchain_community.vectorstores import FAISS
    from langchain.prompts import ChatPromptTemplate
    from langchain_community.chat_models import ChatOpenAI

    INDEX_PATH = "faiss_index"
    OPENAI_API_KEY = st.secrets["OpenAI_key"]

    # Create an Embeddings object
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    # Load the precomputed FAISS index from disk with the embeddings object
    db_faiss = FAISS.load_local(INDEX_PATH, embeddings=embeddings, allow_dangerous_deserialization=True)
    st.write("Getting knowledge at database.")
    print("Getting knowledge at database.")
    docs_faiss = db_faiss.similarity_search(student_result, k=5)

    # Generate an answer based on given user query and retrieved context information
    context_text = "\n\n".join([doc.page_content for doc in docs_faiss])

    # Load retrieved context and user query in the prompt template
    PROMPT_TEMPLATE = """
    You are a music theory teacher. Please provide feedback or answer questions about music theory based on the given context:
    {context}
    Music theory result: {student_result}
    Provide short feedback.
    Do not say "according to the context" or "mentioned in the context" or similar.
    Feedback:
    """
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, student_result=student_result)

    # Call LLM model to generate the feedback based on the given context and student result
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    feedback = model.predict(prompt)
    return feedback

# Retrieve the password from Streamlit secrets
pw = st.secrets["Password"]

# Initialize session state variables for login
if "login" not in st.session_state:
    st.session_state["login"] = False
    st.session_state["pw"] = ""

# Function to handle login button click
def login_button_clicked():
    if st.session_state["pw"] == pw:
        st.session_state["login"] = True
    else:
        st.error("Wrong password")

# Login form
if not st.session_state["login"]:
    with st.popover(label="Login"):
        with st.form(key="login_form"):
            st.session_state["pw"] = st.text_input("Password", key="pwinput", type="password")
            st.form_submit_button("OK", on_click=login_button_clicked)
elif st.session_state["login"]:
    st.write("You are logged in!")

# Main Streamlit app code
st.title("Music Theory Feedback")
if st.session_state["login"]:
    with st.popover("Chat with AI", use_container_width=True):
        prompt = st.chat_input("Ask me anything you want to know about music theory:")
        if prompt:
            st.write(f"User: {prompt}")
            feedback = rag_feedback(prompt)
            st.write(f"AI: {feedback}")
            time.sleep(5)
