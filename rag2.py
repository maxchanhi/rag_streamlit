
import streamlit as st
def rag_feedback(student_result):
    import os
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.prompts import ChatPromptTemplate
    from langchain_community.chat_models import ChatOpenAI
    DOC_PATH = "/Users/chakhangchan/Documents/VS_code/Music_theory_app/AI_feedback/instrument_knowledge_quiz/knowledge.txt"

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

# Streamlit app code
st.title("Music Theory Feedback")

# Get student result input
student_result = st.text_area("Enter the student's music theory result:")

if st.button("Generate Feedback"):
    # Provide feedback using the retrieval and generation process
    feedback = rag_feedback(student_result)
    st.write("Feedback:")
    st.write(feedback)
