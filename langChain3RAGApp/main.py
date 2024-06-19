import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

load_dotenv()

def main():
    
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
        
      # split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text)
      
      # create embeddings
      embeddings = OpenAIEmbeddings()
      knowledge_base = FAISS.from_texts(chunks, embeddings)
      
      # show user input
      user_question = st.text_input("Ask a question about your PDF:")
      if user_question:
        docs = knowledge_base.similarity_search(user_question)
        
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
          response = chain.run(input_documents=docs, question=user_question)
          print(cb)
           
        st.write(response)
    

if __name__ == '__main__':
    main()


# from dotenv import load_dotenv
# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains import load_qa_chain
# from langchain.llms import OpenAI
# from langchain.callbacks import get_openai_callback

# def setupPage():
#     """Setup Streamlit page configuration."""
#     load_dotenv()
#     st.set_page_config(page_title="Ask your PDF")
#     st.header("Ask your PDF 💬")

# def uploadPdf():
#     """Upload a PDF file via Streamlit uploader."""
#     return st.file_uploader("Upload your PDF", type="pdf")

# def extractTextPdf(pdf):
#     """Extract text from the uploaded PDF."""
#     pdf_reader = PdfReader(pdf)
#     text = ""
#     for page in pdf_reader.pages:
#         text += page.extract_text()
#     return text

# def splitTexttoChunks(text):
#     """Split text into manageable chunks."""
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     return text_splitter.split_text(text)

# def createEmbeddings(chunks):
#     """Create embeddings for the text chunks."""
#     openai_api_key = os.getenv('OPENAI_API_KEY')
#     if not openai_api_key:
#         raise ValueError("OpenAI API key not found in environment variables.")
#     embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#     return FAISS.from_texts(chunks, embeddings)

# def getUserQuestion():
#     """Get a question from the user."""
#     return st.text_input("Ask a question about your PDF:")

# def generateResponse(user_question, knowledge_base):
#     """Generate a response to the user's question."""
#     docs = knowledge_base.similarity_search(user_question)
#     llm = OpenAI()
#     chain = load_qa_chain(llm, chain_type="stuff")
#     with get_openai_callback() as cb:
#         response = chain.run(input_documents=docs, question=user_question)
#         print(cb)
#     return response

# def main():
#     """Main function to orchestrate the app workflow."""
#     setupPage()
#     pdf = uploadPdf()
    
#     if pdf is not None:
#         text = extractTextPdf(pdf)
#         chunks = splitTexttoChunks(text)
#         knowledge_base = createEmbeddings(chunks)
        
#         user_question = getUserQuestion()
#         if user_question:
#             response = generateResponse(user_question, knowledge_base)
#             st.write(response)

# if __name__ == '__main__':
#     main()