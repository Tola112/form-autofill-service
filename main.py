import os
import openai
import sys
sys.path.append('../..')
import openai
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key  = os.environ['OPENAI_API_KEY']
embedding = OpenAIEmbeddings()

chunk_size =100
chunk_overlap = 10

#get the current working directory the python debugger script is running in, pdf files where previously not being found
working_directory = os.getcwd()

loaders = [
    PyPDFLoader("docs/TestCV.pdf"),
    PyPDFLoader("docs/ResearchPaper.pdf")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

#split the documents into chunks
splits = text_splitter.split_documents(docs)

#store the chunks in the vector database
vectordb = FAISS.from_documents(splits,embedding)

#query the vector database
docs_should_be_from_research_paper = vectordb.similarity_search("what was spoken about debris removal?", k=5)
print(docs_should_be_from_research_paper)
docs_should_be_from_test_cv = vectordb.similarity_search("what was the candidate experience?", k=5)
print(docs_should_be_from_test_cv)