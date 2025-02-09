from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import faiss


import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyAMXV-2RA3q5uICgpI7FfKrzjYzb2P7fFI"

PdfReader = PdfReader('Data Science.pdf')

from typing_extensions import Concatenate

raw_text=''
for i, page in enumerate(PdfReader.pages):
    content = page.extract_text()
    if content:
        raw_text += content
        
text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size = 800,
    chunk_overlap = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
from langchain_community.vectorstores.faiss import FAISS
document_search = FAISS.from_texts(texts, embeddings)
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.6)

prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
query = " NER stands for " # look at page 10 in the pdf
docs = document_search.similarity_search(query)
chain.run(input_documents = docs, question = query)