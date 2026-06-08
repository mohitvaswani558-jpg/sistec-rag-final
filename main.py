import sys
import os
from dotenv import load_dotenv
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

def load_pdf_text(pdf_path):
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
    return text

def create_vectorstore(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def setup_qa_chain(vectorstore):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are Study Buddy, a strict document-based Q&A assistant built for students.

Your ONLY job is to answer questions based on the content of the PDF document uploaded by the user.

STRICT RULES YOU MUST FOLLOW:

1. ONLY use information from the uploaded document to answer questions.
   - Do NOT use your general training knowledge.
   - Do NOT make assumptions beyond what the document says.

2. OUT-OF-SCOPE REJECTION:
   - If a question cannot be answered from the document, respond exactly with:
     "❌ This question is outside the scope of the uploaded document. I can only answer questions based on its content."
   - Never attempt to guess or supplement with outside knowledge.

3. CONCISE ANSWERS:
   - Keep answers clear, direct, and to the point.
   - Use bullet points or numbered lists when explaining multi-step concepts.
   - Avoid unnecessary filler or repetition.

5. DOCUMENT FAITHFULNESS:
   - Never paraphrase in a way that changes the meaning.
   - If the document uses specific terminology, preserve it exactly.

6. HANDLING AMBIGUOUS QUESTIONS:
   - If a question is vague but answerable from the document, answer it and note your interpretation.
   - If it is too vague to answer reliably, ask the user to clarify.

Context: {context}

Question: {question}

Answer:
"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain, vectorstore

def answer_question(chain, vectorstore, question):
    docs = vectorstore.similarity_search(question, k=5)
    context = '\n'.join([doc.page_content for doc in docs])
    
    answer = chain.run(context=context, question=question).strip()
    
    source_chunks = [doc.page_content for doc in docs]
    
    return answer, source_chunks

def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py <pdf_path> <question>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    question = " ".join(sys.argv[2:])
    
    if not os.path.exists(pdf_path):
        print("PDF file not found.")
        sys.exit(1)
    
    print("Loading PDF...")
    text = load_pdf_text(pdf_path)
    if not text.strip():
        print("No text extracted from PDF.")
        sys.exit(1)
    
    print("Creating vector store...")
    vectorstore = create_vectorstore(text)
    
    print("Setting up QA chain...")
    chain, vectorstore = setup_qa_chain(vectorstore)
    
    print("Answering question...")
    answer, sources = answer_question(chain, vectorstore, question)
    
    print("✅ Answer:", answer)
    if sources:
        print("📄 Source:")
        for i, chunk in enumerate(sources, 1):
            print(f"- Chunk {i}: \"{chunk}\"")
    else:
        print("📄 Source: No sources found.")

if __name__ == "__main__":
    main()
