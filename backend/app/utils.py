import os
import logging
import asyncio
import requests
from dotenv import load_dotenv
from duckduckgo_search import DDGS  # ✅ Επιστροφή στο ddg()

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load .env and configure logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not os.getenv("OPENAI_API_KEY"):
    logger.warning("OPENAI_API_KEY environment variable not set. OpenAI calls will fail.")

# ✅ ΕΠΙΣΤΡΟΦΗ ΣΕ ddg() ΜΕ ΑΣΦΑΛΗ ΕΛΕΓΧΟ
async def find_manual_pdf_url(car_model: str):
    logger.info(f"Searching online for manual for: {car_model}")
    query = f"{car_model} owner's manual filetype:pdf"
    
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query)
            for r in results:
                href = r.get("href", "")
                if href.endswith(".pdf"):
                    logger.info(f"Found PDF URL: {href}")
                    return href
    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {e}")

    logger.warning("No PDF URL found.")
    return None


def download_pdf(url, save_path):
    logger.info(f"Downloading PDF from: {url}")
    response = requests.get(url)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(response.content)
        logger.info(f"Saved manual to: {save_path}")
        return True
    logger.error("Failed to download PDF.")
    return False

async def auto_load_car_manual(car_model: str):
    pdf_url = await find_manual_pdf_url(car_model)
    if not pdf_url:
        return None, None
    local_path = f"./manuals/{car_model.replace(' ', '_')}.pdf"
    if not download_pdf(pdf_url, local_path):
        return None, None
    return await setup_rag_pipeline(local_path)

async def setup_rag_pipeline(manual_file_path: str):
    logger.info(f"Starting RAG pipeline setup with manual: {manual_file_path}")
    loader = PyPDFLoader(manual_file_path)
    try:
        documents = loader.load()
        if not documents:
            logger.error("No documents loaded from PDF.")
            raise ValueError("Failed to load documents from PDF.")
        logger.info(f"Successfully loaded {len(documents)} pages/documents from PDF.")
    except Exception as e:
        logger.error(f"Error loading PDF: {e}")
        raise

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = [chunk for chunk in text_splitter.split_documents(documents) if chunk.page_content.strip()]
    if not chunks:
        logger.error("No valid chunks created from documents.")
        raise ValueError("Failed to split documents into valid chunks.")
    logger.info(f"Split documents into {len(chunks)} valid chunks.")

    try:
        embeddings_model = OpenAIEmbeddings()
        first_batch = chunks[:50]
        vector_store = FAISS.from_documents(first_batch, embeddings_model)
        for i in range(50, len(chunks), 50):
            batch = chunks[i:i + 50]
            if batch:
                logger.info(f"Processing batch {i//50 + 1}")
                try:
                    batch_vector_store = FAISS.from_documents(batch, embeddings_model)
                    vector_store.merge_from(batch_vector_store)
                except Exception as e:
                    logger.warning(f"Batch failed: {e}")
        logger.info("FAISS vector store created.")
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise

    try:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) 
    except Exception as e:
        logger.error(f"Failed to initialize ChatOpenAI: {e}")
        raise

    return llm, vector_store

async def process_query_with_rag(query: str, llm: ChatOpenAI, vector_store: FAISS):
    logger.info(f"Processing query: {query}")
    prompt_template = """
    You are a helpful assistant trained on car manuals. Use the following excerpts from the manual to answer the user's question.
    - Answer clearly and helpfully.
    - If you find relevant instructions in the text, summarize them.
    - If the answer isn't clearly stated, you may use your general knowledge to fill in gaps cautiously.

    Manual excerpts:
    ---
    {context}
        ---
        User question: {input}

        Answer:
        """
    prompt = PromptTemplate.from_template(prompt_template)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    try:
        response = await retrieval_chain.ainvoke({"input": query})
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        return "Sorry, an error occurred.", []

    answer = response.get("answer", "No answer could be generated.")
    page_numbers = []
    if "context" in response and response["context"]:
        seen_pages = set()
        for doc in response["context"]:
            if hasattr(doc, 'metadata') and 'page' in doc.metadata:
                seen_pages.add(doc.metadata['page'] + 1)
        page_numbers = sorted(seen_pages)

    logger.info(f"Answer: {answer}")
    if page_numbers:
        logger.info(f"Pages: {page_numbers}")
    return answer, page_numbers

# CLI λειτουργία
if __name__ == "__main__":
    async def main():
        car_model = input("🔧 Enter your car model (e.g., BMW X1 2020): ")
        llm, vector_store = await auto_load_car_manual(car_model)
        if not llm or not vector_store:
            print("❌ Failed to load manual.")
            return
        while True:
            question = input("\n❓ Ask a question about the manual (or type 'exit'): ")
            if question.lower() == "exit":
                break
            answer, pages = await process_query_with_rag(question, llm, vector_store)
            print(f"\n📘 Answer: {answer}")
            if pages:
                print(f"📄 Referenced page(s): {pages}")
    asyncio.run(main())