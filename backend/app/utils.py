import os
import logging
from . import images
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from dotenv import load_dotenv

# Configure logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = "faiss_index"
REBUILD_INDEX = os.getenv("REBUILD_INDEX", "false").lower() == "true"

# Ensure OPENAI_API_KEY is set in your environment variables
if not os.getenv("OPENAI_API_KEY"):
    logger.warning("OPENAI_API_KEY environment variable not set. OpenAI calls will fail.")


def get_relevant_images(page_numbers: list[int]) -> list[str]:
    """
    Get relevant image paths for the given page numbers.
    Args:
        page_numbers (list[int]): List of page numbers
    Returns:
        list[str]: List of image paths
    """
    try:
        with open("image_database.json", 'r') as f:
            image_database = json.load(f)
            
        relevant_images = []
        for page in page_numbers:
            if str(page) in image_database:  # Convert to str as JSON keys are strings
                relevant_images.extend(image_database[str(page)])
        
        return relevant_images
    except Exception as e:
        logger.error(f"Error retrieving relevant images: {e}")
        return []
    

async def setup_rag_pipeline(manual_file_path: str):
    """
    Sets up the RAG pipeline by loading the PDF, creating embeddings,
    and initializing the vector store and LLM.
    Args:
        manual_file_path (str): Path to the BMW X1 manual PDF.
    Returns:
        tuple: (ChatOpenAI instance, FAISS vector store instance)
    """
    logger.info(f"Starting RAG pipeline setup with manual: {manual_file_path}")

    # Initialize embeddings model first
    embeddings_model = OpenAIEmbeddings()
    logger.info("Initialized OpenAI embeddings model")

    if os.path.exists(FAISS_INDEX_PATH) and not REBUILD_INDEX:
        logger.info("Loading existing FAISS index...")
        try:
            vector_store = FAISS.load_local(
                FAISS_INDEX_PATH, 
                embeddings_model,
                allow_dangerous_deserialization=True  # Only use if you trust the source
            )
            logger.info("Successfully loaded existing FAISS index")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            raise
    else:
        # 1. Load the PDF document
        logger.info("Loading PDF document...")
        loader = PyPDFLoader(manual_file_path)
        try:
            documents = loader.load()
            if not documents:
                logger.error("No documents loaded from PDF. The PDF might be empty or unreadable.")
                raise ValueError("Failed to load documents from PDF.")
            logger.info(f"Successfully loaded {len(documents)} pages/documents from PDF.")
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            raise

        # 2. Split the document into manageable chunks
        logger.info("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        
        # Add validation for empty content
        chunks = [chunk for chunk in text_splitter.split_documents(documents) if chunk.page_content.strip()]
        if not chunks:
            logger.error("No valid chunks created from documents. Text splitting failed.")
            raise ValueError("Failed to split documents into valid chunks.")
        logger.info(f"Split documents into {len(chunks)} valid chunks.")

        logger.info("Extracting images from PDF...")
        try:
            images.extract_images_with_page_numbers(
                pdf_path=manual_file_path,
                output_dir="../data/images"
            )
            logger.info("Successfully extracted images from PDF")
        except Exception as e:
            logger.error(f"Error extracting images: {e}")

        
            
        # 3. Create embeddings in batches
        logger.info("Initializing OpenAI embeddings model...")
        try:
            embeddings_model = OpenAIEmbeddings()
            
            # Initialize the first batch properly
            first_batch = chunks[:50]
            if not first_batch:
                raise ValueError("No valid chunks to process")
                
            vector_store = FAISS.from_documents(first_batch, embeddings_model)
            
            # Process remaining chunks in batches
            for i in range(50, len(chunks), 50):
                batch = chunks[i:i + 50]
                if batch:  # Only process non-empty batches
                    logger.info(f"Processing batch {i//50 + 1} of {(len(chunks)-1)//50 + 1}")
                    try:
                        batch_vector_store = FAISS.from_documents(batch, embeddings_model)
                        vector_store.merge_from(batch_vector_store)
                    except Exception as e:
                        logger.warning(f"Failed to process batch {i//50 + 1}: {e}")
                        continue
                        
            logger.info("FAISS vector store created successfully.")
        except Exception as e:
            logger.error(f"Failed to create FAISS vector store: {e}")
            raise

        # After creating the vector_store, save it
        try:
                vector_store.save_local(FAISS_INDEX_PATH)
                logger.info("Saved FAISS index to disk")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            raise

    # 4. Initialize the LLM
    logger.info("Initializing ChatOpenAI model...")
    try:
        llm = ChatOpenAI(
            model_name="gpt-4.1-mini",
            temperature=0,
            max_tokens=2048
        )
    except Exception as e:
        logger.error(f"Failed to initialize ChatOpenAI model: {e}")
        raise
    
    # Initialize memory with proper error handling
    try:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output",
            system_message=SystemMessage(content="You are a helpful BMW X1 assistant. Answer questions based strictly on the manual.")
        )
        logger.info("Memory initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize memory: {e}")
        raise

    logger.info("RAG pipeline components (LLM, Vector Store, and Memory) initialized.")
    return llm, vector_store, memory

    
async def process_query_with_rag(query: str, llm: ChatOpenAI, vector_store: FAISS, memory: ConversationBufferMemory):
    """
    Processes the user's query using the RAG pipeline.
    Args:
        query (str): The user's question.
        llm (ChatOpenAI): The initialized language model.
        vector_store (FAISS): The initialized vector store.
        memory (ConversationBufferMemory): The conversation memory buffer.
    Returns:
        tuple: (str: The generated answer, list: page numbers referenced)
    """
    logger.info(f"Processing query: {query}")
    
    # Get chat history or empty list for first request
    chat_history = []
    if memory and hasattr(memory, 'chat_memory'):
        chat_history = memory.chat_memory.messages
        logger.info(f"Using conversation history with {len(chat_history)} messages")
    else:
        logger.info("First request: No conversation history available")

    # Define the prompt template
    # This prompt emphasizes answering strictly from the provided context and citing page numbers.
    # [cite: 3, 5, 8, 11]
    prompt_template = """
    You are an AI assistant for a BMW X1. You answer only in greek language. Your task is to answer the user's question based on the following context and chat history.
    
    Previous conversation:
    {chat_history}
    
    Current context from manual:
    ---
    {context}
    ---
    
    Answer the question using the provided context. If referring to previous conversation, still verify information with the manual context.
    If in the context there is a "Υπόδειξη" section, then give exactly that section as it was written in your answer and a warning sign and live a blanc line from above and below to make it clearer.
    At the end of your response, give to the user a relevant to your answer follow-up question (only one) to continue the conversation.
    If the answer cannot be found in the current context, answer "Δεν βρέθηκαν σχετικές πληροφορίες στο εγχειρίδιο." and then 
    think of what they might mean rellated to the context and their current question (only one thing) and complete the response with "Μηπώς εννοούσαται [ό,τι πιστεύεις ότι εννοούσαν]".
    If the question of the user down is something like "ναι or "πες μου" you have to answer to your latest follow up question as the current question.

    Current Question: {input}
    
    Answer:
    """
    prompt = PromptTemplate.from_template(prompt_template)

    # Create a retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 4}) # Retrieve top 4 relevant chunks

    # Create a "stuff" documents chain (passes all retrieved docs to the LLM)
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        document_variable_name="context",
    )

    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain,
    )

    try:
        # Add null check for memory
        chat_history = []
        if memory and hasattr(memory, 'chat_memory'):
            chat_history = memory.chat_memory.messages

        response = await retrieval_chain.ainvoke({
            "input": query,
            "chat_history": chat_history
        })
        
        # Add null check before saving context
        if memory:
            memory.save_context({"input": query}, {"output": response["answer"]})
            
    except Exception as e:
        logger.error(f"Error during retrieval chain invocation: {e}")
        return "Sorry, I encountered an error while processing your request.", []
    
    try:
        # Include memory in the chain invocation
        response = await retrieval_chain.ainvoke({
            "input": query,
            "chat_history": memory.chat_memory.messages
        })
        
        # Update memory with the new interaction
        memory.save_context({"input": query}, {"output": response["answer"]})
        
    except Exception as e:
        logger.error(f"Error during retrieval chain invocation: {e}")
        return "Sorry, I encountered an error while processing your request.", []

    answer = response.get("answer", "No answer could be generated.")
    
    # Extract page numbers from the context documents used for the answer
    # Note: PyPDFLoader adds 'page' (0-indexed) to metadata. We add 1 for user-friendly page numbers.
    page_numbers = []
    if "context" in response and response["context"]:
        seen_pages = set()
        for doc in response["context"]:
            if hasattr(doc, 'metadata') and 'page' in doc.metadata:
                page_num = doc.metadata['page'] + 1
                if page_num not in seen_pages:
                    seen_pages.add(page_num)
        page_numbers = sorted(list(seen_pages))

    relevant_images = get_relevant_images(page_numbers)
    
    logger.info(f"Generated answer: {answer}")
    logger.info(f"Relevant page(s) from manual: {page_numbers}")
    if relevant_images:
        logger.info(f"Found {len(relevant_images)} relevant images")


    logger.info(f"Generated answer: {answer}")
    if page_numbers:
         logger.info(f"Relevant page(s) from manual: {page_numbers}")
         # You might want to append this to the answer string if not already handled by the LLM.
         # For now, returning separately.
         # Example: " (Refer to page(s): 12, 15)" could be added to the answer string.

    return answer, page_numbers, relevant_images