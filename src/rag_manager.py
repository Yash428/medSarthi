import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

RESOURCES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources")
FAISS_INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db", "faiss_indexes")

# Use a fast local embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_specialty_retriever(specialty: str):
    """
    Given a specialty (e.g. 'cardiology', 'neurology'), parses documents
    in resources/{specialty} and returns a FAISS retriever.
    Caches the FAISS index locally.
    """
    specialty = specialty.lower()
    index_path = os.path.join(FAISS_INDEX_DIR, specialty)
    
    # Return cached FAISS if exists
    if os.path.exists(index_path):
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        return vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Otherwise build it
    target_dir = os.path.join(RESOURCES_DIR, specialty)
    
    # If the directory doesn't exist, create it with a dummy text file
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        with open(os.path.join(target_dir, "guidelines.txt"), "w") as f:
            f.write(f"The patient should consult standard {specialty} literature. "
                    f"This is a placeholder for {specialty} resources. "
                    "For emergencies, call an ambulance.")

    docs = []
    for filename in os.listdir(target_dir):
        file_path = os.path.join(target_dir, filename)
        if filename.endswith(".txt") or filename.endswith(".md"):
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                docs.extend(loader.load())
            except:
                pass
        elif filename.endswith(".pdf"):
            try:
                loader = PyPDFLoader(file_path)
                docs.extend(loader.load())
            except:
                pass
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    # If folder is totally empty except non-parsable files, supply a dummy chunk
    if not splits:
        with open(os.path.join(target_dir, "dummy.txt"), "w") as f:
            f.write(f"No specific resources uploaded yet for {specialty}.")
        loader = TextLoader(os.path.join(target_dir, "dummy.txt"), encoding='utf-8')
        splits = text_splitter.split_documents(loader.load())
        
    vector_store = FAISS.from_documents(splits, embeddings)
    
    # Save for future so we don't rebuild every request
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    vector_store.save_local(index_path)
    
    return vector_store.as_retriever(search_kwargs={"k": 3})
