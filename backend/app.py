import os
import numpy as np
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import json
import traceback
import faiss
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

class RAGSystem:
    def __init__(self):
        try:
            model_name = os.getenv("SENTENCE_TRANSFORMERS")
            print(f"Loading model: {model_name}")
            self.model = SentenceTransformer(model_name, cache_folder="/tmp/.cache/sentence_transformers")
            
            # Check if vector store exists
            index_path = "data/vector_store/uk_nhs_index.faiss"
            metadata_path = "data/vector_store/uk_nhs_index_metadata.json"

            if os.path.exists(index_path) and os.path.exists(metadata_path):
                self.index = faiss.read_index(index_path)
                with open(metadata_path, "r") as f:
                    self.metadata = json.load(f)
                print(f"Loaded FAISS index with {len(self.metadata)} documents.")
            else:
                print("Warning: FAISS index not found, creating empty index")
                self.index = None
                self.metadata = []
        except Exception as e:
            print(f"Error initializing RAG system: {e}")
            traceback.print_exc()
            raise
    
    def find_similar(self, query, k=2):
        if self.index is None or len(self.metadata) == 0:
            print("No vector store available, returning empty results")
            return []
        
        try:
            query = np.array(query).astype('float32').reshape(1, -1)
            distances, indices = self.index.search(query, k)
            return [self.metadata[i] for i in indices[0]]
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
        
    def encode_query(self, question):
        try:
            return self.model.encode(question).tolist()
        except Exception as e:
            print(f"Error encoding query: {e}")
            return []

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

# Initialize RAG system
try:
    rag_system = RAGSystem()
    print("RAG system initialized successfully")
except Exception as e:
    print(f"Failed to initialize RAG system: {e}")
    rag_system = None

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")

class Query(BaseModel):
    question: str

def query_hf(context: str, question: str):
    try:
        client = InferenceClient(token=HF_TOKEN)
        completion = client.chat.completions.create(
            model=HF_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Answer using ONLY the NHS/GOV documents provided.",
                },
                {
                    "role": "user",
                    "content": f"Documents:\n{context}\n\nQuestion: {question}",
                },
            ],
            max_tokens=512,
            temperature=0.7,
        )
        return completion
    except TimeoutError:
        return {"error": "Request timeout"}
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

def extract_answer(response):
    try:
        if not response:
            return "No response received from AI model"

        if isinstance(response, dict) and "error" in response:
            return f"AI model error: {response['error']}"

        content = getattr(getattr(response.choices[0], "message", None), "content", None)
        if not content:
            return "No text generated"
        return content.strip()
    except Exception as e:
        return f"Error processing response: {str(e)}"

@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    try:
        status = "healthy" if rag_system is not None else "unhealthy"
        return {
            "status": status,
            "message": "Backend is running",
            "rag_initialized": rag_system is not None,
            "documents_loaded": len(rag_system.metadata) if rag_system else 0
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/ask")
async def ask(query: Query):
    try:
        # Validate inputs
        if not query.question or not query.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        if not HF_TOKEN:
            raise HTTPException(status_code=500, detail="Hugging Face token is not configured")
        
        if not rag_system:
            raise HTTPException(status_code=500, detail="RAG system is not initialized")
        
        # Process query
        print(f"Processing query: {query.question}")
        
        query_embed = rag_system.encode_query(query.question)
        if not query_embed:
            raise HTTPException(status_code=500, detail="Failed to encode query")
        
        similar_docs = rag_system.find_similar(query_embed)
        
        if not similar_docs:
            context = "No relevant documents found."
            sources = []
        else:
            context = "\n".join([f"Source: {doc.get('source', 'Unknown')}, Page: {doc.get('page', 'Unknown')}\n{doc.get('text', '')}" for doc in similar_docs])
            sources = [{"source": doc.get("source", "Unknown"), "page": doc.get("page", "Unknown")} for doc in similar_docs]

        hf_response = query_hf(context, query.question)
        answer = extract_answer(hf_response)
        
        return {
            "answer": answer,
            "source": sources,
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /ask endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
