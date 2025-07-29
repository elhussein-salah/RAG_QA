import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import faiss
import pickle
import os
from typing import List, Tuple
from data_preparation import load_dataset, prepare_contexts_for_embedding

class ArabicEmbedding: 
    def __init__(self, model_name: str = "aubmindlab/bert-base-arabertv2"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.index = None
        self.contexts = None
        
    def load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def encode_text(self, text: str) -> np.ndarray:
        if self.model is None or self.tokenizer is None:
            self.load_model()
            
        # Tokenize and encode
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use the [CLS] token embedding as the sentence representation
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
        return embeddings.flatten()
    
    def encode_texts(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        if self.model is None or self.tokenizer is None:
            self.load_model()
            
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use the [CLS] token embedding as the sentence representation
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all embeddings
        embeddings_matrix = np.vstack(all_embeddings)
        
        return embeddings_matrix
    
    def build_faiss_index(self, embeddings: np.ndarray) -> None:
        dimension = embeddings.shape[1]
        
        # Create FAISS index (using IndexFlatIP for cosine similarity)
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings.astype(np.float32))
    
    def save_index_and_contexts(self, index_path: str, contexts_path: str, contexts: List[str]) -> None:
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save contexts
        with open(contexts_path, 'wb') as f:
            pickle.dump(contexts, f)
        
        self.contexts = contexts
    
    def load_index_and_contexts(self, index_path: str, contexts_path: str) -> None:
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load contexts
        with open(contexts_path, 'rb') as f:
            self.contexts = pickle.load(f)
    
    def create_embeddings_and_index(self, contexts: List[str], save_dir: str = "embeddings") -> None:
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate embeddings
        embeddings = self.encode_texts(contexts)
        
        # Build FAISS index
        self.build_faiss_index(embeddings)
        
        # Save index and contexts
        index_path = os.path.join(save_dir, "faiss_index.index")
        contexts_path = os.path.join(save_dir, "contexts.pkl")
        self.save_index_and_contexts(index_path, contexts_path, contexts)

def main():
    
    # Load dataset
    df = load_dataset()
    contexts = prepare_contexts_for_embedding(df)
    
    # Initialize embedding model
    embedding_model = ArabicEmbedding()
    
    # Create embeddings and index
    embedding_model.create_embeddings_and_index(contexts)
    
    print("Embeddings and FAISS index created successfully!")

if __name__ == "__main__":
    main()
