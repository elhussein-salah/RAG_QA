import numpy as np
import faiss
from typing import List, Tuple
from embedding import ArabicEmbedding
import os


class ContextRetriever:    
    def __init__(self, embedding_model: ArabicEmbedding = None):
        self.embedding_model = embedding_model or ArabicEmbedding()
        
    def load_index(self, index_path: str, contexts_path: str) -> None:
        self.embedding_model.load_index_and_contexts(index_path, contexts_path)
    
    def retrieve_contexts(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        if self.embedding_model.index is None or self.embedding_model.contexts is None:
            raise ValueError("Index and contexts must be loaded first")
        
        # Generate embedding for the query
        query_embedding = self.embedding_model.encode_text(query)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search in the index
        similarities, indices = self.embedding_model.index.search(query_embedding, top_k)
        
        # Prepare results
        retrieved_contexts = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.embedding_model.contexts):
                context = self.embedding_model.contexts[idx]
                retrieved_contexts.append((context, float(similarity)))
        
        return retrieved_contexts
    
    def retrieve_context_text(self, query: str, top_k: int = 3) -> str:
        retrieved_contexts = self.retrieve_contexts(query, top_k)
        
        if not retrieved_contexts:
            return "لم يتم العثور على سياق ذي صلة."
        
        # Combine contexts
        combined_context = "\n\n".join([context for context, _ in retrieved_contexts])
        
        return combined_context
    
    def retrieve_with_metadata(self, query: str, top_k: int = 3) -> List[dict]:
        retrieved_contexts = self.retrieve_contexts(query, top_k)
        
        results = []
        for rank, (context, similarity) in enumerate(retrieved_contexts, 1):
            results.append({
                'rank': rank,
                'context': context,
                'similarity': similarity,
                'relevance': 'عالي' if similarity > 0.8 else 'متوسط' if similarity > 0.6 else 'منخفض'
            })
        
        return results

def main():
    
    # Initialize retriever
    retriever = ContextRetriever()
    
    # Load index (assuming it exists)
    index_path = "embeddings/faiss_index.index"
    contexts_path = "embeddings/contexts.pkl"
    
    if os.path.exists(index_path) and os.path.exists(contexts_path):
        retriever.load_index(index_path, contexts_path)
        
        # Test retrieval
        test_queries = [
            "ما هو الذكاء الاصطناعي؟",
            "أين تقع جمهورية مصر العربية؟",
            "متى تم بناء الهرم الأكبر؟"
        ]
        
        for query in test_queries:
            print(f"\nاستعلام: {query}")
            print("-" * 50)
            
            # Retrieve contexts
            contexts = retriever.retrieve_with_metadata(query, top_k=2)
            
            for result in contexts:
                print(f"الترتيب: {result['rank']}")
                print(f"التشابه: {result['similarity']:.4f}")
                print(f"الصلة: {result['relevance']}")
                print(f"السياق: {result['context'][:100]}...")
                print()
    else:
        print("FAISS index not found. Please run embedding.py first to create the index.")

if __name__ == "__main__":
    main()
