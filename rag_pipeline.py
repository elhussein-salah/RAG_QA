import os
from typing import Dict, Any, Optional
from data_preparation import load_dataset, prepare_contexts_for_embedding
from embedding import ArabicEmbedding
from retrieval import ContextRetriever
from llm_generation import OllamaLLMGenerator

class RAGPipeline:
    def __init__(self, 
                 model_name: str = "aubmindlab/bert-base-arabertv2",
                 llm_model: str = "gemma3:1b",
                 embeddings_dir: str = "embeddings"):

        self.model_name = model_name
        self.llm_model = llm_model
        self.embeddings_dir = embeddings_dir
        
        # Initialize components
        self.embedding_model = ArabicEmbedding(model_name)
        self.retriever = ContextRetriever(self.embedding_model)
        self.llm_generator = OllamaLLMGenerator(llm_model)
        
        self.is_initialized = False
    
    def initialize(self, dataset_path: str = None, force_rebuild: bool = False) -> bool:
        try:
            index_path = os.path.join(self.embeddings_dir, "faiss_index.index")
            contexts_path = os.path.join(self.embeddings_dir, "contexts.pkl")
            
            # Check if index exists and force_rebuild is False
            if not force_rebuild and os.path.exists(index_path) and os.path.exists(contexts_path):
                self.retriever.load_index(index_path, contexts_path)
            else:
                # Load dataset
                df = load_dataset(dataset_path)
                contexts = prepare_contexts_for_embedding(df)
                
                # Create embeddings and index
                self.embedding_model.create_embeddings_and_index(contexts, self.embeddings_dir)
                
                # Load the created index
                self.retriever.load_index(index_path, contexts_path)
            
            # Check LLM availability
            self.llm_generator.ensure_model_ready()
            
            self.is_initialized = True
            return True
        except Exception as e:
            return False
    
    def answer_question(self, 
                       question: str, 
                       top_k: int = 3, 
                       max_tokens: int = 256, 
                       temperature: float = 0.7) -> Dict[str, Any]:
        if not self.is_initialized:
            return {
                'answer': 'النظام غير مهيأ بعد. يرجى تشغيل initialize() أولاً.',
                'error': True,
                'error_message': 'Pipeline not initialized'
            }
        
        # Step 1: Retrieve relevant contexts
        retrieved_contexts = self.retriever.retrieve_with_metadata(question, top_k)
        
        if not retrieved_contexts:
            return {
                'answer': 'لم يتم العثور على سياق ذي صلة بالسؤال.',
                'question': question,
                'retrieved_contexts': [],
                'context_used': '',
                'error': False
            }
        
        # Step 2: Combine contexts
        combined_context = "\n\n".join([ctx['context'] for ctx in retrieved_contexts])
        
        # Step 3: Generate answer using LLM
        answer = self.llm_generator.generate_answer(
            combined_context, 
            question, 
            max_tokens=max_tokens, 
            temperature=temperature
        )
        
        # Prepare result
        result = {
            'answer': answer,
            'question': question,
            'retrieved_contexts': retrieved_contexts,
            'context_used': combined_context,
            'num_contexts_retrieved': len(retrieved_contexts),
            'top_similarity': retrieved_contexts[0]['similarity'] if retrieved_contexts else 0.0,
            'llm_model': self.llm_model,
            'error': False
        }
        
        return result
    
    def batch_answer_questions(self, questions: list, **kwargs) -> list:

        results = []
        for i, question in enumerate(questions):
            result = self.answer_question(question, **kwargs)
            results.append(result)
        
        return results
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        info = {
            'embedding_model': self.model_name,
            'llm_model': self.llm_model,
            'embeddings_dir': self.embeddings_dir,
            'is_initialized': self.is_initialized,
            'llm_available': self.llm_generator.check_model_availability() if self.is_initialized else False
        }
        
        if self.is_initialized and self.embedding_model.index is not None:
            info['num_contexts_indexed'] = self.embedding_model.index.ntotal
        
        return info

def main():
    # Initialize pipeline
    rag = RAGPipeline()
    
    # Initialize with sample data
    if rag.initialize():
        print("RAG Pipeline initialized successfully!")
        print("Pipeline info:", rag.get_pipeline_info())
        
        # Test questions
        test_questions = [
            "ما هو الذكاء الاصطناعي؟",
            "ما هي عاصمة جمهورية مصر العربية؟",
            "متى تم بناء الهرم الأكبر؟",
            "كم عدد الأشخاص الذين يتحدثون باللغة العربية؟"
        ]
        
        # Answer questions
        for question in test_questions:
            print(f"\nسؤال: {question}")
            print("-" * 50)
            
            result = rag.answer_question(question)
            
            if not result['error']:
                print(f"الإجابة: {result['answer']}")
                print(f"عدد السياقات المسترجعة: {result['num_contexts_retrieved']}")
                print(f"أعلى تشابه: {result['top_similarity']:.4f}")
            else:
                print(f"خطأ: {result['error_message']}")
            
            print()
    else:
        print("Failed to initialize RAG pipeline")

if __name__ == "__main__":
    main()
