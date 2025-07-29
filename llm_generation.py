import ollama
from typing import Optional, Dict, Any
import json

class OllamaLLMGenerator: 
    def __init__(self, model_name: str = "gemma3:1b"):
        self.model_name = model_name
        self.client = ollama.Client()
        
    def check_model_availability(self) -> bool:
        try:
            models = self.client.list()
            
            # Handle different possible response structures
            if hasattr(models, 'models'):
                model_list = models.models
            elif isinstance(models, dict) and 'models' in models:
                model_list = models['models']
            else:
                return False
            
            # Extract model names safely
            available_models = []
            for model in model_list:
                if hasattr(model, 'name'):
                    available_models.append(model.name)
                elif isinstance(model, dict) and 'name' in model:
                    available_models.append(model['name'])
                elif isinstance(model, dict) and 'model' in model:
                    available_models.append(model['model'])
                elif isinstance(model, str):
                    available_models.append(model)
            
            is_available = any(self.model_name in model for model in available_models)
            return is_available
        except Exception as e:
            # If there's any error checking availability, return False
            return False
    
    def pull_model(self) -> bool:
        try:
            self.client.pull(self.model_name)
            return True
        except Exception as e:
            return False
    
    def ensure_model_ready(self) -> bool:
        if not self.check_model_availability():
            return self.pull_model()
        return True
    
    def create_arabic_prompt(self, context: str, question: str) -> str:

        prompt = f"""أنت مساعد ذكي يجيب على الأسئلة باللغة العربية بناءً على السياق المعطى.

السياق:
{context}

السؤال: {question}

التعليمات:
- اجب على السؤال بناءً على المعلومات الموجودة في السياق فقط
- إذا لم تجد الإجابة في السياق، قل "لا توجد معلومات كافية في السياق المعطى للإجابة على هذا السؤال"
- اجعل إجابتك واضحة ومفيدة ومختصرة
- استخدم اللغة العربية الفصحى

الإجابة:"""
        
        return prompt
    
    def generate_answer(self, context: str, question: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        # Ensure model is ready
        if not self.ensure_model_ready():
            return "عذراً، النموذج غير متاح حالياً. يرجى التأكد من تشغيل Ollama وتحميل النموذج."
        
        try:
            # Create prompt
            prompt = self.create_arabic_prompt(context, question)
            
            # Generate response
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'num_predict': max_tokens,
                    'temperature': temperature,
                    'stop': ['\n\n', 'السؤال:', 'السياق:']
                }
            )
            
            answer = response['response'].strip()
            
            # Clean up the answer
            if answer.startswith('الإجابة:'):
                answer = answer[8:].strip()
            
            return answer
        except Exception as e:
            return f"عذراً، حدث خطأ أثناء توليد الإجابة: {str(e)}"
    
    def generate_with_metadata(self, context: str, question: str, **kwargs) -> Dict[str, Any]:

        answer = self.generate_answer(context, question, **kwargs)
        
        metadata = {
            'answer': answer,
            'model_used': self.model_name,
            'context_length': len(context),
            'question_length': len(question),
            'has_context': len(context.strip()) > 0,
            'generation_params': kwargs
        }
        
        return metadata
    
    def chat_conversation(self, messages: list, max_tokens: int = 256, temperature: float = 0.7) -> str:
        if not self.ensure_model_ready():
            return "عذراً، النموذج غير متاح حالياً."
        
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    'num_predict': max_tokens,
                    'temperature': temperature
                }
            )
            
            return response['message']['content']
        except Exception as e:
            return f"عذراً، حدث خطأ: {str(e)}"

def main():
    # Initialize generator
    generator = OllamaLLMGenerator()
    
    # Test context and question
    test_context = """الذكاء الاصطناعي هو مجال في علوم الكمبيوتر يهدف إلى إنشاء أنظمة قادرة على أداء مهام تتطلب عادة ذكاء بشري. 
    يشمل هذا المجال التعلم الآلي، ومعالجة اللغة الطبيعية، والرؤية الحاسوبية. 
    تطبيقات الذكاء الاصطناعي تشمل المساعدين الافتراضيين، والسيارات ذاتية القيادة، وأنظمة التوصية."""
    
    test_question = "ما هو الذكاء الاصطناعي؟"
    
    # Test basic generation
    print("Testing answer generation...")
    answer = generator.generate_answer(test_context, test_question)
    print(f"السؤال: {test_question}")
    print(f"الإجابة: {answer}")
    
    # Test with metadata
    print("\nTesting with metadata...")
    result = generator.generate_with_metadata(test_context, test_question)
    print(f"الإجابة: {result['answer']}")
    print(f"النموذج المستخدم: {result['model_used']}")
    print(f"طول السياق: {result['context_length']}")

if __name__ == "__main__":
    main()
