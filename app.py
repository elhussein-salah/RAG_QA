import gradio as gr
from rag_pipeline import RAGPipeline
import os
import sys


class RAGApp:

    
    def __init__(self):
        self.rag_pipeline = RAGPipeline()
        self.is_initialized = False
    
    def initialize_pipeline(self):
        if not self.is_initialized:
            success = self.rag_pipeline.initialize()
            if success:
                self.is_initialized = True
                return "✅ تم تهيئة النظام بنجاح"
            else:
                return "❌ فشل في تهيئة النظام"
        return "✅ النظام مهيأ مسبقاً"
    
    def initialize_on_load(self):
        if not self.is_initialized:
            self.rag_pipeline.initialize()
            self.is_initialized = True
    
    def answer_question(self, question: str):
        if not self.is_initialized:
            init_msg = self.initialize_pipeline()
            if "فشل" in init_msg:
                return "", "يرجى تهيئة النظام أولاً"
        
        if not question.strip():
            return "", "يرجى إدخال سؤال"
        
        result = self.rag_pipeline.answer_question(
            question=question,
            top_k=3,
            temperature=0.7
        )
        
        if result['error']:
            return "", result['error_message']
        
        return result['answer'], ""
    
    def create_interface(self):
        # Enhanced CSS for better Arabic UI
        custom_css = """
        .rtl { 
            direction: rtl; 
            text-align: right; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 20px;
            text-align: center;
        }
        .section-header {
            color: #4a5568;
            border-bottom: 2px solid #764ba2;
            padding-bottom: 10px;
            margin: 20px 0 15px 0;
        }
        .question-input {
            border: 2px solid #764ba2;
            border-radius: 10px;
            padding: 15px;
        }
        .answer-output {
            background: #f7fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 15px;
        }
        .sample-btn {
            margin: 5px;
            background: #000;
            border: 1px solid #764ba2;
            border-radius: 8px;
            padding: 10px 15px;
            transition: all 0.3s ease;
        }
        .sample-btn:hover {
            background: #667eea;
            transform: translateY(-2px);
        }
        """
        
        with gr.Blocks(
            title="نظام الإجابة على الأسئلة العربية باستخدام RAG",
            theme=gr.themes.Soft(),
            css=custom_css
        ) as app:
            
            gr.HTML("""
            <div class="main-header">
                <h1 style="margin: 0; font-size: 2.5em;">🤖 نظام الإجابة على الأسئلة العربية</h1>
                <p style="margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9;">
                    نظام ذكي للإجابة على الأسئلة باللغة العربية باستخدام تقنية RAG
                </p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Input section
                    gr.HTML('<h3 class="section-header">🔍 اطرح سؤالك</h3>')
                    
                    question_input = gr.Textbox(
                        label="",
                        placeholder="اكتب سؤالك باللغة العربية هنا...",
                        lines=3,
                        elem_classes=["rtl", "question-input"],
                        show_label=False
                    )
                    
                    submit_btn = gr.Button(
                        "🚀 احصل على الإجابة", 
                        variant="primary", 
                        size="lg",
                        elem_classes=["rtl"]
                    )
            
            # Output section
            gr.HTML('<h3 class="section-header">💬 الإجابة</h3>')
            
            answer_output = gr.Textbox(
                label="",
                lines=6,
                interactive=False,
                elem_classes=["rtl", "answer-output"],
                show_label=False,
                placeholder="ستظهر الإجابة هنا..."
            )
            
            error_output = gr.Textbox(
                label="رسائل الخطأ",
                visible=False,
                interactive=False
            )
            
            # Sample questions
            gr.HTML('<h3 class="section-header">🔖 أسئلة مقترحة</h3>')
            
            sample_questions = [
                "ما هو الذكاء الاصطناعي؟",
                "ما هي عاصمة جمهورية مصر العربية؟",
                "متى تم بناء الهرم الأكبر؟",
                "كم عدد الأشخاص الذين يتحدثون باللغة العربية؟",
                "ما هي أهداف رؤية مصر 2030؟"
            ]
            
            with gr.Row():
                for i, sample_q in enumerate(sample_questions):
                    if i < 3:  # First row with 3 buttons
                        gr.Button(
                            sample_q,
                            elem_classes=["rtl", "sample-btn"],
                            size="sm"
                        ).click(
                            lambda q=sample_q: q,
                            outputs=question_input
                        )
            
            with gr.Row():
                for i, sample_q in enumerate(sample_questions):
                    if i >= 3:  # Second row with remaining buttons
                        gr.Button(
                            sample_q,
                            elem_classes=["rtl", "sample-btn"],
                            size="sm"
                        ).click(
                            lambda q=sample_q: q,
                            outputs=question_input
                        )
            
            # Event handlers
            def handle_submit(question):
                answer, error = self.answer_question(question)
                
                if error:
                    return f"❌ خطأ: {error}", gr.update(visible=False, value="")
                else:
                    return answer, gr.update(visible=False, value="")
            
            submit_btn.click(
                handle_submit,
                inputs=[question_input],
                outputs=[answer_output, error_output]
            )
            
            # Enter key support
            question_input.submit(
                handle_submit,
                inputs=[question_input],
                outputs=[answer_output, error_output]
            )
            
            # Initialize on load
            app.load(
                self.initialize_on_load
            )
        
        return app

def main():
    # Create and launch the app
    rag_app = RAGApp()
    app = rag_app.create_interface()
    
    # Launch with custom settings
    app.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
        show_api=False,
        show_error=True
    )

def run_cli():
    """Simple command-line interface for testing."""
    print("🤖 نظام الإجابة على الأسئلة العربية - واجهة سطر الأوامر")
    print("=" * 60)
    
    rag_app = RAGApp()
    init_status = rag_app.initialize_pipeline()
    print(f"حالة التهيئة: {init_status}")
    
    if "فشل" in init_status:
        print("تعذرت تهيئة النظام. يرجى التحقق من المتطلبات.")
        return
    
    print("\nيمكنك الآن طرح الأسئلة (اكتب 'خروج' للإنهاء)")
    print("-" * 40)
    
    while True:
        question = input("\nالسؤال: ").strip()
        
        if question.lower() in ['خروج', 'exit', 'quit']:
            print("شكراً لاستخدام النظام!")
            break
        
        if not question:
            continue
        
        answer, error = rag_app.answer_question(question)
        
        if error:
            print(f"خطأ: {error}")
        else:
            print(f"\nالإجابة: {answer}")

if __name__ == "__main__":
    
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_cli()
    else:
        main()
