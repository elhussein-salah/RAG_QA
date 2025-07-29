import pandas as pd
import json
from typing import List, Dict, Any

def create_sample_arabic_dataset() -> List[Dict[str, str]]:
    # Minimal fallback dataset
    return [
        {
            "context": "هذا سياق تجريبي للاختبار فقط.",
            "question": "ما هذا؟",
            "answer": "هذا مثال تجريبي."
        }
    ]

def load_dataset(file_path: str = None) -> pd.DataFrame:
    # Set default file path if none provided
    if file_path is None:
        file_path = "sample_arabic_dataset.json"
    
    if pd.io.common.file_exists(file_path):
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
    else:
        data = create_sample_arabic_dataset()
        df = pd.DataFrame(data)
        
    # Validate required columns
    required_columns = ['context', 'question', 'answer']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Dataset must contain columns: {required_columns}")
        
    return df

def save_dataset(df: pd.DataFrame, file_path: str) -> None:
    if file_path.endswith('.csv'):
        df.to_csv(file_path, index=False, encoding='utf-8')
    elif file_path.endswith('.json'):
        df.to_json(file_path, orient='records', ensure_ascii=False, indent=2)
    else:
        raise ValueError("Unsupported file format. Use CSV or JSON.")

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    # Basic cleaning
    text = text.strip()
    # Remove extra whitespaces
    text = ' '.join(text.split())
    
    return text

def prepare_contexts_for_embedding(df: pd.DataFrame) -> List[str]:
    contexts = []
    for _, row in df.iterrows():
        context = preprocess_text(row['context'])
        contexts.append(context)
    
    return contexts

if __name__ == "__main__":
    # Example usage - Load from JSON file (default behavior)
    print("Loading dataset from JSON file...")
    df = load_dataset()  # Will use sample_arabic_dataset.json by default
    print(f"Dataset loaded with {len(df)} records")
    print("\nFirst few records:")
    print(df.head(2))
    
    # You can also load from a specific file
    # df = load_dataset("my_custom_dataset.json")
    # df = load_dataset("my_dataset.csv")
    
    # Save dataset in different format if needed
    save_dataset(df, "dataset_backup.csv")
    print("\nDataset also saved as CSV backup")
    
    # Prepare contexts for embedding
    contexts = prepare_contexts_for_embedding(df)
    print(f"\nPrepared {len(contexts)} contexts for embedding")
    print(f"First context preview: {contexts[0][:100]}...")
