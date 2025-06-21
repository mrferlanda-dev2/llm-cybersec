from ragas import evaluate
import json
import pandas as pd
from datasets import Dataset
import argparse
from ragas import EvaluationDataset
import re
import os 
from ragas.llms import LangchainLLMWrapper 
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, SemanticSimilarity

os.environ["GOOGLE_API_KEY"] = "keyhere"

def evaluate_ragas(dataset):
    llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
    )
    metrics = [
        LLMContextRecall(llm=llm), 
        FactualCorrectness(llm=llm), 
        Faithfulness(llm=llm),
        SemanticSimilarity(embeddings=embeddings)
    ]

    result = evaluate(
        dataset,
        show_progress=True,
        batch_size=24,
        metrics=metrics,
        )
    return result

def save_ragas_results(result, filename):
    df = result.to_pandas()
    df.to_csv(filename, index=False)

def split_rag_context(rag_context):
    return re.split(r'Document \d+:', rag_context)

def convert_csv_to_ragas_dataset(csv_file, sample_size):
    df = pd.read_csv(csv_file)
  
    if sample_size:
        df = df.sample(n=sample_size)

    # Clean data - convert all text fields to strings and handle NaN
    text_columns = ['question', 'choices', 'rag_explanation', 'correct_answer', 'rag_context']
    for col in text_columns:
        if col in df.columns:
            # Convert to string and replace nan with empty string
            df[col] = df[col].astype(str)
            df[col] = df[col].replace('nan', 'Not available')
            df[col] = df[col].replace('', 'Not available')
    
    # Filter out rows where essential fields are missing
    df = df.dropna(subset=['question', 'rag_explanation', 'correct_answer', 'rag_context'])
    df = df[df['rag_explanation'] != 'Not available']
    df = df[df['rag_context'] != 'Not available']
    
    print(f"Using {len(df)} valid rows after cleaning")

    df['user_input'] = df['question'] + "\n" + df['choices']
    #currently the rag_context format is string with Document 1: ... Document 3.. split into list with regex
    df['retrieved_contexts'] = df['rag_context'].apply(split_rag_context)
    
    # Clean the split contexts - remove empty strings and ensure all are strings
    df['retrieved_contexts'] = df['retrieved_contexts'].apply(
        lambda contexts: [str(ctx).strip() for ctx in contexts if str(ctx).strip() and str(ctx).strip() != 'Not available']
    )
    
    # Ensure at least one context per row
    df['retrieved_contexts'] = df['retrieved_contexts'].apply(
        lambda contexts: contexts if len(contexts) > 0 else ['No context available']
    )
    
    print(f"Sample retrieved_contexts: {df['retrieved_contexts'].iloc[0]}")
    print(f"Sample response: {str(df['rag_explanation'].iloc[0])[:100]}...")
   
    dataset = []
    for index, row in df.iterrows():
        dataset.append(
            {
                'user_input': str(row['user_input']),
                'reference': str(row['human_explanation']),
                'retrieved_contexts': row['retrieved_contexts'],
                'response': str(row['rag_explanation'])
            }
        )
    
    dataset = EvaluationDataset.from_list(dataset)
    return dataset

def main():
    parser = argparse.ArgumentParser(description='Evaluate RAGAS')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')
    parser.add_argument('--sample_size', type=int, required=False, help='Number of samples to evaluate')
    args = parser.parse_args()

    dataset = convert_csv_to_ragas_dataset(args.csv_file, args.sample_size)
    result = evaluate_ragas(dataset)
    save_ragas_results(result, args.output_file)

if __name__ == "__main__":
    main()