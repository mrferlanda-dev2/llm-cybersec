import os
import warnings
from typing import Dict, Any
import pandas as pd
from datasets import load_dataset, concatenate_datasets, Dataset
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import  RecursiveCharacterTextSplitter
import faiss
# Suppress KMP duplicate library warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings('ignore')
from langchain_community.vectorstores import FAISS 
from langchain_community.docstore.in_memory import InMemoryDocstore

def extract_text_field(field_name: str):
    """Create a function to extract a specific field and rename it to 'text'."""
    def extract_fn(example: Dict[str, Any]) -> Dict[str, str]:
        return {'text': example[field_name]}
    return extract_fn


def load_secqa_datasets() -> Dataset:
    """Load and merge SecQA v1 and v2 datasets."""
    try:
        ds1_v1 = load_dataset("zefang-liu/secqa", "secqa_v1")
        ds1_v2 = load_dataset("zefang-liu/secqa", "secqa_v2")
        
        available_datasets = []
        
        # Add v2 train split if available
        if 'train' in ds1_v2:
            available_datasets.append(ds1_v2['train'])
        
        # Add all v1 splits
        available_datasets.extend(ds1_v1.values())
        
        # Merge and extract text
        merged_secqa = concatenate_datasets(available_datasets)
        return merged_secqa.map(extract_text_field('Explanation'), remove_columns=merged_secqa.column_names)
    
    except Exception as e:
        print(f"Error loading SecQA datasets: {e}")
        return None


def load_cybersecurity_dataset() -> Dataset:
    """Load and process cybersecurity dataset."""
    try:
        ds2 = load_dataset("AlicanKiraz0/Cybersecurity-Dataset-v1")
        
        # Process all splits
        processed_datasets = [
            ds2[split].map(extract_text_field('assistant'), remove_columns=ds2[split].column_names)
            for split in ds2.keys()
        ]
        
        return concatenate_datasets(processed_datasets)
    
    except Exception as e:
        print(f"Error loading Cybersecurity dataset: {e}")
        return None


def load_and_merge_datasets() -> Dataset:
    """Load and merge all cybersecurity datasets into a unified format."""
    print("Loading datasets...")
    
    # Load individual datasets
    secqa_data = load_secqa_datasets()
    cyber_data = load_cybersecurity_dataset()
    
    # Filter out failed loads
    datasets_to_merge = [ds for ds in [secqa_data, cyber_data] if ds is not None]
    
    if not datasets_to_merge:
        raise ValueError("No datasets could be loaded successfully")
    
    # Merge all datasets
    final_dataset = concatenate_datasets(datasets_to_merge)
    
    print(f"Successfully merged {len(datasets_to_merge)} datasets")
    print(f"Final dataset size: {len(final_dataset):,} samples")
    print(f"Columns: {final_dataset.column_names}")
    
    return final_dataset


def preprocess_datasets(ds: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess cybersecurity text data.
    
    Args:
        ds: Input pandas DataFrame with 'text' column
        
    Returns:
        Cleaned pandas DataFrame
    """
    if ds.empty:
        return ds
    
    # Create a copy to avoid modifying original
    df = ds.copy()
    
    # Basic text cleaning
    if 'text' in df.columns:
        # Remove duplicates
        initial_size = len(df)
        df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
        
        # Remove empty or null text entries
        df = df.dropna(subset=['text'])
        df = df[df['text'].str.strip() != ''].reset_index(drop=True)
        
        # Basic text preprocessing
        df['text'] = df['text'].str.strip()
        
        print(f"Preprocessing complete:")
        print(f"  - Final size: {len(df):,} samples")
    
    return df

def convert_to_documents(df: pd.DataFrame) -> list[Document]:
    """Convert pandas DataFrame to list of Document objects."""
    print("Converting to documents...")
    documents = []
    for _, row in df.iterrows():
        text = row['text']
        if isinstance(text, str) and text.strip():
            doc = Document(page_content=text)
            documents.append(doc)
    print(f"Successfully converted to {len(documents)} documents")
    return documents

def chunking_documents(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 100) -> list[Document]:
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunk = text_splitter.split_documents(documents)
    print(f"Successfully chunked to {len(chunk)} chunks")
    return chunk

def create_vector_store(chunk: list[Document]) -> FAISS:
    print("Creating vector store... This may take a while")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector = embeddings.embed_query("Hello World")
    index = faiss.IndexFlatL2(len(vector))
    vector_store = FAISS(
        embedding_function=embeddings, 
        index=index, 
        docstore=InMemoryDocstore(), 
        index_to_docstore_id={})
    
    vector_store.add_documents(chunk)
    print("Successfully created vector store")

    return vector_store

def save_vector_store(vector_store: FAISS, path: str):
    print("Saving vector store...")
    vector_store.save_local(path)
    print(f"Successfully saved vector store to {path}")


def main():
    """Main function to demonstrate dataset loading and preprocessing."""
    try:
        # Load and merge datasets
        dataset = load_and_merge_datasets()
        
        # Convert to pandas for preprocessing
        df = dataset.to_pandas()
        
        # Apply preprocessing
        cleaned_df = preprocess_datasets(df)

        documents = convert_to_documents(cleaned_df)
        chunk = chunking_documents(documents)
        vector_store = create_vector_store(chunk)
        save_vector_store(vector_store, "db/vectors")
        
        print("\nDataset ready for use!")
        return cleaned_df
        
    except Exception as e:
        print(f"Error in main pipeline: {e}")
        return None


if __name__ == "__main__":
    main()