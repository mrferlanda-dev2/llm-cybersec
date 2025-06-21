#!/usr/bin/env python3
"""
Simple RAG vs Non-RAG Comparison Script
Follows the user's requested flow:
1. Evaluate model WITH RAG on pentest dataset
2. Save questions that were actually answered (not refused)
3. Evaluate model WITHOUT RAG on the same questions
4. Compare results
"""

import os
import sys
import pandas as pd
import re
from datetime import datetime

# Add path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import CyberSecurityLLM
from rag.rag import RAGRetriever, RAGPipeline

def load_pentest_data(sample_size=50):
    """Load and format pentest questions."""
    print(f"Loading {sample_size} pentest questions from HuggingFace...")
    
    df = pd.read_parquet("hf://datasets/preemware/pentesting-eval/data/train-00000-of-00001.parquet")
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    # Format choices
    df_sample["choices_formatted"] = df_sample["choices"].apply(
        lambda x: "A. " + x[0] + "\nB. " + x[1] + "\nC. " + x[2] + "\nD. " + x[3]
    )
    
    # Format answers
    dict_choices = {0: "A", 1: "B", 2: "C", 3: "D"}
    df_sample["answer_formatted"] = df_sample["answer"].apply(lambda x: dict_choices[x])
    
    print(f"Loaded {len(df_sample)} questions")
    return df_sample

def ask_model_with_rag(questions_df, model_name="llama3.2:latest"):
    """Evaluate model WITH RAG."""
    print("\n1. EVALUATING WITH RAG")
    print("-" * 40)
    
    # Initialize RAG system
    rag_retriever = RAGRetriever(vector_store_path="db/vectors")
    rag_pipeline = RAGPipeline(rag_retriever)
    llm = CyberSecurityLLM(model_name=model_name)
    
    results = []
    
    for i, row in questions_df.iterrows():
        print(f"Question {len(results)+1}/{len(questions_df)}")
        
        question = row["question"]
        choices = row["choices_formatted"]
        human_explanation = row["explanation"]
        correct_answer = row["answer_formatted"]
        
        # Form question with choices
        question_with_choices = f"Question: \n{question}\n{choices}"
        
        # Get RAG context
        context = rag_pipeline.get_context_for_query(question_with_choices)
        
        # Use same prompt as notebook
        prompt = f"""
You are an assistant for multiple choice question-answering tasks. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just choose random option.
    IMPORTANT RULES:
    1. Answer in format [choice]. For example if the answer is C, then answer [C] and give explanation of the answer. 
    2. Make sure your answer is relevant to the question and it is answered from the context only.
    Question: {question_with_choices} 
    Context: {context} 
"""
        
        # Get model response
        response = llm.generate_response_sync(question=prompt, context="")
        
        # Extract choice
        choice_match = re.search(r'\[(A|B|C|D)\]', response)
        
        if choice_match:
            model_choice = choice_match.group(1)
            explanation = response.replace(f"[{model_choice}]", "").strip()
            
            # Clean explanation
            explanation = re.sub(r'\[\]', '', explanation)
            explanation = re.sub(r'\n', ' ', explanation)
            explanation = re.sub(r'^\. ', '', explanation)
            explanation = explanation.strip()
            
            # Check if it's a refusal
            is_refusal = "can't provide" in explanation.lower() or "cannot provide" in explanation.lower()
            
            if not is_refusal:
                results.append({
                    'question': question,
                    'choices': choices,
                    'human_explanation': human_explanation,
                    'correct_answer': correct_answer,
                    'rag_choice': model_choice,
                    'rag_explanation': explanation,
                    'rag_context': context
                })
                print(f"  ✓ Answered: {model_choice}")
            else:
                print(f"  ✗ Refused to answer")
        else:
            print(f"  ✗ Invalid format")
    
    print(f"RAG model answered {len(results)} out of {len(questions_df)} questions")
    return pd.DataFrame(results)

def ask_model_without_rag(answered_questions_df, model_name="llama3.2:latest"):
    """Evaluate model WITHOUT RAG on the same questions."""
    print("\n2. EVALUATING WITHOUT RAG (same questions)")
    print("-" * 40)
    
    llm = CyberSecurityLLM(model_name=model_name)
    
    for i, row in answered_questions_df.iterrows():
        print(f"Question {i+1}/{len(answered_questions_df)}")
        
        question = row["question"]
        choices = row["choices"]
        
        question_with_choices = f"Question: \n{question}\n{choices}"
        
        # Use same prompt but with no context
        prompt = f"""
You are an assistant for multiple choice question-answering tasks. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just choose random option.
    IMPORTANT RULES:
    1. Answer in format [choice]. For example if the answer is C, then answer [C] and give explanation of the answer. 
    2. Make sure your answer is relevant to the question and it is answered from the context only.
    Question: {question_with_choices} 
    Context: No additional context provided
"""
        
        # Get model response
        response = llm.generate_response_sync(question=prompt, context="")
        
        # Extract choice
        choice_match = re.search(r'\[(A|B|C|D)\]', response)
        
        if choice_match:
            model_choice = choice_match.group(1)
            explanation = response.replace(f"[{model_choice}]", "").strip()
            
            # Clean explanation
            explanation = re.sub(r'\[\]', '', explanation)
            explanation = re.sub(r'\n', ' ', explanation)
            explanation = re.sub(r'^\. ', '', explanation)
            explanation = explanation.strip()
            
            answered_questions_df.at[i, 'no_rag_choice'] = model_choice
            answered_questions_df.at[i, 'no_rag_explanation'] = explanation
            print(f"  ✓ Answered: {model_choice}")
        else:
            answered_questions_df.at[i, 'no_rag_choice'] = "INVALID"
            answered_questions_df.at[i, 'no_rag_explanation'] = "Invalid response format"
            print(f"  ✗ Invalid format")
    
    return answered_questions_df

def compare_results(results_df):
    """Compare RAG vs Non-RAG results."""
    print("\n3. COMPARISON RESULTS")
    print("=" * 50)
    
    # Calculate accuracy
    total_questions = len(results_df)
    rag_correct = (results_df['rag_choice'] == results_df['correct_answer']).sum()
    no_rag_correct = (results_df['no_rag_choice'] == results_df['correct_answer']).sum()
    
    rag_accuracy = rag_correct / total_questions
    no_rag_accuracy = no_rag_correct / total_questions
    improvement = rag_accuracy - no_rag_accuracy
    
    print(f"Total Questions Compared: {total_questions}")
    print(f"")
    print(f"WITH RAG:")
    print(f"  Correct: {rag_correct}/{total_questions}")
    print(f"  Accuracy: {rag_accuracy:.3f} ({rag_accuracy*100:.1f}%)")
    print(f"")
    print(f"WITHOUT RAG:")
    print(f"  Correct: {no_rag_correct}/{total_questions}")
    print(f"  Accuracy: {no_rag_accuracy:.3f} ({no_rag_accuracy*100:.1f}%)")
    print(f"")
    print(f"RAG IMPACT:")
    print(f"  Accuracy Change: {improvement:+.3f} ({improvement*100:+.1f}%)")
    
    if improvement > 0.05:
        print(f"  Result: ✓ RAG provides significant improvement")
    elif improvement > 0.02:
        print(f"  Result: ✓ RAG provides moderate improvement")
    elif improvement > -0.02:
        print(f"  Result: ~ RAG has minimal impact")
    else:
        print(f"  Result: ✗ RAG hurts performance")
    
    # Detailed comparison
    print(f"\nDETAILED COMPARISON:")
    print(f"Questions where RAG helped: {((results_df['rag_choice'] == results_df['correct_answer']) & (results_df['no_rag_choice'] != results_df['correct_answer'])).sum()}")
    print(f"Questions where RAG hurt: {((results_df['rag_choice'] != results_df['correct_answer']) & (results_df['no_rag_choice'] == results_df['correct_answer'])).sum()}")
    print(f"Questions where both correct: {((results_df['rag_choice'] == results_df['correct_answer']) & (results_df['no_rag_choice'] == results_df['correct_answer'])).sum()}")
    print(f"Questions where both wrong: {((results_df['rag_choice'] != results_df['correct_answer']) & (results_df['no_rag_choice'] != results_df['correct_answer'])).sum()}")
    
    return {
        'total_questions': total_questions,
        'rag_accuracy': rag_accuracy,
        'no_rag_accuracy': no_rag_accuracy,
        'improvement': improvement,
        'rag_correct': rag_correct,
        'no_rag_correct': no_rag_correct
    }

def save_results(results_df, comparison_stats, model_name):
    """Save results to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_file = f"eval/results/simple_comparison_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\nDetailed results saved to: {results_file}")
    
    # Save summary
    summary_file = f"eval/results/simple_comparison_summary_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write("SIMPLE RAG vs NON-RAG COMPARISON\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Questions Compared: {comparison_stats['total_questions']}\n\n")
        
        f.write("RESULTS:\n")
        f.write(f"WITH RAG: {comparison_stats['rag_correct']}/{comparison_stats['total_questions']} = {comparison_stats['rag_accuracy']:.3f}\n")
        f.write(f"WITHOUT RAG: {comparison_stats['no_rag_correct']}/{comparison_stats['total_questions']} = {comparison_stats['no_rag_accuracy']:.3f}\n")
        f.write(f"IMPROVEMENT: {comparison_stats['improvement']:+.3f} ({comparison_stats['improvement']*100:+.1f}%)\n")
    
    print(f"Summary saved to: {summary_file}")

def main():
    """Main comparison function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple RAG vs Non-RAG comparison")
    parser.add_argument("--model-name", default="llama3.2:latest", help="Model to evaluate")
    parser.add_argument("--sample-size", type=int, default=20, help="Number of questions to test")
    
    args = parser.parse_args()
    
    print("SIMPLE RAG vs NON-RAG COMPARISON")
    print("=" * 50)
    print(f"Model: {args.model_name}")
    print(f"Sample Size: {args.sample_size}")
    print("=" * 50)
    
    try:
        # Step 1: Load questions
        questions_df = load_pentest_data(args.sample_size)
        
        # Step 2: Evaluate WITH RAG (save answered questions)
        answered_df = ask_model_with_rag(questions_df, args.model_name)
        
        if len(answered_df) == 0:
            print("No questions were answered with RAG. Cannot compare.")
            return
        
        # Step 3: Evaluate WITHOUT RAG (same questions)
        final_df = ask_model_without_rag(answered_df, args.model_name)
        
        # Step 4: Compare results
        comparison_stats = compare_results(final_df)
        
        # Step 5: Save results
        save_results(final_df, comparison_stats, args.model_name)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 