import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    pipeline
)
import matplotlib.pyplot as plt
import seaborn as sns
from rouge_score import rouge_scorer
import time
import psutil
from datasets import load_dataset
from IPython.display import display  # Import display for DataFrame

def calculate_rouge_scores(predicted, reference):
    """Calculate ROUGE scores for predicted summary."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(predicted, reference)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

def evaluate_model(model_name):
    """Evaluate a model's performance on various metrics."""
    try:
        # Load model and tokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create summarization pipeline with specific parameters
        summarizer = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            max_length=130,
            min_length=30,
            truncation=True
        )
        
        # Load test data (using first 5 examples from CNN/DailyMail dataset)
        dataset = load_dataset("cnn_dailymail", '3.0.0', split='test[:5]')
        
        # Initialize metrics
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        latencies = []
        memory_usages = []
        
        # Evaluate on test examples
        for example in dataset:
            try:
                article = example['article'][:1024]  # Truncate to first 1024 chars to avoid issues
                reference = example['highlights']
                
                # Measure latency and memory
                start_time = time.time()
                process = psutil.Process()
                initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
                
                # Generate summary with error handling
                try:
                    summary = summarizer(article)[0]['summary_text']
                except Exception as e:
                    print(f"Error generating summary for {model_name}: {str(e)}")
                    continue
                
                # Record metrics
                latencies.append((time.time() - start_time) * 1000)  # Convert to ms
                memory_usages.append(process.memory_info().rss / (1024 * 1024) - initial_memory)
                
                # Calculate ROUGE scores
                scores = calculate_rouge_scores(summary, reference)
                for metric, score in scores.items():
                    rouge_scores[metric].append(score)
                    
            except Exception as e:
                print(f"Error processing example for {model_name}: {str(e)}")
                continue
        
        # Calculate average metrics (with error handling for empty lists)
        avg_metrics = {
            'Model': model_name,
            'ROUGE-1': np.mean(rouge_scores['rouge1']) if rouge_scores['rouge1'] else 0,
            'ROUGE-2': np.mean(rouge_scores['rouge2']) if rouge_scores['rouge2'] else 0,
            'ROUGE-L': np.mean(rouge_scores['rougeL']) if rouge_scores['rougeL'] else 0,
            'Latency (ms)': np.mean(latencies) if latencies else 0,
            'Memory (MB)': np.mean(memory_usages) if memory_usages else 0,
            'Parameters (M)': sum(p.numel() for p in model.parameters()) / 1e6
        }
        
        return avg_metrics
        
    except Exception as e:
        print(f"Failed to evaluate {model_name}: {str(e)}")
        return {
            'Model': model_name,
            'ROUGE-1': 0,
            'ROUGE-2': 0,
            'ROUGE-L': 0,
            'Latency (ms)': 0,
            'Memory (MB)': 0,
            'Parameters (M)': 0
        }

def perform_topsis(data_df, weights, criteria):
    """Perform TOPSIS analysis."""
    # Normalize the decision matrix
    normalized = data_df.copy()
    for column in normalized.columns[1:]:  # Skip the 'Model' column
        if normalized[column].sum() != 0:  # Avoid division by zero
            normalized[column] = normalized[column] / np.sqrt(sum(normalized[column]**2))
    
    # Apply weights
    for column, weight in zip(normalized.columns[1:], weights):
        normalized[column] = normalized[column] * weight
    
    # Identify ideal and negative-ideal solutions
    ideal = []
    negative_ideal = []
    for column, criterion in zip(normalized.columns[1:], criteria):
        if criterion == 'benefit':
            ideal.append(normalized[column].max())
            negative_ideal.append(normalized[column].min())
        else:
            ideal.append(normalized[column].min())
            negative_ideal.append(normalized[column].max())
    
    # Calculate separation measures
    S_positive = np.sqrt(((normalized.iloc[:, 1:] - ideal) ** 2).sum(axis=1))
    S_negative = np.sqrt(((normalized.iloc[:, 1:] - negative_ideal) ** 2).sum(axis=1))
    
    # Calculate relative closeness to ideal solution
    C = S_negative / (S_positive + S_negative)
    
    # Add TOPSIS scores to original dataframe
    data_df['TOPSIS Score'] = C
    data_df = data_df.sort_values('TOPSIS Score', ascending=False)
    
    return data_df

def plot_results(df):
    """Create visualizations for the results."""
    try:
        # TOPSIS Score Plot
        plt.figure(figsize=(10, 6))
        plt.barh(df['Model'], df['TOPSIS Score'], color='skyblue')
        plt.xlabel('TOPSIS Score')
        plt.title('Model Ranking based on TOPSIS Analysis')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()  # Show plot in Colab
        
        # Metrics Heatmap
        plt.figure(figsize=(12, 6))
        metrics_df = df.drop(['TOPSIS Score'], axis=1).set_index('Model')
        sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='coolwarm', linewidths=0.5)
        plt.title('Model Evaluation Metrics')
        plt.tight_layout()
        plt.show()  # Show heatmap in Colab
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")

def main():
    # Define models to evaluate
    models = [
        't5-small',
        'google/pegasus-xsum',
        'sshleifer/distilbart-cnn-6-6'
    ]
    
    # Collect metrics for each model
    results = []
    for model_name in models:
        print(f"Evaluating {model_name}...")
        metrics = evaluate_model(model_name)
        results.append(metrics)
    
    # Create DataFrame with results
    df = pd.DataFrame(results)
    
    # Define TOPSIS parameters
    weights = [0.2, 0.2, 0.2, 0.15, 0.15, 0.1]  # Weights for each metric
    criteria = ['benefit', 'benefit', 'benefit', 'cost', 'cost', 'cost']  # benefit = higher is better, cost = lower is better
    
    # Perform TOPSIS analysis
    final_df = perform_topsis(df, weights, criteria)
    
    # Display results as table
    print("\nFinal Results:")
    display(final_df)  # Display DataFrame in Colab
    
    # Create visualizations
    plot_results(final_df)
    
    # Save results to CSV
    final_df.to_csv('model_evaluation_results.csv', index=False)

if __name__ == "__main__":
    main()
