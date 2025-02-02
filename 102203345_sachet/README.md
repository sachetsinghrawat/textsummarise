# Text Summarization Model Evaluation Framework

A comprehensive framework for evaluating and comparing text summarization models using multiple metrics including ROUGE scores, latency, and memory usage, with TOPSIS analysis for multi-criteria decision making.

## Features

- Automated evaluation of text summarization models
- Multiple performance metrics:
  - ROUGE-1, ROUGE-2, and ROUGE-L scores
  - Processing latency
  - Memory usage
  - Model size (parameters)
- TOPSIS analysis for overall model ranking
- Visualization of results through plots and heatmaps
- Results export to CSV

## Prerequisites

```python
numpy
pandas
torch
transformers
matplotlib
seaborn
rouge_score
psutil
datasets
```

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install numpy pandas torch transformers matplotlib seaborn rouge_score psutil datasets
```

## Usage

Run the evaluation framework with the default configuration:

```python
python model_evaluation.py
```

The script will:
1. Evaluate the following models:
   - t5-small
   - google/pegasus-xsum
   - sshleifer/distilbart-cnn-6-6
2. Generate performance metrics
3. Perform TOPSIS analysis
4. Create visualizations
5. Save results to 'model_evaluation_results.csv'

## Customization

### Adding New Models

Modify the `models` list in the `main()` function:

```python
models = [
    't5-small',
    'google/pegasus-xsum',
    'sshleifer/distilbart-cnn-6-6',
    'your-new-model'  # Add new models here
]
```

### Adjusting TOPSIS Parameters

Modify weights and criteria in the `main()` function:

```python
weights = [0.2, 0.2, 0.2, 0.15, 0.15, 0.1]  # Adjust weights
criteria = ['benefit', 'benefit', 'benefit', 'cost', 'cost', 'cost']  # Modify criteria
```

## Output

The framework generates:
1. A DataFrame with all evaluation metrics and TOPSIS scores
2. A bar plot showing model rankings based on TOPSIS analysis
3. A heatmap visualizing all metrics across models
4. A CSV file containing all results

## Functions

- `calculate_rouge_scores()`: Calculates ROUGE metrics for generated summaries
- `evaluate_model()`: Evaluates a single model across all metrics
- `perform_topsis()`: Performs TOPSIS analysis on evaluation results
- `plot_results()`: Creates visualization of the results
- `main()`: Orchestrates the evaluation process

## Error Handling

The framework includes comprehensive error handling:
- Individual model evaluation failures won't stop the entire process
- Failed evaluations return zero values for metrics
- Visualization errors are caught and reported
- Data processing errors are handled gracefully
