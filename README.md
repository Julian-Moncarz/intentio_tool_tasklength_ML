# Task Duration Prediction from Intent Embeddings

This project uses machine learning to predict task duration based on sentence embeddings of task intentions.

## 📁 Project Structure

```
intentiontool_logs_ML/
├── data/                           # Raw and processed data
│   ├── logs.csv                   # Input data with Intent and Duration(min) columns
│   ├── embeddings_data.pkl        # Processed embeddings for ML training
│   ├── embeddings_data.json       # Human-readable embeddings
│   └── embeddings_summary.csv     # Summary of processed data
├── src/                           # Source code
│   ├── generate_embeddings.py     # Generates sentence embeddings from task intents
│   ├── train_random_forest.py     # Trains a Random Forest model 
│   ├── comprehensive_model_search.py # Comprehensive model comparison
│   ├── predict_task_duration.py   # Predict duration for new task sentences
│   ├── analyze_embedding_similarity.py # Analyze semantic similarity
│   └── validate_setup.py          # Validation script to check setup
├── models/                        # Trained models
│   ├── random_forest_model.pkl    # Random Forest model
│   ├── best_model.pkl             # Best performing model
│   └── all_model_results.pkl      # All model results
├── results/                       # Output files and visualizations
│   ├── model_comparison_results.csv
│   ├── comprehensive_model_comparison.png
│   ├── best_model_predictions.png
│   └── random_forest_results.png
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Validate setup
cd src
python validate_setup.py
```

### 2. Run Pipeline (IN ORDER)
```bash
cd src

# Step 1: Generate embeddings
python generate_embeddings.py

# Step 2: Train Random Forest (quick baseline)
python train_random_forest.py

# Step 3: Comprehensive model search (takes longer)
python comprehensive_model_search.py
```

## 📊 Output Files

After running the pipeline, you'll get organized outputs in their respective directories:

**Data files (`data/`):**
- `embeddings_data.pkl` - Processed embeddings for ML training
- `embeddings_data.json` - Human-readable embeddings
- `embeddings_summary.csv` - Summary of processed data

**Models (`models/`):**
- `random_forest_model.pkl` - Trained Random Forest model
- `best_model.pkl` - Best performing model from comprehensive search
- `all_model_results.pkl` - All model results for analysis

**Results (`results/`):**
- `random_forest_results.png` - Random Forest performance visualizations
- `model_comparison_results.csv` - Performance comparison table
- `comprehensive_model_comparison.png` - Model comparison visualizations
- `best_model_predictions.png` - Best model predictions plot

## 🔮 Making Predictions

After training models, predict duration for new task sentences:

```bash
cd src

# Interactive mode
python predict_task_duration.py

# Command line mode
python predict_task_duration.py "Write a research report on machine learning"
```

The script will:
- Generate embeddings for your input sentence
- Run predictions through both Random Forest and Best Model
- Show comparison results and model performance metrics

## ⚠️ Important Notes

1. **Run scripts from src/ directory** - All scripts are designed to be run from the `src/` folder
2. **Run scripts in order** - Each script depends on outputs from previous ones
3. **Data requirements** - Need at least 10 valid rows in `data/logs.csv` for meaningful results
4. **Memory usage** - `comprehensive_model_search.py` can be memory-intensive with large datasets
5. **Training time** - Comprehensive search can take 10-30 minutes depending on data size

## 🔧 Troubleshooting

Run `python validate_setup.py` from the `src/` directory to check for common issues:
- Missing dependencies
- Invalid data format
- Missing input files
- Insufficient data

If you encounter path issues, ensure you're running scripts from the `src/` directory.

## 💡 Model Performance

The project trains multiple models including:
- Linear Regression (Ridge, Lasso, ElasticNet)
- Tree-based (Random Forest, Gradient Boosting, XGBoost)
- Neural Networks (MLP)
- Support Vector Regression
- K-Nearest Neighbors

Results are automatically compared and the best model is saved for future use.

## 🔬 Analysis Tools

Additional analysis scripts:
- `analyze_embedding_similarity.py` - Analyze semantic similarity between task embeddings
- Use this to validate that the sentence embeddings capture meaningful semantic relationships
