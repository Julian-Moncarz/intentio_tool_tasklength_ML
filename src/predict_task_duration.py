#!/usr/bin/env python3
"""
Script to predict task duration from a sentence using both models:
1. Random Forest (quickly trained baseline)
2. Best Model (from comprehensive search)

Usage: python predict_task_duration.py
Or pass sentence as argument: python predict_task_duration.py "Your task sentence here"
"""

import sys
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import os
from pathlib import Path

# Fix for torch device compatibility
if not hasattr(torch, 'get_default_device'):
    def get_default_device():
        return 'cpu' if not torch.cuda.is_available() else 'cuda'
    torch.get_default_device = get_default_device

def preprocess_for_sentence_embeddings(text):
    """Basic preprocessing for sentence embeddings - same as training"""
    if text is None or text == "":
        return ""
    
    # Basic cleaning
    text = str(text).strip()  # Convert to string and remove whitespace
    text = text.replace("\0", "")  # Remove null bytes if present
    
    return text

def load_embedding_model():
    """Load the sentence transformer model"""
    print("Loading SentenceTransformer model...")
    try:
        model = SentenceTransformer('all-mpnet-base-v2')
        print("✓ SentenceTransformer model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Error loading SentenceTransformer model: {e}")
        return None

def generate_embedding(text, embedding_model):
    """Generate embedding for input text"""
    cleaned_text = preprocess_for_sentence_embeddings(text)
    if not cleaned_text:
        print("❌ Error: Empty text after preprocessing")
        return None
    
    try:
        embedding = embedding_model.encode([cleaned_text])
        return embedding[0]  # Return single embedding vector
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return None

def load_random_forest_model():
    """Load the Random Forest model"""
    model_path = '../models/random_forest_model.pkl'
    if not os.path.exists(model_path):
        print(f"❌ Random Forest model not found: {model_path}")
        print("   Run 'python train_random_forest.py' first")
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        print("✓ Random Forest model loaded successfully")
        return model_data
    except Exception as e:
        print(f"❌ Error loading Random Forest model: {e}")
        return None

def load_best_model():
    """Load the best model from comprehensive search"""
    model_path = '../models/best_model.pkl'
    if not os.path.exists(model_path):
        print(f"❌ Best model not found: {model_path}")
        print("   Run 'python comprehensive_model_search.py' first")
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        print("✓ Best model loaded successfully")
        return model_data
    except Exception as e:
        print(f"❌ Error loading best model: {e}")
        return None

def predict_with_random_forest(embedding, rf_model_data):
    """Make prediction using Random Forest model"""
    try:
        model = rf_model_data['model']
        prediction = model.predict([embedding])[0]
        return prediction
    except Exception as e:
        print(f"❌ Error making Random Forest prediction: {e}")
        return None

def predict_with_best_model(embedding, best_model_data):
    """Make prediction using the best model"""
    try:
        model = best_model_data['model']
        prediction = model.predict([embedding])[0]
        return prediction
    except Exception as e:
        print(f"❌ Error making best model prediction: {e}")
        return None

def validate_embedding_dimension(embedding, rf_model_data, best_model_data):
    """Validate that embedding dimensions match what models expect"""
    embedding_dim = len(embedding)
    
    if rf_model_data and rf_model_data.get('embedding_dimension') != embedding_dim:
        print(f"❌ Warning: Random Forest expects {rf_model_data.get('embedding_dimension')} dimensions, got {embedding_dim}")
        return False
    
    # For best model, we can check if it has feature expectations
    # Most sklearn models will handle this gracefully, but let's warn if dimensions seem off
    if embedding_dim == 0:
        print("❌ Error: Embedding has 0 dimensions")
        return False
    
    return True

def get_user_input():
    """Get sentence input from user or command line arguments"""
    if len(sys.argv) > 1:
        # Use command line argument
        sentence = ' '.join(sys.argv[1:])
        print(f"📝 Input sentence: '{sentence}'")
        return sentence
    else:
        # Interactive input
        print("\n" + "="*60)
        print("TASK DURATION PREDICTION")
        print("="*60)
        print("Enter a task description to predict its duration:")
        sentence = input("Task: ").strip()
        return sentence

def format_prediction_output(sentence, rf_prediction, best_prediction, rf_model_data, best_model_data):
    """Format and display prediction results"""
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Input: '{sentence}'")
    print()
    
    if rf_prediction is not None:
        print(f"🌲 Random Forest Model:")
        print(f"   Predicted Duration: {rf_prediction:.1f} minutes ({rf_prediction/60:.1f} hours)")
        if rf_model_data:
            print(f"   Training Samples: {rf_model_data.get('training_samples', 'Unknown')}")
    else:
        print("🌲 Random Forest Model: ❌ Prediction failed")
    
    print()
    
    if best_prediction is not None:
        print(f"🏆 Best Model ({best_model_data.get('model_name', 'Unknown')}):")
        print(f"   Predicted Duration: {best_prediction:.1f} minutes ({best_prediction/60:.1f} hours)")
        if best_model_data:
            print(f"   Model R² Score: {best_model_data.get('test_r2', 'Unknown'):.3f}")
            print(f"   Model MAE: {best_model_data.get('test_mae', 'Unknown'):.1f} minutes")
    else:
        print("🏆 Best Model: ❌ Prediction failed")
    
    print()
    
    # Compare predictions if both succeeded
    if rf_prediction is not None and best_prediction is not None:
        diff = abs(rf_prediction - best_prediction)
        print(f"📊 Prediction Difference: {diff:.1f} minutes")
        if diff > 30:  # If predictions differ by more than 30 minutes
            print(f"   ⚠️  Large difference detected - consider model reliability")
        
        # Show average
        avg_prediction = (rf_prediction + best_prediction) / 2
        print(f"📈 Average Prediction: {avg_prediction:.1f} minutes ({avg_prediction/60:.1f} hours)")
    
    print("="*60)

def main():
    """Main function to orchestrate prediction pipeline"""
    print("🔮 Task Duration Prediction Tool")
    print("Using both Random Forest and Best Model for comparison")
    print()
    
    # Load embedding model
    embedding_model = load_embedding_model()
    if embedding_model is None:
        return
    
    # Load prediction models
    print("\nLoading prediction models...")
    rf_model_data = load_random_forest_model()
    best_model_data = load_best_model()
    
    if rf_model_data is None and best_model_data is None:
        print("❌ No models available for prediction!")
        print("   Run the training pipeline first:")
        print("   1. python generate_embeddings.py")
        print("   2. python train_random_forest.py")
        print("   3. python comprehensive_model_search.py")
        return
    
    # Get input sentence
    sentence = get_user_input()
    if not sentence:
        print("❌ No input provided!")
        return
    
    # Generate embedding
    print(f"\n🧠 Generating embedding for: '{sentence}'")
    embedding = generate_embedding(sentence, embedding_model)
    if embedding is None:
        return
    
    print(f"✓ Generated {len(embedding)}-dimensional embedding")
    
    # Validate embedding dimensions
    if not validate_embedding_dimension(embedding, rf_model_data, best_model_data):
        print("❌ Embedding dimension validation failed")
        return
    
    # Make predictions
    print("\n🎯 Making predictions...")
    
    rf_prediction = None
    best_prediction = None
    
    if rf_model_data:
        rf_prediction = predict_with_random_forest(embedding, rf_model_data)
        if rf_prediction is not None:
            print(f"✓ Random Forest prediction: {rf_prediction:.1f} minutes")
    
    if best_model_data:
        best_prediction = predict_with_best_model(embedding, best_model_data)
        if best_prediction is not None:
            print(f"✓ Best model prediction: {best_prediction:.1f} minutes")
    
    # Display results
    format_prediction_output(sentence, rf_prediction, best_prediction, rf_model_data, best_model_data)

if __name__ == "__main__":
    main()
