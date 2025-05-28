#!/usr/bin/env python3
"""
Simple Random Forest script to predict task duration based on sentence embeddings.
Assumes embeddings have been generated using generate_embeddings.py
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def load_embeddings_data():
    """Load the embeddings data from pickle file"""
    try:
        with open('../data/embeddings_data.pkl', 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded embeddings data: {len(data['embeddings'])} samples with {data['embedding_dimension']} dimensions")
        return data
    except FileNotFoundError:
        print("embeddings_data.pkl not found. Please run generate_embeddings.py first.")
        return None

def prepare_data(data):
    """Prepare features (embeddings) and target (duration) for training"""
    X = np.array(data['embeddings'])  # Features: sentence embeddings
    y = np.array(data['duration_minutes'])  # Target: task duration in minutes
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Duration range: {y.min():.1f} - {y.max():.1f} minutes")
    print(f"Mean duration: {y.mean():.1f} minutes")
    
    return X, y

def train_random_forest(X, y, test_size=0.3, random_state=42):
    """Train a Random Forest regressor"""
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create and train the Random Forest model
    rf_model = RandomForestRegressor(
        n_estimators=100,           # Number of trees
        max_depth=10,               # Maximum depth of trees
        min_samples_split=5,        # Minimum samples required to split
        min_samples_leaf=2,         # Minimum samples required at leaf
        random_state=random_state,
        n_jobs=-1                   # Use all available cores
    )
    
    print("\nTraining Random Forest model...")
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Print results
    print("\n" + "="*50)
    print("TRAINING RESULTS:")
    print("="*50)
    print(f"Training MSE:  {train_mse:.2f}")
    print(f"Test MSE:      {test_mse:.2f}")
    print(f"Training MAE:  {train_mae:.2f} minutes")
    print(f"Test MAE:      {test_mae:.2f} minutes")
    print(f"Training R²:   {train_r2:.3f}")
    print(f"Test R²:       {test_r2:.3f}")
    print("="*50)
    
    # Feature importance (top 10)
    feature_importance = rf_model.feature_importances_
    top_features = np.argsort(feature_importance)[-10:][::-1]
    
    print("\nTop 10 Most Important Embedding Dimensions:")
    for i, feature_idx in enumerate(top_features):
        print(f"{i+1:2d}. Dimension {feature_idx:3d}: {feature_importance[feature_idx]:.4f}")
    
    return rf_model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred

def create_visualizations(y_train, y_test, y_train_pred, y_test_pred):
    """Create visualizations of the model performance"""
    
    plt.figure(figsize=(15, 5))
    
    # Actual vs Predicted - Training
    plt.subplot(1, 3, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.6, color='blue', label='Training')
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    plt.xlabel('Actual Duration (minutes)')
    plt.ylabel('Predicted Duration (minutes)')
    plt.title('Training Set: Actual vs Predicted')
    plt.legend()
    
    # Actual vs Predicted - Test
    plt.subplot(1, 3, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.6, color='green', label='Test')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Duration (minutes)')
    plt.ylabel('Predicted Duration (minutes)')
    plt.title('Test Set: Actual vs Predicted')
    plt.legend()
    
    # Residuals plot
    plt.subplot(1, 3, 3)
    residuals_test = y_test - y_test_pred
    plt.scatter(y_test_pred, residuals_test, alpha=0.6, color='orange')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Duration (minutes)')
    plt.ylabel('Residuals (minutes)')
    plt.title('Residuals Plot (Test Set)')
    
    plt.tight_layout()
    plt.savefig('../results/random_forest_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'random_forest_results.png'")
    
def save_model(model, data):
    """Save the trained model and some metadata"""
    model_data = {
        'model': model,
        'embedding_dimension': data['embedding_dimension'],
        'training_samples': len(data['embeddings'])
    }
    
    with open('../models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model saved as 'random_forest_model.pkl'")
    
def predict_new_task(model, embedding):
    """Example function to predict duration for a new task embedding"""
    prediction = model.predict([embedding])[0]
    return prediction

def main():
    print("Random Forest Training Script for Task Duration Prediction")
    print("=" * 60)
    
    # Load embeddings data
    data = load_embeddings_data()
    if data is None:
        return
    
    # Prepare features and target
    X, y = prepare_data(data)
    
    # Train the model
    model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred = train_random_forest(X, y)
    
    # Create visualizations
    create_visualizations(y_train, y_test, y_train_pred, y_test_pred)
    
    # Save the model
    save_model(model, data)
    
    # Example prediction
    print("\nExample prediction for first test sample:")
    example_embedding = X_test[0]
    predicted_duration = predict_new_task(model, example_embedding)
    actual_duration = y_test[0]
    print(f"Predicted: {predicted_duration:.1f} minutes")
    print(f"Actual:    {actual_duration:.1f} minutes")
    print(f"Error:     {abs(predicted_duration - actual_duration):.1f} minutes")
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("Files created:")
    print("- models/random_forest_model.pkl (trained model)")
    print("- results/random_forest_results.png (performance plots)")
    print("="*60)

if __name__ == "__main__":
    main()
