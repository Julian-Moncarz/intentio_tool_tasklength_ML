#!/usr/bin/env python3
"""
Script to generate sentence embeddings from logs.csv
Outputs: original intent text, embeddings, and duration
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import json
import os
import torch

# Fix for torch device compatibility
if not hasattr(torch, 'get_default_device'):
    def get_default_device():
        return 'cpu' if not torch.cuda.is_available() else 'cuda'
    torch.get_default_device = get_default_device

def preprocess_for_sentence_embeddings(text):
    """Basic preprocessing for sentence embeddings"""
    if text is None or text == "":
        return ""
    
    # Basic cleaning
    text = str(text).strip()  # Convert to string and remove whitespace
    text = text.replace("\0", "")  # Remove null bytes if present
    
    return text

def main():
    print("Loading logs.csv...")
    
    # Load the CSV file
    df = pd.read_csv('../data/logs.csv')
    print(f"Loaded {len(df)} rows")
    
    # Clean and filter data
    print("Cleaning data...")
    
    # Clean the Intent column
    df['Intent_clean'] = df['Intent'].apply(preprocess_for_sentence_embeddings)
    
    # Convert Duration to numeric, handling any errors
    df['Duration_numeric'] = pd.to_numeric(df['Duration(min)'], errors='coerce')
    
    # Filter out rows with missing or invalid data
    valid_data = df.dropna(subset=['Intent_clean', 'Duration_numeric'])
    valid_data = valid_data[valid_data['Intent_clean'] != '']
    valid_data = valid_data[valid_data['Duration_numeric'] > 0]  # Remove zero or negative durations
    
    print(f"After cleaning: {len(valid_data)} valid rows")
    
    if len(valid_data) == 0:
        print("No valid data found! Check your CSV file.")
        return
    
    # Load sentence transformer model
    print("Loading Sentence Transformer model...")
    model = SentenceTransformer('all-mpnet-base-v2')
    print("Model loaded successfully")

    
    # Generate embeddings
    print("Generating embeddings...")
    intents = valid_data['Intent_clean'].tolist()
    embeddings = model.encode(intents, show_progress_bar=True)
    
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Prepare output data
    output_data = {
        'original_intent': valid_data['Intent'].tolist(),
        'clean_intent': valid_data['Intent_clean'].tolist(),
        'duration_minutes': valid_data['Duration_numeric'].tolist(),
        'embeddings': embeddings.tolist(),  # Convert numpy array to list for JSON serialization
        'embedding_dimension': embeddings.shape[1]
    }
    
    # Save as JSON (preserves embedding arrays better than CSV)
    print("Saving embeddings to embeddings_data.json...")
    with open('../data/embeddings_data.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    # Also save as pickle for easy loading in Python
    print("Saving embeddings to embeddings_data.pkl...")
    with open('../data/embeddings_data.pkl', 'wb') as f:
        pickle.dump(output_data, f)
    
    # Create a summary CSV for easy viewing
    print("Creating summary CSV...")
    summary_df = pd.DataFrame({
        'original_intent': output_data['original_intent'],
        'clean_intent': output_data['clean_intent'],
        'duration_minutes': output_data['duration_minutes'],
        'embedding_preview': [str(emb[:5]) + '...' for emb in output_data['embeddings']]  # Show first 5 dimensions
    })
    
    summary_df.to_csv('../data/embeddings_summary.csv', index=False)
    
    print("\n" + "="*50)
    print("RESULTS:")
    print(f"✓ Processed {len(valid_data)} valid intentions")
    print(f"✓ Generated {embeddings.shape[1]}-dimensional embeddings")
    print(f"✓ Saved full data to: embeddings_data.json and embeddings_data.pkl")
    print(f"✓ Saved summary to: embeddings_summary.csv")
    print("="*50)
    
    # Show some examples
    print("\nSample data:")
    print(summary_df.head().to_string(index=False))
    
    print(f"\nEmbedding statistics:")
    print(f"- Min value: {np.min(embeddings):.4f}")
    print(f"- Max value: {np.max(embeddings):.4f}")
    print(f"- Mean value: {np.mean(embeddings):.4f}")
    print(f"- Std value: {np.std(embeddings):.4f}")

if __name__ == "__main__":
    main()
