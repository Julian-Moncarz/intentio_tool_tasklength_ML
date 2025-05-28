#!/usr/bin/env python3
"""
Script to analyze sentence embedding similarity by finding closest vectors
and displaying their natural language meanings to validate semantic coherence.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

def load_embeddings():
    """Load embeddings data from pickle file"""
    try:
        with open('../data/embeddings_data.pkl', 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Loaded {len(data['embeddings'])} embeddings")
        return data
    except FileNotFoundError:
        print("❌ embeddings_data.pkl not found. Run generate_embeddings.py first.")
        return None

def calculate_similarity_matrix(embeddings, metric='cosine'):
    """Calculate pairwise similarity matrix"""
    if metric == 'cosine':
        # Cosine similarity (higher = more similar)
        return cosine_similarity(embeddings)
    elif metric == 'euclidean':
        # Euclidean distance (lower = more similar)
        distances = euclidean_distances(embeddings)
        # Convert to similarity (higher = more similar)
        max_distance = np.max(distances)
        return max_distance - distances
    else:
        raise ValueError("Metric must be 'cosine' or 'euclidean'")

def find_most_similar_pairs(similarity_matrix, texts, n_pairs=10):
    """Find the most similar pairs of texts (excluding perfect matches)"""
    # Get upper triangle indices (avoid diagonal and duplicates)
    indices = np.triu_indices_from(similarity_matrix, k=1)
    
    # Get similarity scores for these pairs
    similarities = similarity_matrix[indices]
    
    # Filter out perfect matches (similarity = 1.0) to avoid showing duplicates
    non_perfect_mask = similarities < 1.0
    filtered_similarities = similarities[non_perfect_mask]
    filtered_indices = (indices[0][non_perfect_mask], indices[1][non_perfect_mask])
    
    # Sort by similarity (highest first)
    sorted_indices = np.argsort(filtered_similarities)[::-1]
    
    pairs = []
    for i in range(min(n_pairs, len(sorted_indices))):
        idx = sorted_indices[i]
        row, col = filtered_indices[0][idx], filtered_indices[1][idx]
        pairs.append({
            'similarity': filtered_similarities[idx],
            'text1': texts[row],
            'text2': texts[col],
            'index1': row,
            'index2': col
        })
    
    return pairs

def find_least_similar_pairs(similarity_matrix, texts, n_pairs=5):
    """Find the least similar pairs of texts"""
    # Get upper triangle indices (avoid diagonal and duplicates)
    indices = np.triu_indices_from(similarity_matrix, k=1)
    
    # Get similarity scores for these pairs
    similarities = similarity_matrix[indices]
    
    # Sort by similarity (lowest first)
    sorted_indices = np.argsort(similarities)
    
    pairs = []
    for i in range(min(n_pairs, len(sorted_indices))):
        idx = sorted_indices[i]
        row, col = indices[0][idx], indices[1][idx]
        pairs.append({
            'similarity': similarities[idx],
            'text1': texts[row],
            'text2': texts[col],
            'index1': row,
            'index2': col
        })
    
    return pairs

def find_nearest_neighbors(embeddings, texts, query_idx, n_neighbors=5):
    """Find nearest neighbors for a specific text"""
    query_embedding = embeddings[query_idx].reshape(1, -1)
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Sort by similarity (excluding the query itself)
    indices = np.argsort(similarities)[::-1]
    
    neighbors = []
    count = 0
    for i, idx in enumerate(indices):
        if idx != query_idx:  # Skip the query itself
            neighbors.append({
                'similarity': similarities[idx],
                'text': texts[idx],
                'index': idx
            })
            count += 1
            if count >= n_neighbors:
                break
    
    return neighbors

def analyze_similarity_by_category(data):
    """Analyze similarity patterns by task categories"""
    texts = data['clean_intent']
    embeddings = np.array(data['embeddings'])
    durations = data['duration_minutes']
    
    # Create categories based on keywords
    categories = {}
    for i, text in enumerate(texts):
        text_lower = text.lower()
        if any(word in text_lower for word in ['bio', 'isu', 'biology']):
            category = 'Biology/ISU'
        elif any(word in text_lower for word in ['ml', 'machine learning', 'model']):
            category = 'Machine Learning'
        elif any(word in text_lower for word in ['todo', 'to-do', 'list']):
            category = 'Todo Lists'
        elif any(word in text_lower for word in ['test', 'testing']):
            category = 'Testing'
        elif any(word in text_lower for word in ['flashcard', 'myth']):
            category = 'Study/Flashcards'
        elif any(word in text_lower for word in ['report', 'mosfet']):
            category = 'Reports'
        else:
            category = 'Other'
        
        if category not in categories:
            categories[category] = []
        categories[category].append(i)
    
    print("\n" + "="*60)
    print("CATEGORY ANALYSIS")
    print("="*60)
    
    for category, indices in categories.items():
        if len(indices) > 1:  # Only analyze categories with multiple items
            print(f"\n{category} ({len(indices)} items):")
            category_embeddings = embeddings[indices]
            category_texts = [texts[i] for i in indices]
            
            # Calculate within-category similarity
            sim_matrix = cosine_similarity(category_embeddings)
            avg_similarity = np.mean(sim_matrix[np.triu_indices_from(sim_matrix, k=1)])
            
            print(f"  Average within-category similarity: {avg_similarity:.3f}")
            print("  Items:")
            for i, text in enumerate(category_texts):
                duration = durations[indices[i]]
                print(f"    - {text} ({duration} min)")

def create_similarity_heatmap(similarity_matrix, texts, title="Embedding Similarity Heatmap"):
    """Create a heatmap visualization of similarity matrix"""
    plt.figure(figsize=(12, 10))
    
    # Truncate text labels for readability
    labels = [text[:30] + "..." if len(text) > 30 else text for text in texts]
    
    sns.heatmap(similarity_matrix, 
                xticklabels=labels, 
                yticklabels=labels,
                cmap='viridis',
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={'label': 'Cosine Similarity'})
    
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('embedding_similarity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("="*60)
    print("EMBEDDING SIMILARITY ANALYSIS")
    print("="*60)
    
    # Load data
    data = load_embeddings()
    if data is None:
        return
    
    embeddings = np.array(data['embeddings'])
    texts = data['clean_intent']
    original_texts = data['original_intent']
    durations = data['duration_minutes']
    
    print(f"Analyzing {len(embeddings)} embeddings with {embeddings.shape[1]} dimensions")
    
    # Calculate similarity matrix
    print("\nCalculating cosine similarity matrix...")
    similarity_matrix = calculate_similarity_matrix(embeddings, metric='cosine')
    
    # Find most similar pairs
    print("\n" + "="*60)
    print("MOST SIMILAR TASK PAIRS (Semantic Validation)")
    print("="*60)
    
    similar_pairs = find_most_similar_pairs(similarity_matrix, texts, n_pairs=10)
    
    for i, pair in enumerate(similar_pairs, 1):
        duration1 = durations[pair['index1']]
        duration2 = durations[pair['index2']]
        print(f"\n{i}. Similarity: {pair['similarity']:.3f}")
        print(f"   Text 1: '{pair['text1']}' ({duration1} min)")
        print(f"   Text 2: '{pair['text2']}' ({duration2} min)")
        
        # Check if durations are similar too
        duration_diff = abs(duration1 - duration2)
        if duration_diff <= 5:
            print(f"   ✓ Similar durations (diff: {duration_diff} min)")
        elif duration_diff > 15:
            print(f"   ⚠ Very different durations (diff: {duration_diff} min)")
    
    # Find least similar pairs
    print("\n" + "="*60)
    print("LEAST SIMILAR TASK PAIRS")
    print("="*60)
    
    dissimilar_pairs = find_least_similar_pairs(similarity_matrix, texts, n_pairs=5)
    
    for i, pair in enumerate(dissimilar_pairs, 1):
        duration1 = durations[pair['index1']]
        duration2 = durations[pair['index2']]
        print(f"\n{i}. Similarity: {pair['similarity']:.3f}")
        print(f"   Text 1: '{pair['text1']}' ({duration1} min)")
        print(f"   Text 2: '{pair['text2']}' ({duration2} min)")
    
    # Analyze nearest neighbors for a few examples
    print("\n" + "="*60)
    print("NEAREST NEIGHBORS ANALYSIS")
    print("="*60)
    
    # Pick a few interesting examples for neighbor analysis
    example_indices = []
    for i, text in enumerate(texts):
        text_lower = text.lower()
        if 'bio isu' in text_lower and len(example_indices) < 3:
            example_indices.append(i)
        elif 'todo' in text_lower and len(example_indices) < 3:
            example_indices.append(i)
        elif 'test' in text_lower and len(example_indices) < 3:
            example_indices.append(i)
    
    # If we don't have enough examples, pick the first few
    if len(example_indices) < 3:
        example_indices = list(range(min(3, len(texts))))
    
    for query_idx in example_indices:
        query_text = texts[query_idx]
        query_duration = durations[query_idx]
        
        print(f"\nQuery: '{query_text}' ({query_duration} min)")
        print("Nearest neighbors:")
        
        neighbors = find_nearest_neighbors(embeddings, texts, query_idx, n_neighbors=3)
        for j, neighbor in enumerate(neighbors, 1):
            neighbor_duration = durations[neighbor['index']]
            print(f"  {j}. Similarity: {neighbor['similarity']:.3f}")
            print(f"     Text: '{neighbor['text']}' ({neighbor_duration} min)")
    
    # Category analysis
    analyze_similarity_by_category(data)
    
    # Overall statistics
    print("\n" + "="*60)
    print("OVERALL SIMILARITY STATISTICS")
    print("="*60)
    
    # Remove diagonal (self-similarities)
    upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    
    print(f"Average pairwise similarity: {np.mean(upper_triangle):.3f}")
    print(f"Median pairwise similarity: {np.median(upper_triangle):.3f}")
    print(f"Standard deviation: {np.std(upper_triangle):.3f}")
    print(f"Min similarity: {np.min(upper_triangle):.3f}")
    print(f"Max similarity: {np.max(upper_triangle):.3f}")
    
    # Create visualization
    if len(embeddings) <= 20:  # Only create heatmap for smaller datasets
        print("\nCreating similarity heatmap...")
        create_similarity_heatmap(similarity_matrix, texts)
    else:
        print(f"\nDataset too large ({len(embeddings)} items) for heatmap visualization.")
        print("Consider creating heatmap for a subset of the data.")
    
    print("\n" + "="*60)
    print("SEMANTIC VALIDATION SUMMARY")
    print("="*60)
    print("✓ Check if similar tasks have high similarity scores")
    print("✓ Check if dissimilar tasks have low similarity scores") 
    print("✓ Look for semantic clusters (Bio ISU, Todo lists, etc.)")
    print("✓ Verify that embeddings capture meaningful relationships")
    print("="*60)

if __name__ == "__main__":
    main()
