import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from itertools import combinations
import numpy as np

# Initialize the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')

def calculate_diversity_score(ideas):
    """
    Calculate the median diversity score for a list of ideas.
    
    Args:
        ideas (list of str): The ideas to calculate diversity for.
    
    Returns:
        float: The median diversity score for the given ideas.
    """
    embeddings = model.encode(ideas)
    scores = []

    # Calculate pairwise cosine distance for each unique pair of ideas
    for emb1, emb2 in combinations(embeddings, 2):
        scores.append(cosine(emb1, emb2))
    
    # Return the median of the calculated scores
    return np.median(scores) if scores else 0

def process_csv_files(folder_path):
    """
    Processes each CSV file in the given folder, calculating the diversity of ideas for each session and item combination.
    
    Args:
        folder_path (str): The path to the folder containing the CSV files.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            
            # Check if the necessary columns are present
            if 'session_id' in df.columns and 'item' in df.columns and 'idea' in df.columns:
                print(f"Processing file: {filename}")
                
                # Calculate diversity score for each group
                df['diversity'] = df.groupby(['session_id', 'item'])['idea'].transform(lambda x: calculate_diversity_score(x.tolist()))
                
                df.to_csv(file_path, index=False)
                print(f"Updated file saved: {filename}")
            else:
                print(f"Skipped file (required columns missing): {filename}")

# Example usage:
# Replace '/path/to/your/folder' with the actual path to your folder containing the CSV files
folder_path = '/Users/harsh/Desktop/llm-creativity-alignment/datasets'
process_csv_files(folder_path)
