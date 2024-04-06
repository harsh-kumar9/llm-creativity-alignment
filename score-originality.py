import os
import pandas as pd
import requests

# Your provided function
def call_llm_api(prompt, idea):
    """
    Calls the LLM API to calculate the originality of an idea based on a prompt.
    
    Args:
        prompt (str): The prompt to use with the API.
        idea (str): The idea to calculate originality for.
    
    Returns:
        float: Originality score from 1-5 or an error message.
    """
    api_endpoint = "https://openscoring.du.edu/llm"
    params = {
        "model": "ocsai-1.5",
        "prompt": prompt,
        "input": [idea],
        "input_type": "csv",
        "elab_method": "none",
        "language": "English",
        "task": "uses",
    }
    
    if idea == "":
        return -1.0

    response = requests.get(api_endpoint, params=params)

    if response.status_code == 200:
        data = response.json()
        originality_score = data["scores"][0]["originality"]
        return originality_score
    else:
        return {"error": f"API call failed with status code {response.status_code}"}

def process_csv_files(folder_path):
    """
    Processes each CSV file in the given folder, calculating the originality of each idea and saving the updated CSV.
    
    Args:
        folder_path (str): The path to the folder containing the CSV files.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            
            # Check if the necessary columns are present
            if 'item' in df.columns and 'idea' in df.columns:
                print(f"Processing file: {filename}")
                df['originality'] = df.apply(lambda x: call_llm_api(x['item'], x['idea']), axis=1)
                df.to_csv(file_path, index=False)
                print(f"Updated file saved: {filename}")
            else:
                print(f"Skipped file (required columns missing): {filename}")

folder_path = '/Users/harsh/Desktop/llm-creativity-alignment/datasets'
process_csv_files(folder_path)
