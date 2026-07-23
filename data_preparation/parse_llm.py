import pandas as pd
import json
from tqdm import tqdm # Per vedere la barra di caricamento
import os
from dotenv import load_dotenv
import ollama

load_dotenv()

def analyze_description(description, target_name):
    prompt = f"""
        Target Object to find: "{target_name}"
        Description to analyze: "{description}"

        STRICT RULES:
        1. Search the description for the word "{target_name}" or very similar nouns.
        2. If the noun is present (e.g., "the {target_name} is..."), then 'is_implicit' MUST BE false.
        3. If the description only uses "it", "this", or "the object", then 'is_implicit' MUST BE true.

        Return ONLY a JSON object:
        {{
        "word_found_in_text": "write the word you found here, or 'none'",
        "is_implicit": true/false,
        "has_negation": true/false,
        "spatial_complexity": "low/medium/high",
        "explanation": "why did you choose those values? Write a brief explanation."
        }}
    """
    try:
        # Chiamata locale a Ollama
        response = ollama.chat(
            model='llama3:8b',
            messages=[{'role': 'user', 'content': prompt}],
            format='json' 
        )
        
        # Il contenuto è in response['message']['content']
        return json.loads(response['message']['content'])
    except Exception as e:
        print(f"Errore locale: {e}")
        return None
    
if __name__ == "__main__":

    df_unique = pd.read_csv("data/refer_it_3d/ScanRefer_filtered_val_ScanEnts3D_val.csv")
    # Esempio su un subset di 10 campioni per testare
    subset = df_unique.head(10).copy()

    results = []
    for idx, row in tqdm(subset.iterrows(), total=len(subset)):
        res = analyze_description(row['description'], row['object_name'])
        results.append(res if res else {"is_implicit": None, "has_negation": None, "spatial_complexity": None})

    # 4. Salvataggio
    analysis_df = pd.DataFrame(results)
    final_df = pd.concat([subset.reset_index(drop=True), analysis_df], axis=1)
    final_df.to_csv("data/test_llm/scanrefer_unique_analyzed.csv", index=False)

    print("Analisi completata!")