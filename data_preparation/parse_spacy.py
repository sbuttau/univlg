import argparse

import pandas as pd
import spacy
from tqdm import tqdm

from dataset_stats import DEF_ANNOTATION_FILE

# Carica il modello inglese di spaCy
# Se non lo hai: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

def has_relevant_negation(text):
    doc = nlp(text.lower())
    for token in doc:
        # Cerca particelle negative (not, n't) o determinanti (no, neither)
        if token.dep_ == "neg" or token.lemma_ in ["no", "other than", "except", "but"]:
            # Puoi anche controllare a cosa è legata la negazione
            head = token.head.text
            return True
    return False


if __name__ == "__main__":
    tqdm.pandas(desc="Extracting negations")
    parser = argparse.ArgumentParser(description="Estrai descrizioni con negazioni rilevanti.")
    parser.add_argument('--annotation_file', type=str, default=DEF_ANNOTATION_FILE, help="Percorso al file CSV.")
    args = parser.parse_args()

    df = pd.read_csv(args.annotation_file)
    print(f"Loaded {len(df)} samples from {args.annotation_file}")
    print(f"Extracting samples with relevant negations...")
    
    df_negations = df[df['description'].progress_apply(has_relevant_negation)].copy()
    print(f"Found {len(df_negations)} samples with relevant negations.")
    df_negations.to_csv(f"{args.annotation_file.split('.')[0]}_negations_only.csv", index=False)
    print(f"Negation samples saved to {args.annotation_file.split('.')[0]}_negations_only.csv")