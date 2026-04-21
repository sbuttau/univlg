import pandas as pd
import spacy

# Carica il modello inglese di spaCy
# Se non lo hai: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

def has_relevant_negation(text):
    doc = nlp(text.lower())
    for token in doc:
        # Cerca particelle negative (not, n't) o determinanti (no, neither)
        if token.dep_ == "neg" or token.lemma_ in ["no", "never", "except", "but"]:
            # Puoi anche controllare a cosa è legata la negazione
            head = token.head.text
            return True
    return False

# Applica al tuo dataset
df_unique = pd.read_csv("data/refer_it_3d/ScanRefer_filtered_val_ScanEnts3D_val_long_v4_all_multiple.csv")
df_negations = df_unique[df_unique['description'].apply(has_relevant_negation)].copy()

print(f"Trovati {len(df_negations)} campioni con negazioni.")
df_negations.to_csv("scanrefer_negations_only.csv", index=False)