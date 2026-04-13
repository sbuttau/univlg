import os
import ast
import json
import pandas as pd
import nltk
from nltk import sent_tokenize, word_tokenize

# --- CONFIGURAZIONE ---
DEF_ANNOTATION_FILE = "data/refer_it_3d/ScanRefer_filtered_val_ScanEnts3D_val.csv" 
nltk.download('punkt')

def get_scene_category_counts(df):
    """Conta quante istanze di ogni categoria ci sono in ogni scena."""
    scene_counts = {}
    for entry in df.to_dict(orient='records'):
        scene_id = entry['scan_id']
        category = entry['object_name']
        obj_id = entry['object_id']
        
        if scene_id not in scene_counts:
            scene_counts[scene_id] = {}
        if category not in scene_counts[scene_id]:
            scene_counts[scene_id][category] = set()
        
        scene_counts[scene_id][category].add(obj_id)
    return scene_counts

def get_n_sentences_data(row, n):
    """
    Taglia in modo sincrono description, tokens ed entities.
    """
    description = str(row['description'])
    
    # Parsing sicuro dei token (da stringa a lista)
    try:
        all_tokens = ast.literal_eval(row['tokens'])
    except:
        all_tokens = str(row['tokens']).split()

    sentences = sent_tokenize(description)
    selected_sentences = sentences[:n]
    
    # 1. Nuova Description
    new_description = " ".join(selected_sentences)
    
    # 2. Nuovi Tokens
    # Calcoliamo quanti token appartengono alle frasi selezionate
    word_count = 0
    for sent in selected_sentences:
        word_count += len(word_tokenize(sent))
    new_tokens = all_tokens[:word_count]
    
    # 3. Nuove Entities (Filtriamo i riferimenti fuori dal range dei nuovi token)
    try:
        raw_entities = ast.literal_eval(row['entities'])
        new_entities = []
        for ent in raw_entities:
            # ent[0] è la lista dei token_id. Teniamo solo l'entità se TUTTI i suoi
            # token sono dentro il nuovo limite word_count.
            token_ids = ent[0]
            if all(tid < word_count for tid in token_ids):
                new_entities.append(ent)
    except:
        new_entities = row['entities'] # Fallback se fallisce il parsing

    return pd.Series([new_description, new_tokens, str(new_entities)])

if __name__ == "__main__":
    if not os.path.exists(DEF_ANNOTATION_FILE):
        print(f"Errore: File {DEF_ANNOTATION_FILE} non trovato.")
        exit()

    df = pd.read_csv(DEF_ANNOTATION_FILE)
    print(f"Caricati {len(df)} campioni.")

    # Analisi unicità (Target Unique vs Multiple)
    scene_counts = get_scene_category_counts(df)
    df['is_unique'] = df.apply(lambda r: len(scene_counts[r['scan_id']].get(r['object_name'], [])) == 1, axis=1)
    
    # Ci concentriamo sui campioni Unique (come nel tuo script precedente)
    df_unique = df[df['is_unique'] == True].copy()
    print(f"Campioni Unique: {len(df_unique)}")

    # Conta frasi
    df_unique['num_sentences'] = df_unique['description'].apply(lambda x: len(sent_tokenize(str(x))))
    
    # Filtriamo i "Long Samples" (> 3 frasi)
    long_sent = df_unique[df_unique['num_sentences'] > 3].copy()
    
    if not long_sent.empty:
        print(f"Elaborazione di {len(long_sent)} campioni lunghi...")
        base_name = DEF_ANNOTATION_FILE.split('/')[-1].replace('.csv', '')

        # Generazione delle 4 versioni
        for v in range(1, 5):
            df_v = long_sent.copy()
            
            if v < 4:
                # Applichiamo il taglio per v1, v2, v3
                df_v[['description', 'tokens', 'entities']] = df_v.apply(
                    lambda row: get_n_sentences_data(row, v), axis=1
                )
                suffix = f"long_v{v}_{v}sent"
            else:
                # v4 è l'originale completo
                suffix = "long_v4_all"

            output_path = f"data/refer_it_3d/{base_name}_{suffix}.csv"
            
            # Pulizia colonne extra prima del salvataggio
            if 'num_sentences' in df_v.columns: df_v.drop(columns=['num_sentences'], inplace=True)
            if 'is_unique' in df_v.columns: df_v.drop(columns=['is_unique'], inplace=True)
            
            df_v.to_csv(output_path, index=False)
            print(f"Salvato: {output_path}")

        print("\nProcesso completato con successo.")
    else:
        print("Nessun campione con più di 3 frasi trovato.")