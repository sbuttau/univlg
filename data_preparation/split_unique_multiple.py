import argparse
import os
import ast
import json
import pandas as pd
import nltk
from nltk import sent_tokenize, word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURAZIONE ---
DEF_ANNOTATION_FILE = "data/refer_it_3d/ScanRefer_filtered_val_ScanEnts3D_val.csv" 
nltk.download('punkt')

def extract_target_category(row):
    try:
        entities_list = ast.literal_eval(row['entities'])
        target_id = int(row['object_id'])
        
        # Cicla su tutte le entità estratte nella frase
        for entity in entities_list:
            # entity[1] contiene la lista di stringhe tipo ['5_chair'] o ['0_floor', '21_wall']
            for item in entity[1]:
                # Separiamo l'ID dal nome (es. "5_chair" -> ["5", "chair"])
                parts = item.split('_')
                if parts[0].isdigit() and int(parts[0]) == target_id:
                    # Abbiamo trovato l'entità che ha lo stesso ID del target!
                    return parts[1]
    except Exception:
        pass
    
    # Fallback di sicurezza: se la lista entities è corrotta o vuota,
    # usiamo object_name ma pulito/normalizzato (es. armchair -> chair)
    return row['object_name']

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

# Creiamo una funzione per categorizzare la lunghezza
def categorize_sentences(n):
    if n == 1: return '1 Sent'
    if n == 2: return '2 Sents'
    if n == 3: return '3 Sents'
    return '>3 Sents'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara dataset con diverse lunghezze di descrizione.")
    parser.add_argument('--annotation_file', type=str, default=DEF_ANNOTATION_FILE, help="Percorso al file CSV di annotazione.")
    parser.add_argument('--multiple', action='store_true', help="Se attivo, processa i campioni Multiple invece degli Unique, altrimenti entrambi.")
    parser.add_argument('--stats', action='store_true', help="Se attivo, estrae statistiche.")
    parser.add_argument('--save', action='store_true', help="Se attivo, salva i file CSV generati.")
    args = parser.parse_args()

    if not os.path.exists(args.annotation_file):
        print(f"Errore: File {args.annotation_file} non trovato.")
        exit()

    # df = pd.read_csv(args.annotation_file)
    # print(f"Caricati {len(df)} campioni.")

    # # Analisi unicità (Target Unique vs Multiple)
    # scene_counts = get_scene_category_counts(df)
    # df['is_unique'] = df.apply(lambda r: len(scene_counts[r['scan_id']].get(r['object_name'], [])) == 1, axis=1)
    # df['is_multiple'] = df.apply(lambda r: len(scene_counts[r['scan_id']].get(r['object_name'], [])) > 1, axis=1)

    # # Ci concentriamo sui campioni Unique (come nel tuo script precedente)
    # df_unique = df[df['is_unique'] == True].copy()
    # df_multiple = df[df['is_multiple'] == True].copy()
    # print(f"Campioni Unique: {len(df_unique)}")
    # print(f"Campioni Multiple: {len(df_multiple)}")

    # # Conta frasi
    # df_unique['num_sentences'] = df_unique['description'].apply(lambda x: len(sent_tokenize(str(x))))
    # df_multiple['num_sentences'] = df_multiple['description'].apply(lambda x: len(sent_tokenize(str(x))))
    df = pd.read_csv(args.annotation_file)

    # Estrai la categoria ufficiale per ogni riga
    df['clean_category'] = df.apply(extract_target_category, axis=1)

    # Mappa ogni scena a quali oggetti reali (object_id) appartengono a ciascuna categoria
    scene_manifest = {}
    for _, row in df.iterrows():
        sid = row['scene_id']
        cat = row['clean_category']
        obj_id = row['object_id']
        
        if not cat: 
            continue
            
        if sid not in scene_manifest:
            scene_manifest[sid] = {}
        if cat not in scene_manifest[sid]:
            scene_manifest[sid][cat] = set()
            
        # Usiamo un set per raccogliere gli ID unici degli oggetti reali presenti nella scena
        scene_manifest[sid][cat].add(obj_id)

    # Calcola l'unicità basandoti sul numero di OGGETTI REALI in quella stanza per quella categoria
    df['is_unique'] = df.apply(lambda r: len(scene_manifest.get(r['scene_id'], {}).get(r['clean_category'], [])) == 1 if r['clean_category'] else False, axis=1)
    df['is_multiple'] = ~df['is_unique']

    # Ora puoi separare i dataframe in modo sicuro
    df_unique = df[df['is_unique']].copy()
    df_multiple = df[df['is_multiple']].copy()
    # Store splits
    splits = {
        'unique': df_unique,
        'multiple': df_multiple
    }
    unique_output_path = f'data/refer_it_3d/ScanRefer_filtered_val_ScanEnts3D_val_unique.csv'
    multiple_output_path = f'data/refer_it_3d/ScanRefer_filtered_val_ScanEnts3D_val_multiple.csv'
    df_unique.to_csv(unique_output_path, index=False)
    print(f"Salvato: {unique_output_path}")
    df_multiple.to_csv(multiple_output_path, index=False)
    print(f"Salvato: {multiple_output_path}")
    
    if False:  # Debugging - what follows is a statistics summary of sentence counts
        # Filtriamo i "Long Samples" (> 3 frasi)
        long_sent = df_unique[df_unique['num_sentences'] > 3].copy()
        long_sent_multiple = df_multiple[df_multiple['num_sentences'] > 3].copy()

        print(f"Campioni Unique con più di 3 frasi: {len(long_sent)}"
            f"\nCampioni Multiple con più di 3 frasi: {len(long_sent_multiple)}")
        
        if not long_sent.empty:
            long_sent = long_sent_multiple

            print(f"Elaborazione di {len(long_sent)} campioni unique e {len(long_sent_multiple)} campioni multiple...")
            base_name = DEF_ANNOTATION_FILE.split('/')[-1].replace('.csv', '')

            # Generazione delle 4 versioni
            for v in range(1, 5):
                df_v = long_sent.copy()
                df_v_multiple = long_sent_multiple.copy()
                
                if v < 4:
                    # Applichiamo il taglio per v1, v2, v3
                    df_v[['description', 'tokens', 'entities']] = df_v.apply(
                        lambda row: get_n_sentences_data(row, v), axis=1
                    )
                    df_v_multiple[['description', 'tokens', 'entities']] = df_v_multiple.apply(
                        lambda row: get_n_sentences_data(row, v), axis=1
                    )
                    suffix_multiple = f"long_v{v}_{v}sent_multiple"
                    suffix = f"long_v{v}_{v}sent"
                else:
                    suffix_multiple = f"long_v{v}_all_multiple"
                    suffix = "long_v4_all"

                if args.save:
                    output_path = f"data/refer_it_3d/{base_name}_{suffix}.csv"
                    output_path_multiple = f"data/refer_it_3d/{base_name}_{suffix_multiple}.csv"

                if args.save:
                    # Pulizia colonne extra prima del salvataggio
                    if 'num_sentences' in df_v.columns: df_v.drop(columns=['num_sentences'], inplace=True)
                    if 'is_unique' in df_v.columns: df_v.drop(columns=['is_unique'], inplace=True)
                    if 'is_multiple' in df_v_multiple.columns: df_v_multiple.drop(columns=['is_multiple'], inplace=True)
                    if 'num_sentences' in df_v_multiple.columns: df_v_multiple.drop(columns=['num_sentences'], inplace=True)
                    
                    df_v.to_csv(output_path, index=False)
                    print(f"Salvato: {output_path}")
                    df_v_multiple.to_csv(output_path_multiple, index=False)
                    print(f"Salvato: {output_path_multiple}")

            print("\nProcesso completato con successo.")
        else:
            print("Nessun campione con più di 3 frasi trovato.")
            
        # --- GENERAZIONE STATISTICHE E PLOT ---
        print("\nGenerazione statistiche e grafici...")
        
        # 1. Prepariamo i dati per il plot

        # Uniamo i due dataframe per avere un'unica vista statistica
        df_unique['category'] = 'Unique'
        df_multiple['category'] = 'Multiple'
        full_stats_df = pd.concat([df_unique, df_multiple])
        
        full_stats_df['Length Category'] = full_stats_df['num_sentences'].apply(categorize_sentences)

        # 2. Creazione della tabella pivot per le statistiche testuali
        stats_table = pd.crosstab(full_stats_df['category'], full_stats_df['Length Category'])
        # Ordiniamo le colonne correttamente
        column_order = ['1 Sent', '2 Sents', '3 Sents', '>3 Sents']
        stats_table = stats_table.reindex(columns=[c for c in column_order if c in stats_table.columns])
        
        print("\nTabella Riassuntiva:")
        print(stats_table)

        # 3. Plotting con Seaborn
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        # Creiamo il grafico a barre raggruppate
        ax = sns.countplot(
            data=full_stats_df, 
            x='category', 
            hue='Length Category', 
            hue_order=column_order,
            palette='viridis'
        )

        # Aggiungiamo i numeri sopra le barre per precisione
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f'{int(p.get_height())}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'center', 
                            xytext = (0, 9), 
                            textcoords = 'offset points',
                            fontsize=9)

        plt.title(f'Dataset info {args.annotation_file}: Unique vs Multiple')
        plt.xlabel('Tipo di Target nella Scena')
        plt.ylabel('Numero di Annotazioni')
        plt.legend(title='Lunghezza Description', loc='upper right')
        
        # Salva il grafico
        plt.tight_layout()
        plt.savefig("data_preparation/dataset_distribution_plot.png")
        print("\nGrafico salvato in: data_preparation/dataset_distribution_plot.png")
        plt.show()