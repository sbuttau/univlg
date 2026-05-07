import argparse
import os
import pandas as pd
import nltk
from nltk import sent_tokenize, word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from split_unique_multiple import categorize_sentences
# --- CONFIGURAZIONE ---
DEF_ANNOTATION_FILE = "data/refer_it_3d/ScanRefer_filtered_val_ScanEnts3D_val.csv" 
nltk.download('punkt')

def categorize_tokens(n):
    """Categorizza la lunghezza della descrizione in base ai token."""
    if n <= 10: return '0-10'
    if n <= 20: return '11-20'
    if n <= 30: return '21-30'
    if n <= 40: return '31-40'
    return '>40'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analisi della distribuzione dei token nel dataset.")
    parser.add_argument('--annotation_file', type=str, default=DEF_ANNOTATION_FILE, help="Percorso al file CSV.")
    args = parser.parse_args()

    if not os.path.exists(args.annotation_file):
        print(f"Errore: File {args.annotation_file} non trovato.")
        exit()

    # 1. Caricamento Dataset
    df = pd.read_csv(args.annotation_file)
    print(f"Caricati {len(df)} campioni da {args.annotation_file}")

    # 2. Conteggio Token
    # Usiamo word_tokenize per gestire punteggiatura e abbreviazioni correttamente
    print("Conteggio token in corso...")
    df['num_tokens'] = df['description'].apply(lambda x: len(word_tokenize(str(x))))

    # 3. Statistiche Descrittive
    print("\n--- STATISTICHE TOKEN ---")
    print(df['num_tokens'].describe())

    # 4. Categorizzazione per il Plot
    df['Token Category'] = df['num_tokens'].apply(categorize_tokens)
    
    df['num_sentences'] = df['description'].apply(lambda x: len(sent_tokenize(str(x))))

    df['Length Category'] = df['num_sentences'].apply(categorize_sentences)
    column_order = ['0-10', '11-20', '21-30', '31-40', '>40']

    # --- GENERAZIONE GRAFICO ---
    print("\nGenerazione grafici...")
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Palette sfumata per i token
    palette = sns.color_palette("magma", n_colors=len(column_order))
    
    ax = sns.countplot(
        data=df, 
        x='Token Category', 
        order=column_order,
        palette=palette
    )

    # Aggiunta etichette sopra le barre
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', 
                    xytext=(0, 5), 
                    textcoords='offset points',
                    fontsize=10, weight='bold')
    split = 'val' if 'val' in args.annotation_file else 'train' if 'train' in args.annotation_file else 'test'

    plt.title(f'Distribuzione Lunghezza Descrizioni (Token)\n File: {os.path.basename(args.annotation_file)}')
    plt.xlabel('Range Numero di Token')
    plt.ylabel('Frequenza (Numero di Annotazioni)')
    
    # Salvataggio e visualizzazione
    output_plot = f"data_preparation/plots/token_distribution_plot_{split}_{os.path.splitext(os.path.basename(args.annotation_file))[0]}.png"
    os.makedirs("data_preparation/plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"Grafico salvato in: {output_plot}")
    
    # --- SALVATAGGIO CSV ARRICCHITO (Opzionale) ---
    # Se vuoi tenere traccia dei conteggi per analisi future
    # output_csv = args.annotation_file.replace(".csv", "_with_token_counts.csv")
    # df.to_csv(output_csv, index=False)
    # print(f"Dati con conteggi salvati in: {output_csv}")

    # plt.show()