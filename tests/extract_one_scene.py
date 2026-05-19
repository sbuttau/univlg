'''
This script extracts one scene from the dataset and saves it as a .csv file. It is used for debugging and visualization purposes.'''


import argparse
import os
import pandas as pd

DEF_ANNOTATION_FILE = "data/refer_it_3d/ScanRefer_filtered_val_ScanEnts3D_val_negations_only.csv"
# So far I'm importing data/refer_it_3d/ScanRefer_filtered_val_ScanEnts3D_val_negations_only.csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analisi della distribuzione dei token nel dataset.")
    parser.add_argument('--annotation_file', type=str, default=DEF_ANNOTATION_FILE, help="Percorso al file CSV.")
    parser.add_argument('--scene_id', type=str, default='scene_0329_00', help="ID della scena da estrarre.")
    args = parser.parse_args()

    if not os.path.exists(args.annotation_file):
        print(f"Errore: File {args.annotation_file} non trovato.")
        exit()

    # 1. Caricamento Dataset
    df = pd.read_csv(args.annotation_file)
    print(f"Caricati {len(df)} campioni da {args.annotation_file}")

    # 1. Filtra per la scena
    try:
        scene_samples = df[df['scan_id'] == args.scene_id]
        print(f"Trovati {len(scene_samples)} campioni per la scena {args.scene_id}")
    except KeyError:
        print("Errore: La colonna 'scan_id' non è presente nel dataset.")
        exit()

    # 2. Prendi la seconda riga (indice 1)
    # Uso [[1:2]] per mantenere il formato DataFrame ed evitare errori se la riga non esiste
    for idx in range(len(scene_samples)):
        print(f"Campione estratto {idx}:\n{scene_samples.iloc[idx]['description']}\n")
        # Chiedi conferma all'utente che sia la scena giusta
        confirm = input("È la scena giusta? (s/n): ")
        if confirm.lower() == 'n':
            if idx == len(scene_samples) - 1:
                print("Non ci sono più campioni da questa scena. Operazione annullata.")
                exit()
            else:
                continue
        else:
            break
    entry = scene_samples.iloc[[idx]]
    print("Vuoi modificare la descrizione? (s/n): ")
    if input().lower() == 's':
        print("Inserisci la nuova descrizione:")
        new_desc = input()
        entry.at[entry.index[0], 'description'] = new_desc
        print("Descrizione aggiornata.")
        
    # 3. Salva
    output_path = f"data/refer_it_3d/{args.scene_id}_scanrefer_val.csv"
    entry.to_csv(output_path, index=False)
    print(f"Campione salvato in {output_path}")