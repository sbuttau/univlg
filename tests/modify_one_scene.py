import argparse
import os
import pandas as pd
import ast
import re
DEF_ANNOTATION_FILE = "data/refer_it_3d/ScanRefer_filtered_val_ScanEnts3D_val_negations_only.csv"

def manual_tokenization(text):
    """Semplice tokenizzazione per allineare la lista tokens alla descrizione."""
    # Rimuove la punteggiatura comune per evitare mismatch
    # for char in [".", ",", "!", "?", ";", ":"]:
    #     text = text.replace(char, "")
    text = re.sub(r'([\.!,\?;:])', r' \1 ', text)
    return text.lower().split()


import ast

def filter_tokens_and_entities(entry, keyword, new_description=None):
    """
    Rimuove la keyword sia dalla lista dei tokens che dalla struttura entities.
    Aggiorna anche la descrizione per riflettere la rimozione.
    """
    # 1. Recupero dati originali
    try:
        tokens_list = ast.literal_eval(entry.iloc[0]['tokens']) if isinstance(entry.iloc[0]['tokens'], str) else entry.iloc[0]['tokens']
        entities_list = ast.literal_eval(entry.iloc[0]['entities']) if isinstance(entry.iloc[0]['entities'], str) else entry.iloc[0]['entities']
    except (ValueError, SyntaxError):
        print("Errore nel parsing dei dati.")
        return entry

    # 2. Filtro Entities (basato sulla funzione precedente)
    #  [[[4], ['6_table']], [[7, 8], ['12_monitor', '33_monitor', '32_monitor']], [[10], ['6_table']], [[12], ['6_table']], [[18], ['13_chair']], [[20], ['6_table']]]
    # entities_list = [[[4], ['6_table']], [[8], ['12_monitor']], [[10], ['6_table']], [[12], ['6_table']], [[18], ['13_chair']], [[20], ['6_table']]] # hard coded for testing
    # entities_list = [[[4], ['6_table']], [[8], ['12_monitor']], [[10], ['6_table']], [[12], ['6_table']], [[18], ['13_chair']], [[20], ['6_table']]] # hard coded for testing

    # scene 355: [[[1], ['6_chair']], [[9], ['6_chair']], [[14], ['15_table']], [[23], ['3_chair']]]
    entities_list = [[[2], ['3_chair']]]
    import pdb; pdb.set_trace()
    # 3. Filtro Tokens
    # Rimuoviamo il token esatto se presente (es. 'table')
    new_tokens = manual_tokenization(new_description) if new_description else [t for t in tokens_list if t.lower() != keyword.lower()]

    # 4. Ricostruzione Descrizione (opzionale ma consigliata per coerenza)
    # Crea una frase dai nuovi token
    new_desc = " ".join(new_tokens) if new_description is None else new_description

    # 5. Aggiornamento Entry
    idx = entry.index[0]
    entry.at[idx, 'tokens'] = str(new_tokens)
    entry.at[idx, 'token'] = str(new_tokens)
    entry.at[idx, 'entities'] = str(entities_list)
    entry.at[idx, 'description'] = new_desc
    # entry.at[idx, 'object_name'] = 'chair'
    entry.at[idx, 'object_id'] = 3 
    entry.at[idx, 'target_id'] = 3


    print(f"--- Pulizia completata per: {keyword} ---")
    print(f"Nuova Descrizione: {new_desc}")
    return entry

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analisi e modifica dei campioni nel dataset.")
    parser.add_argument('--annotation_file', type=str, default=DEF_ANNOTATION_FILE, help="Percorso al file CSV.")
    parser.add_argument('--scene_id', type=str, default='scene_0329_00', help="ID della scena da estrarre.")
    args = parser.parse_args()

    if not os.path.exists(args.annotation_file):
        print(f"Errore: File {args.annotation_file} non trovato.")
        exit()

    # 1. Caricamento Dataset
    df = pd.read_csv(args.annotation_file)
    print(f"Caricati {len(df)} campioni da {args.annotation_file}")

    # 2. Filtra per la scena
    try:
        scene_samples = df[df['scan_id'] == args.scene_id]
        print(f"Trovati {len(scene_samples)} campioni per la scena {args.scene_id}")
    except KeyError:
        print("Errore: La colonna 'scan_id' non è presente nel dataset.")
        exit()

    if scene_samples.empty:
        print(f"Nessun campione trovato per la scena {args.scene_id}.")
        exit()

    # 3. Selezione del campione
    selected_entry = None
    for idx in range(len(scene_samples)):
        current_row = scene_samples.iloc[idx]
        print(f"\n--- Campione [{idx}] ---")
        print(f"Descrizione: {current_row['description']}")
        print(f"Object Name: {current_row['object_name']}")
        
        confirm = input("È questo il campione che vuoi modificare/estrarre? (s/n): ")
        if confirm.lower() == 's':
            selected_entry = scene_samples.iloc[[idx]].copy()
            break
    
    if selected_entry is None:
        print("Operazione annullata.")
        exit()
    # 4. Modifica della descrizione e sincronizzazione
    print("\nVuoi modificare la descrizione per il test di Attention Sink? (s/n): ")
    if input().lower() == 's':
        old_desc = selected_entry.iloc[0]['description']
        print(f"Descrizione attuale: {old_desc}")
        
        # new_desc = input("Inserisci la nuova descrizione: ")
        # ask the token want to remove
        keyword = input(f"Quale entity vuoi eliminare completamente? {selected_entry.iloc[0]['entities']}   ")
        descr= input("Scrivi la nuova descrizione    ")
        selected_entry = filter_tokens_and_entities(selected_entry, keyword, descr)  

        # Aggiornamento 'entities'
        print(f"\nNuove entities: {selected_entry.iloc[0]['entities']}")
        print(f"Tokens: {selected_entry.iloc[0]['tokens']}")
        

    # 5. Salva
    # 
    output_path = f"data/refer_it_3d/{args.scene_id}_monitor.csv"
    selected_entry.to_csv(output_path, index=False)
    print(f"\n--- Salvataggio completato ---")
    print(f"Campione salvato in: {output_path}")
    print(f"Nuova descrizione: {selected_entry.iloc[0]['description']}")
    print(f"Nuovi tokens: {selected_entry.iloc[0]['tokens']}")