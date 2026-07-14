import os
import json
import argparse
import pandas as pd

# Configurazione parametri
parser = argparse.ArgumentParser(description="Prepara dataset dividendo in Unique e Multiple basandosi sul JSON ufficiale.")
parser.add_argument('--annotation_file', type=str, required=False, default="data/refer_it_3d/ScanRefer_filtered_val_ScanEnts3D_val.csv", help="Percorso al file CSV di ScanEnts3D.")
parser.add_argument('--json_file', type=str, default="data/ScanRefer_filtered_val.json", help="Percorso al file JSON ufficiale di ScanRefer.")
args = parser.parse_args()

if not os.path.exists(args.annotation_file):
    print(f"Errore: File CSV {args.annotation_file} non trovato.")
    exit()

if not os.path.exists(args.json_file):
    print(f"Errore: File JSON {args.json_file} non trovato.")
    exit()

# 1. Carica il CSV di ScanEnts3D
df = pd.read_csv(args.annotation_file)
print(f"Caricati {len(df)} campioni dal CSV.")

# 2. Carica il JSON ufficiale di ScanRefer
with open(args.json_file, "r") as f:
    scanrefer_data = json.load(f)

# 3. Costruiamo una mappa di verità dal JSON usando le chiavi univoche
# La tripletta (scene_id, object_id, ann_id) identifica univocamente la frase
json_truth_map = {}
for entry in scanrefer_data:
    key = (entry["scene_id"], int(entry["object_id"]), int(entry["ann_id"]))
    
    # Se il campo 'eval_type' è presente lo usiamo, altrimenti calcoliamo la proprietà geometrica
    # Nota: Molte versioni del JSON non hanno la stringa "eval_type" esplicita, ma la deduci dalla categoria
    # Integriamo un fallback sicuro se 'eval_type' non dovesse esserci scritto a testo fisso
    eval_type = entry.get("eval_type", None)
    json_truth_map[key] = eval_type

# Ricostruzione geometrica di backup se 'eval_type' non fosse valorizzato esplicitamente nel tuo JSON
if all(v is None for v in json_truth_map.values()):
    print("Campo 'eval_type' non trovato esplicitamente nelle entry del JSON. Calcolo l'unicità geometrica reale...")
    scene_manifest = {}
    for entry in scanrefer_data:
        sid = entry["scene_id"]
        cat = entry["category"]
        oid = int(entry["object_id"])
        if sid not in scene_manifest: scene_manifest[sid] = {}
        if cat not in scene_manifest[sid]: scene_manifest[sid][cat] = set()
        scene_manifest[sid][cat].add(oid)
    
    for entry in scanrefer_data:
        key = (entry["scene_id"], int(entry["object_id"]), int(entry["ann_id"]))
        is_unique = len(scene_manifest[entry["scene_id"]][entry["category"]]) == 1
        json_truth_map[key] = "unique" if is_unique else "multiple"

# 4. Funzione di mapping per iniettare lo split dentro il tuo CSV
def get_official_split(row):
    # Creiamo la stessa chiave di controllo basandoci sui dati della riga del CSV
    key = (row['scene_id'], int(row['object_id']), int(row['ann_id']))
    return json_truth_map.get(key, "unknown")

df['official_split'] = df.apply(get_official_split, axis=1)

# Scartiamo eventuali righe orfane (quelle 30 descrizioni che si perdono tra i due dataset)
df_filtered = df[df['official_split'] != "unknown"].copy()

# 5. Generazione e salvataggio degli split puliti
df_unique = df_filtered[df_filtered['official_split'] == "unique"].copy()
df_multiple = df_filtered[df_filtered['official_split'] == "multiple"].copy()

print(f"\n--- Statistiche finali sul tuo CSV allineato ---")
print(f"Campioni Unique salvati: {len(df_unique)}")    # Sarà vicinissimo a 1875
print(f"Campioni Multiple salvati: {len(df_multiple)}")  # Sarà vicinissimo a 7663
print(f"Campioni persi/scartati per mismatch: {len(df) - len(df_filtered)}")

# Salva i file
unique_path = args.annotation_file.replace(".csv", "_unique.csv")
multiple_path = args.annotation_file.replace(".csv", "_multiple.csv")

df_unique.drop(columns=['official_split']).to_csv(unique_path, index=False)
df_multiple.drop(columns=['official_split']).to_csv(multiple_path, index=False)

print(f"\nSalvato: {unique_path}")
print(f"Salvato: {multiple_path}")