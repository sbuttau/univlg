import os
import json
import argparse
import numpy as np
import open3d as o3d
import pandas as pd
import ast

# --- CONFIGURAZIONE ---
ROOT_PATH = "/workspaces/univlg/data/ScanNetv2"
DEF_ANNOTATION_FILE = "data/refer_it_3d/ScanRefer_filtered_val_ScanEnts3D_val.csv" 

class DatasetVisualizer:
    def __init__(self, samples):
        self.samples = samples
        self.current_idx = 0
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        
    def load_and_update(self, vis):
        """Carica il campione corrente e aggiorna la scena."""
        if self.current_idx >= len(self.samples):
            print("\nFine dei campioni disponibili!")
            # Invece di chiudere, avvisiamo l'utente
            return

        sample = self.samples.iloc[self.current_idx].to_dict()
        scene_id = sample['scene_id']
        target_id = sample['object_id']
        utterance = sample.get('utterance', sample.get('description', 'N/A'))
        
        # Parsing entities
        raw_entities = sample.get('entities', "[]")
        if isinstance(raw_entities, str):
            try: raw_entities = ast.literal_eval(raw_entities)
            except: raw_entities = []

        print(f"\n[{self.current_idx+1}/{len(self.samples)}] SCENA: {scene_id}")
        print(f"UTTERANCE: {utterance}")
        print("COMANDI: [N] Prossimo | [Q] Esci")

        # Caricamento Mesh
        new_mesh, _ = self.load_scene(scene_id)
        if new_mesh is None:
            print(f"Mesh non trovata per {scene_id}, salto al prossimo...")
            self.current_idx += 1
            self.load_and_update(vis)
            return

        # Applicazione maschera
        new_mesh = self.apply_mask(new_mesh, scene_id, target_id, raw_entities)

        # Aggiornamento geometria
        vis.clear_geometries()
        vis.add_geometry(new_mesh)
        vis.reset_view_point(True)
        
    def next_sample(self, vis):
        if self.current_idx < len(self.samples) - 1:
            self.current_idx += 1
            self.load_and_update(vis)
        else:
            print("Sei già all'ultimo campione.")

    def load_scene(self, scene_id):
        ply_path = os.path.join(ROOT_PATH, "scans", scene_id, f"{scene_id}_vh_clean_2.ply")
        txt_path = os.path.join(ROOT_PATH, "scans", scene_id, f"{scene_id}.txt")
        if not os.path.exists(ply_path): return None, None
        mesh = o3d.io.read_triangle_mesh(ply_path)
        mesh.compute_vertex_normals()
        
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    if 'axisAlignment' in line:
                        matrix = np.array([float(x) for x in line.split('=')[1].strip().split(' ')]).reshape(4,4)
                        mesh.transform(matrix)
                        break
        return mesh, None

    def apply_mask(self, mesh, scene_id, target_id, entities):
        segs_path = os.path.join(ROOT_PATH, "scans", scene_id, f"{scene_id}_vh_clean_2.0.010000.segs.json")
        if not os.path.exists(segs_path): return mesh
        with open(segs_path, 'r') as f: segs_data = json.load(f)
        
        vertex_to_seg = np.array(segs_data['segIndices'])
        colors = np.asarray(mesh.vertex_colors).copy()
        # colors *= 0.3 
        target_id_clean = str(target_id).split('_')[0]

        for ent in entities:
            for obj in ent:
                try:
                    curr_id = str(obj[0]).split('_')[0]
                    segments = self.get_segments(scene_id, curr_id)
                    mask = np.isin(vertex_to_seg, segments)
                    colors[mask] = [1, 0, 0] if curr_id == target_id_clean else [0, 0.5, 1]
                except: continue
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        return mesh

    def get_segments(self, scene_id, obj_id):
        agg_path = os.path.join(ROOT_PATH, "scans", scene_id, f"{scene_id}.aggregation.json")
        if not os.path.exists(agg_path): return []
        with open(agg_path, 'r') as f: data = json.load(f)
        for group in data['segGroups']:
            if str(group['objectId']) == str(obj_id): return group['segments']
        return []

    def run(self):
        self.vis.create_window(window_name="ScanEnts3D Browser", width=1200, height=800)
        self.vis.register_key_callback(ord("N"), self.next_sample)
        self.load_and_update(self.vis)
        self.vis.run()
        self.vis.destroy_window()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default=DEF_ANNOTATION_FILE)
    parser.add_argument('--scene_id', type=str, default=None, help="ID della scena specifica da visualizzare (es. scene0011_00)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    # --- LOGICA DI FILTRAGGIO ---
    if args.scene_id:
        filtered_df = df[df['scan_id'] == args.scene_id].copy()
        if filtered_df.empty:
            print(f"Errore: Nessuna annotazione trovata per la scena {args.scene_id}")
            exit()
        print(f"Filtrate {len(filtered_df)} annotazioni per la scena {args.scene_id}")
        df = filtered_df
    # ----------------------------

    visualizer = DatasetVisualizer(df)
    visualizer.run()