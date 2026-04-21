import os
import sys
import json

# Prendi la root dei dati dagli argomenti
if len(sys.argv) < 2:
    print("Utilizzo: python script.py /percorso/della/cartella/output")
    sys.exit(1)

data_root = sys.argv[1]
index_path = os.path.join(data_root, "index.html")

# Prendi solo le cartelle reali, ordinandole
folders = sorted([
    f for f in os.listdir(data_root)
    if os.path.isdir(os.path.join(data_root, f))
])

# Usiamo una f-string, ma raddoppiamo le graffe per CSS e JS
html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Scene Navigator</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            display: flex;
            height: 100vh;
            background: #121212;
            color: #eee;
        }}

        #sidebar {{
            width: 380px;
            padding: 25px;
            border-right: 1px solid #333;
            box-sizing: border-box;
            overflow-y: auto;
            background: #1a1a1a;
        }}

        #viewer {{
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #000;
        }}

        #scene {{
            font-size: 16px;
            margin: 20px 0;
            word-break: break-word;
            line-height: 1.6;
            background: #252525;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #4db8ff;
        }}

        #frame {{
            flex: 1;
            width: 100%;
            border: none;
        }}

        button {{
            padding: 10px 15px;
            font-size: 14px;
            margin: 5px;
            cursor: pointer;
            background: #333;
            color: white;
            border: 1px solid #444;
            border-radius: 4px;
        }}

        button:hover {{
            background: #444;
        }}

        #controls {{
            margin-top: 20px;
            display: flex;
            justify-content: center;
        }}
    </style>
</head>
<body>

    <div id="sidebar">
        <h2 style="color: #4db8ff; margin-top: 0;">Scene Browser</h2>
        <div id="scene">Caricamento...</div>

        <div id="controls">
            <button onclick="prevScene()">← Prev</button>
            <button onclick="nextScene()">Next →</button>
            <button onclick="openScene()">Reload</button>
        </div>
        <p style="font-size: 11px; color: #666; margin-top: 20px;">Use Left/Right arrows to navigate</p>
    </div>

    <div id="viewer">
        <iframe id="frame"></iframe>
    </div>

    <script>
        const scenes = {json.dumps(folders)};
        let idx = 0;

        async function update() {{
            const sceneName = scenes[idx];
            const scenePath = "./" + encodeURIComponent(sceneName);
            const sceneContainer = document.getElementById("scene");
            
            try {{
                const response = await fetch(scenePath + "/metadata.json");
                if (response.ok) {{
                    const data = await response.json();
                    sceneContainer.innerHTML = `
                        <div style="font-size: 12px; color: #888; margin-bottom: 8px;">ID: ${{sceneName.split('_').slice(0,2).join('_')}}</div>
                        <div style="margin-bottom: 10px;">${{data.caption}}</div>
                        <div style="font-size: 13px; color: #4db8ff;">Target: <b>${{data.target}}</b></div>
                    `;
                }} else {{
                    sceneContainer.innerText = sceneName.replaceAll('_', ' ');
                }}
            }} catch (e) {{
                sceneContainer.innerText = sceneName.replaceAll('_', ' ');
            }}

            const url = scenePath + "/index.html?ts=" + Date.now();
            document.getElementById("frame").src = url;
        }}

        function openScene() {{ update(); }}
        function nextScene() {{ idx = (idx + 1) % scenes.length; update(); }}
        function prevScene() {{ idx = (idx - 1 + scenes.length) % scenes.length; update(); }}

        document.addEventListener("keydown", e => {{
            if (e.key === "ArrowRight") nextScene();
            if (e.key === "ArrowLeft") prevScene();
        }});

        update();
    </script>

</body>
</html>
"""

with open(index_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f" Successo! Visualizzatore salvato in: {{index_path}}")