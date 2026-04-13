import os
import sys
import json

data_root = sys.argv[1]
index_path = os.path.join(data_root, "index.html")

folders = sorted([
    f for f in os.listdir(data_root)
    if os.path.isdir(os.path.join(data_root, f))
])

html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Scene Navigator</title>
    <style>
        body {{
            font-family: Arial;
            margin: 0;
            display: flex;
            height: 100vh;
        }}

        #sidebar {{
            width: 350px;
            padding: 20px;
            border-right: 1px solid #ccc;
            box-sizing: border-box;
            overflow-y: auto;
        }}

        #viewer {{
            flex: 1;
            display: flex;
            flex-direction: column;
        }}

        #scene {{
            font-size: 18px;
            margin: 20px 0;
            word-break: break-word;
            line-height: 1.5;
        }}

        #frame {{
            flex: 1;
            width: 100%;
            border: none;
        }}

        button {{
            padding: 10px 20px;
            font-size: 16px;
            margin: 5px;
        }}

        #controls {{
            margin-top: 20px;
        }}
    </style>
</head>
<body>

    <div id="sidebar">
        <h2>Scene Browser</h2>
        <div id="scene"></div>

        <div id="controls">
            <button onclick="prevScene()">← Prev</button>
            <button onclick="nextScene()">Next →</button>
            <button onclick="openScene()">Reload</button>
        </div>
    </div>

    <div id="viewer">
        <iframe id="frame"></iframe>
    </div>

    <script>
        const scenes = {json.dumps(folders)};
        let idx = 0;

        function update() {{
            const sceneName = scenes[idx];

            // Sostituisce i trattini bassi con spazi per renderlo leggibile
            document.getElementById("scene").innerText = sceneName.replaceAll('_', ' ');
            // ----------------------------

            const url = "./" + encodeURIComponent(sceneName) + "/index.html?ts=" + Date.now();

            console.log("loading:", url);

            document.getElementById("frame").src = url;
        }}
        function openScene() {{
            update();
        }}

        function nextScene() {{
            idx = (idx + 1) % scenes.length;
            update();
        }}

        function prevScene() {{
            idx = (idx - 1 + scenes.length) % scenes.length;
            update();
        }}

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

print("saved:", index_path)