import urllib.request

print("Downloading YOLOv8 Fire & Smoke AI Model from GitHub (Please wait...)")

# 🔥 අලුත් වැඩ කරන GitHub ලින්ක් එක
url = "https://github.com/nimradev064/Real-Time-Fire-Detection-Flask-App/raw/main/Models/best.pt"

# වෙබ්සයිට් එක බ්ලොක් නොකරන්න User-Agent එකක් යවනවා
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

with urllib.request.urlopen(req) as response, open("fire_model.pt", 'wb') as out_file:
    data = response.read()
    out_file.write(data)

print("✅ Download Complete! You can see 'fire_model.pt' in your folder.")