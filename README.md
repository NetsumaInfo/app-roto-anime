# BiRefNet Background Removal

Remove backgrounds from images and videos using AI.

## Quick Start

### Windows (Recommended)
1.  **Double-cliquez sur `install.bat`** : Crée l'environnement virtuel et installe les dépendances.
2.  **Double-cliquez sur `run.bat`** : Lance l'application et ouvre le navigateur.

### Manuel (Ligne de commande)
```bash
# 1. Installation
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
pip install gradio

# 2. Lancement
python app.py
```

Browser opens automatically at **http://localhost:7860**

---

## Models

| Model | Optimized For | Accuracy |
|-------|---------------|----------|
| **BiRefNet** | Photos, portraits, products | State-of-the-art |
| **ToonOut** | Anime, manga, illustrations | 99.5% |

## Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| Resolution | 512-2048px | Higher = finer details, slower processing |
| Threshold | 0.1-0.9 | Lower = more aggressive background removal |

## Features

- Process images & videos
- Real-time before/after comparison
- Batch frame extraction
- ZIP download for video frames
- Run/Stop/Pause controls

---

## ToonOut Weights

Download from [HuggingFace](https://huggingface.co/joelseytre/toonout) → `weights/birefnet_finetuned_toonout.pth`

## Requirements

- Python 3.10+
- CUDA GPU (recommended)
- ffmpeg (for video)

## Links

- [BiRefNet GitHub](https://github.com/ZhengPeng7/BiRefNet)
- [ToonOut GitHub](https://github.com/MatteoKartoon/BiRefNet)
- [ToonOut Weights](https://huggingface.co/joelseytre/toonout)

## License

MIT
