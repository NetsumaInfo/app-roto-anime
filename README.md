# BiRefNet Background Removal

Suppression de fonds d'images et vidéos par IA.

---

## ⚠️ Prérequis (PC nu)

### 1. Python 3.10+
Télécharger et installer : **[python.org/downloads](https://www.python.org/downloads/)**

> ⚡ **Important** : Cocher **"Add Python to PATH"** pendant l'installation !

### 2. ffmpeg (pour les vidéos)
Télécharger : **[ffmpeg.org/download](https://ffmpeg.org/download.html)**

Ou avec winget :
```bash
winget install ffmpeg
```

### 3. GPU NVIDIA (recommandé)
Installer les drivers CUDA : **[developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)**

---

## 🚀 Installation

1. **Télécharger** ce projet (Code → Download ZIP)
2. **Extraire** le ZIP
3. **Double-clic sur `install.bat`**
4. **Double-clic sur `run.bat`**

Le navigateur s'ouvre automatiquement sur **http://localhost:7860**

---

## 🎨 Modèles

| Modèle | Optimisé pour | Précision |
|--------|---------------|-----------|
| **BiRefNet** | Photos, portraits | SOTA |
| **ToonOut** | Anime, manga | 99.5% |

## ⚙️ Paramètres

| Paramètre | Plage | Description |
|-----------|-------|-------------|
| Resolution | 512-2048px | Plus haut = plus de détails, plus lent |
| Threshold | 0.1-0.9 | Plus bas = suppression plus agressive |

---

## 📦 ToonOut (optionnel)

Pour le modèle anime, télécharger les poids depuis [HuggingFace](https://huggingface.co/joelseytre/toonout) → `weights/birefnet_finetuned_toonout.pth`

## 🔗 Liens

- [BiRefNet GitHub](https://github.com/ZhengPeng7/BiRefNet)
- [ToonOut Weights](https://huggingface.co/joelseytre/toonout)

## 📄 License

MIT
