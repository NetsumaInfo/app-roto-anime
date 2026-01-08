# 🎨 Roto Anime - Background Removal

> Suppression de fond pour images et vidéos avec IA — optimisé pour l'anime et les photos.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![CUDA](https://img.shields.io/badge/CUDA-12.1-green?logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Fonctionnalités

- 🖼️ **Images & Vidéos** — Traitement par lots avec extraction automatique des frames
- 🎭 **2 modèles IA** — BiRefNet (photos) et ToonOut (anime/manga)
- ⚖️ **Comparaison Before/After** — Slider interactif + navigation entre frames
- 📦 **Export ZIP** — Téléchargement de toutes les frames traitées
- 🎛️ **Paramètres ajustables** — Résolution et seuil de détection

---

## 🚀 Installation Rapide

### Prérequis

| Outil | Requis | Installation |
|-------|--------|--------------|
| **Python** | 3.10+ | [python.org](https://www.python.org/downloads/) ⚠️ Cocher "Add to PATH" |
| **ffmpeg** | Pour vidéos | `winget install ffmpeg` |
| **NVIDIA GPU** | Recommandé | [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) |

### Installation

```bash
# 1. Cloner ou télécharger le projet
git clone https://github.com/votre-repo/roto-anime.git

# 2. Lancer l'installation (télécharge automatiquement les modèles)
install.bat

# 3. Démarrer l'application
run.bat
```

L'interface s'ouvre automatiquement sur **http://localhost:7860**

---

## � Modèles

| Modèle | Optimisé pour | Source |
|--------|---------------|--------|
| **BiRefNet** | Photos, portraits, produits | [GitHub](https://github.com/ZhengPeng7/BiRefNet) |
| **ToonOut** | Anime, manga, illustrations | [HuggingFace](https://huggingface.co/joelseytre/toonout) |

> 💡 ToonOut est téléchargé automatiquement lors de l'installation (885 MB)

---

## ⚙️ Paramètres

| Paramètre | Plage | Description |
|-----------|-------|-------------|
| **Resolution** | 512 - 2048 px | Plus haut = plus de détails (mais plus lent) |
| **Threshold** | 0.1 - 0.9 | Plus bas = suppression plus agressive |

---

## � Structure

```
roto-anime/
├── app.py           # Application principale
├── install.bat      # Script d'installation
├── run.bat          # Script de lancement
├── weights/         # Poids des modèles (auto-téléchargé)
└── output/          # Résultats (images/vidéos traitées)
```

---

## 🛠️ Dépannage

| Problème | Solution |
|----------|----------|
| `Python not found` | Réinstaller Python avec "Add to PATH" coché |
| `ffmpeg not found` | Exécuter `winget install ffmpeg` puis redémarrer |
| Traitement lent | Vérifier que CUDA est installé (GPU NVIDIA requis) |
| ToonOut identique à BiRefNet | Vérifier que `weights/birefnet_finetuned_toonout.pth` existe |

---

## � Licence

MIT — Libre d'utilisation et modification.

---

<p align="center">
  <b>BiRefNet</b> par <a href="https://github.com/ZhengPeng7">ZhengPeng7</a> • 
  <b>ToonOut</b> par <a href="https://huggingface.co/joelseytre">Kartoon AI</a>
</p>
