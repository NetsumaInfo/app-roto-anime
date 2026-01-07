# BiRefNet Background Removal

Remove backgrounds from images and videos using AI.

## Quick Start

```bash
# Windows
run.bat

# Manual
pip install -r requirements.txt
pip install gradio
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
