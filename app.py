"""
BiRefNet & ToonOut - Background Removal
Clean interface with Compare and Gallery tabs
"""

import torch
from PIL import Image
from torchvision import transforms
import gradio as gr
import os
import numpy as np
import subprocess
import zipfile
from pathlib import Path
from datetime import datetime

# Fix for BiRefNet compatibility
import transformers.configuration_utils
original_getattribute = transformers.configuration_utils.PretrainedConfig.__getattribute__

def patched_getattribute(self, key):
    if key == 'is_encoder_decoder':
        return False
    return original_getattribute(self, key)

transformers.configuration_utils.PretrainedConfig.__getattribute__ = patched_getattribute

from transformers import AutoModelForImageSegmentation

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.webm', '.mkv', '.flv'}

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
TOONOUT_WEIGHTS = BASE_DIR / "weights" / "birefnet_finetuned_toonout.pth"

models = {"birefnet": None, "toonout": None}

def load_birefnet():
    global models
    if models["birefnet"] is None:
        print("⏳ Loading BiRefNet...")
        models["birefnet"] = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True
        ).to(DEVICE).eval()
        print("✅ BiRefNet ready")
    return models["birefnet"]

def load_toonout():
    global models
    if models["toonout"] is None:
        print("⏳ Loading ToonOut...")
        model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True
        )
        if TOONOUT_WEIGHTS.exists():
            state_dict = torch.load(str(TOONOUT_WEIGHTS), map_location='cpu', weights_only=True)
            clean = {}
            for k, v in state_dict.items():
                new_k = k.replace("module._orig_mod.", "").replace("module.", "")
                new_k = new_k.replace("squeeze_0.", "squeeze_module.0.")
                clean[new_k] = v
            model.load_state_dict(clean, strict=False)
        models["toonout"] = model.to(DEVICE).eval()
        print("✅ ToonOut ready")
    return models["toonout"]

def get_model(name): 
    return load_toonout() if "ToonOut" in name else load_birefnet()

def create_checkerboard(size, square_size=16):
    w, h = size
    img = Image.new('RGB', (w, h))
    pixels = img.load()
    for i in range(w):
        for j in range(h):
            if (i // square_size + j // square_size) % 2 == 0:
                pixels[i, j] = (180, 180, 180)
            else:
                pixels[i, j] = (140, 140, 140)
    return img

def composite_on_checkerboard(rgba_img):
    if rgba_img.mode != 'RGBA':
        return rgba_img
    checker = create_checkerboard(rgba_img.size)
    checker.paste(rgba_img, mask=rgba_img.split()[3])
    return checker

def process_image(img: Image.Image, model_name: str, res: int, thresh: float):
    model = get_model(model_name)
    orig_size = img.size
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        pred = model(transform(img).unsqueeze(0).to(DEVICE))[-1].sigmoid().cpu()
    
    mask = transforms.ToPILImage()((pred[0].squeeze() > thresh).float())
    mask = mask.resize(orig_size, Image.Resampling.LANCZOS)
    
    result = img.copy()
    result.putalpha(mask)
    
    return result, img

def extract_frames(video: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-i", video,
        "-pix_fmt", "rgba",           # Keep alpha channel if present
        "-compression_level", "0",     # Lossless PNG (fastest, best quality)
        os.path.join(out_dir, "frame_%05d.png"), "-y"
    ], capture_output=True)
    return sorted([os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith('.png')])

def detect_type(path: str):
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXTENSIONS: return "image"
    if ext in VIDEO_EXTENSIONS: return "video"
    return None

def preview_media(file):
    if not file:
        return gr.update(visible=False), gr.update(visible=False)
    path = file.name if hasattr(file, 'name') else file
    ftype = detect_type(path)
    if ftype == "image":
        return gr.update(value=path, visible=True), gr.update(visible=False)
    elif ftype == "video":
        return gr.update(visible=False), gr.update(value=path, visible=True)
    return gr.update(visible=False), gr.update(visible=False)

# Processing state
processing_state = {"running": False, "paused": False, "stop": False}

def stop_processing():
    """Stop and reset to initial state"""
    processing_state["stop"] = True
    processing_state["running"] = False
    processing_state["paused"] = False
    # Show Run, hide Pause/Stop
    return (
        gr.update(visible=True),   # run_btn
        gr.update(visible=False),  # pause_btn
        gr.update(visible=False),  # stop_btn
        "⏹️ Stopped"               # status
    )

def pause_processing():
    """Toggle pause/resume"""
    processing_state["paused"] = not processing_state["paused"]
    if processing_state["paused"]:
        return gr.update(value="▶️ Resume"), "⏸️ Paused"
    else:
        return gr.update(value="⏸️ Pause"), "▶️ Resumed"

def start_processing():
    """Called when Run is clicked - show control buttons"""
    return (
        gr.update(visible=False),  # run_btn
        gr.update(visible=True, value="⏸️ Pause"),  # pause_btn
        gr.update(visible=True),   # stop_btn
    )

def process(file, model, res, thresh, progress=gr.Progress()):
    global processing_state
    processing_state = {"running": True, "paused": False, "stop": False}
    
    if not file:
        return None, None, None, "", ""
    
    path = file.name if hasattr(file, 'name') else file
    ftype = detect_type(path)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = OUTPUT_DIR / ts
    session.mkdir(exist_ok=True)
    
    if ftype == "image":
        progress(0.2, desc="Processing...")
        img = Image.open(path)
        result, orig = process_image(img, model, res, thresh)
        result.save(session / "result.png")
        
        result_checker = composite_on_checkerboard(result)
        slider_data = (np.array(orig), np.array(result_checker))
        
        processing_state["running"] = False
        progress(1.0)
        return slider_data, [np.array(result)], None, "✅ Done", str(session)
    
    elif ftype == "video":
        frames_in = session / "input"
        frames_out = session / "output"
        frames_in.mkdir(); frames_out.mkdir()
        
        progress(0.05, desc="Extracting frames...")
        frames = extract_frames(path, str(frames_in))
        total = len(frames)
        if not total:
            return None, None, None, "❌ No frames", ""
        
        results = []
        first_slider = None
        
        for i, fp in enumerate(frames):
            # Check for stop
            if processing_state["stop"]:
                processing_state["running"] = False
                return first_slider, results, None, f"⏹️ Stopped at {i}/{total}", str(frames_out)
            
            # Check for pause
            import time
            while processing_state["paused"]:
                time.sleep(0.1)
                if processing_state["stop"]:
                    break
            
            progress((i+1)/total, desc=f"Frame {i+1}/{total}")
            img = Image.open(fp)
            result, orig = process_image(img, model, res, thresh)
            result.save(frames_out / f"frame_{i:05d}.png")
            results.append(np.array(result))
            
            if i == 0:
                result_checker = composite_on_checkerboard(result)
                first_slider = (np.array(orig), np.array(result_checker))
        
        progress(0.98, desc="Creating ZIP...")
        zip_path = session / "frames.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(frames_out.iterdir()):
                zf.write(f, f.name)
        
        processing_state["running"] = False
        progress(1.0)
        return first_slider, results, str(zip_path), f"✅ {total} frames", str(frames_out)

def create_app():
    with gr.Blocks(title="Background Removal") as app:
        
        gr.Markdown("# 🎨 Background Removal")
        gr.Markdown("*BiRefNet for photos • ToonOut for anime*")
        
        with gr.Row():
            # Left Panel
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Input")
                file_input = gr.File(label=None, file_types=["image", "video"], type="filepath")
                
                gr.Markdown("### 👁️ Preview")
                img_preview = gr.Image(label=None, height=150, visible=False, show_label=False)
                video_preview = gr.Video(label=None, height=150, visible=False, show_label=False)
                
                gr.Markdown("### ⚙️ Settings")
                model_select = gr.Radio(
                    ["BiRefNet (Photos)", "ToonOut (Anime)"],
                    value="BiRefNet (Photos)",
                    label="Model"
                )
                res_slider = gr.Slider(512, 2048, 1024, step=256, label="Resolution")
                thresh_slider = gr.Slider(0.1, 0.9, 0.5, step=0.05, label="Threshold")
                
                with gr.Row():
                    process_btn = gr.Button("▶️ Run", variant="primary", size="lg")
                    pause_btn = gr.Button("⏸️ Pause", size="lg")
                    stop_btn = gr.Button("⏹️ Stop", variant="stop", size="lg")
                
                status = gr.Textbox(label="Status", interactive=False)
                output_folder = gr.Textbox(label="📂 Output", interactive=False)
            
            # Right Panel
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("⚖️ Compare"):
                        comparison_slider = gr.ImageSlider(label="Before / After", type="numpy")
                    
                    with gr.TabItem("🖼️ Gallery"):
                        gallery = gr.Gallery(label="Results", columns=4, height=400, object_fit="contain")
                
                download_zip = gr.File(label="📦 Download ZIP")
        
        with gr.Accordion("ℹ️ Info", open=False):
            gr.Markdown(f"""
**Models**
| Model | Optimized For | Accuracy |
|-------|---------------|----------|
| BiRefNet | Photos, portraits, products | SOTA |
| ToonOut | Anime, manga, illustrations | 99.5% |

**Parameters**
- **Resolution**: Higher = finer details, slower (512-2048px)
- **Threshold**: Lower = more aggressive removal (0.1-0.9)

**Links**
- [BiRefNet GitHub](https://github.com/ZhengPeng7/BiRefNet)
- [ToonOut Weights](https://huggingface.co/joelseytre/toonout)

GPU: **{'✅ CUDA' if DEVICE == 'cuda' else '❌ CPU'}**
""")
        
        file_input.change(preview_media, [file_input], [img_preview, video_preview])
        
        process_btn.click(
            process,
            [file_input, model_select, res_slider, thresh_slider],
            [comparison_slider, gallery, download_zip, status, output_folder]
        )
        
        stop_btn.click(stop_processing, outputs=[status])
        pause_btn.click(pause_processing, outputs=[status])
    
    return app

if __name__ == "__main__":
    print(f"🚀 Starting on {DEVICE}")
    print(f"📂 Output: {OUTPUT_DIR}")
    load_birefnet()
    
    app = create_app()
    
    # Try ports 7860-7869
    import socket
    for port in range(7860, 7870):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
            print(f"🌐 Using port {port}")
            app.launch(server_name="0.0.0.0", server_port=port, inbrowser=True)
            break
        except OSError:
            print(f"⚠️ Port {port} busy, trying next...")
            continue
