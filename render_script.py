# ========== FULL PIPELINE 1 GPU T4: Sheet Image URLs → LTX Image-To-Video (No Drive) → TTS → Merge ==========
import os
import sys
import json
import time
import gc
import requests
import subprocess
import pandas as pd
import glob
import shutil

# ==========================================
# CÀI ĐẶT DEPENDENCY
# ==========================================
def install_all_deps():
    print("📦 Đang kiểm tra và cài đặt dependency...")

    try:
        import ftfy
        print("✅ ftfy đã có")
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "ftfy"], check=False)
        import ftfy

    try:
        from diffusers import LTXImageToVideoPipeline
        print("✅ diffusers + LTXImageToVideoPipeline đã có")
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q",
            "diffusers>=0.32.0", "transformers>=4.46.0", "accelerate>=1.1.0",
            "sentencepiece", "imageio", "imageio-ffmpeg", "edge-tts",
            "safetensors", "huggingface_hub", "protobuf<6", "opencv-python-headless", "ftfy", "kaggle"
        ], check=False)

    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        subprocess.run("apt-get update -qq && apt-get install -y -qq ffmpeg", shell=True, check=False)

    print("✅ Dependency sẵn sàng\n")

install_all_deps()

import torch
import ftfy

# ==========================================
# CẤU HÌNH PIPELINE
# ==========================================
TOTAL_SCENES = 100
OUTPUT_DIR = "./output_scenes"
VIDEO_DIR = "./output_videos"
FINAL_VIDEO = "final_full_movie.mp4"
LOCAL_LTX_DIR = "/kaggle/working/ltx-video-model"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(LOCAL_LTX_DIR, exist_ok=True)

SHEET_ID = os.environ.get("SHEET_ID", "1DmA-yuPwDl1riceSMGzWPXhuxrL4y987lOZ6Af351l8")
GID_SCENES = os.environ.get("GID_SCENES", "0")

DEFAULT_MODEL_ID = "Lightricks/LTX-Video"
LTX_WIDTH = 512
LTX_HEIGHT = 512
LTX_NUM_FRAMES = 65
LTX_STEPS = 25
LTX_FPS = 24
FINAL_FPS = 24
FINAL_WIDTH = 768
FINAL_HEIGHT = 512

# Bộ Negative Prompt khóa cố định nhân vật, chống méo mó hình thể
NEGATIVE_PROMPT_LOCK = (
    "worst quality, low quality, inconsistent motion, blurry, jittery, distorted face, "
    "deformed eyes, morphed limbs, extra limbs, bad anatomy, warped features, "
    "changing face shape, floating body parts, duplicate heads, unstable character design"
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:32"

# ==========================================
# TỰ ĐỘNG TẢI DATASET LTX
# ==========================================
def download_ltx_dataset():
    print("🔍 Kiểm tra dataset LTX local...")

    if os.path.exists("/kaggle/input"):
        for d in os.listdir("/kaggle/input"):
            full = os.path.join("/kaggle/input", d)
            if "ltx" in d.lower() and os.path.isdir(full):
                files = os.listdir(full)
                if any(f.endswith(('.safetensors', '.bin', '.json')) for f in files):
                    print(f"✅ Đã có dataset trong Input: {full}")
                    return full

    if os.path.exists(LOCAL_LTX_DIR) and len(os.listdir(LOCAL_LTX_DIR)) > 3:
        print(f"✅ Đã có model local tại: {LOCAL_LTX_DIR}")
        return LOCAL_LTX_DIR

    print("📥 Đang tải dataset LTX bằng Kaggle API...")
    datasets_to_try = [
        "atusatus/ltx-video",
        "alihadi/ltx-video",
        "alihadi/ltx-video-gguf",
    ]

    kaggle_json = "/root/.kaggle/kaggle.json"
    if not os.path.exists(kaggle_json):
        print("⚠️ Chưa có kaggle.json → dùng Hugging Face online")
        return DEFAULT_MODEL_ID

    for ds in datasets_to_try:
        try:
            print(f"   Thử tải: {ds}")
            cmd = f'kaggle datasets download -d {ds} -p {LOCAL_LTX_DIR} --unzip -q'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
            if result.returncode == 0 and len(os.listdir(LOCAL_LTX_DIR)) > 2:
                print(f"✅ Tải thành công: {ds}")
                return LOCAL_LTX_DIR
        except Exception as e:
            print(f"   Lỗi khi tải {ds}: {e}")
            continue

    print("⚠️ Không tải được dataset → dùng Hugging Face online")
    return DEFAULT_MODEL_ID

def find_local_model_path():
    return download_ltx_dataset()

# ==========================================
# GOOGLE SHEET UTILS & TẢI ẢNH TỪ LINK SHEET
# ==========================================
def safe_get(row, *keys, default=""):
    for key in keys:
        val = row.get(key)
        if pd.notna(val) and str(val).strip().lower() not in ("nan", "none", "null", "", "[empty]"):
            return str(val).strip()
    return default

def get_sheet_csv(gid: str) -> pd.DataFrame:
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={gid}"
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip().str.lower()
        return df.dropna(how="all")
    except Exception as e:
        print(f"❌ Sheet: {e}")
        return pd.DataFrame()

def download_url(url: str, path: str) -> bool:
    try:
        r = requests.get(url, timeout=90)
        if r.status_code == 200 and len(r.content) > 2000:
            with open(path, "wb") as f:
                f.write(r.content)
            return True
    except Exception as e:
        print(f"❌ Tải ảnh từ URL thất bại: {e}")
    return False

# ==========================================
# GIAI ĐOẠN 1: TẢI ẢNH CHỈ TỪ LINK SHEET
# ==========================================
def step_1_ensure_images():
    print("\n" + "="*55)
    print("🔍 GIAI ĐOẠN 1: Tải ảnh trực tiếp từ link trên Google Sheet (No Drive)")
    print("="*55)
    df = get_sheet_csv(GID_SCENES)
    if df.empty:
        print("❌ Không lấy được dữ liệu từ Google Sheet")
        return False

    ok = 0
    for idx, row in df.iterrows():
        scene_num = int(row["scene_index"]) if pd.notna(row.get("scene_index")) else idx + 1
        if scene_num > TOTAL_SCENES:
            break
        name = f"scene_{scene_num:03d}.png"
        local = f"{OUTPUT_DIR}/{name}"

        # 1. Nếu đã có file sẵn ở local thì giữ nguyên
        if os.path.exists(local) and os.path.getsize(local) > 2000:
            ok += 1
            continue

        # 2. Đọc trực tiếp link từ cột image_url trong Sheet
        img_url = safe_get(row, "image_url", "url", "link_anh")
        if img_url and img_url.startswith("http"):
            print(f"📥 [{scene_num}] Tải ảnh nhân vật từ Sheet: {img_url[:60]}...")
            if download_url(img_url, local):
                ok += 1
                print(f"✅ [{scene_num}] Tải ảnh thành công")
                continue

        print(f"⚠️ [{scene_num}] Không tìm thấy link ảnh hợp lệ trong Sheet!")

    print(f"\n✅ Giai đoạn 1 hoàn tất: {ok} ảnh đã tải về sẵn sàng")
    return ok > 0

# ==========================================
# HÀM HỖ TRỢ KỸ THUẬT RENDER
# ==========================================
def get_video_prompt(row) -> str:
    vp = safe_get(row, "video_prompt")
    if vp: return f"Stable character, subtle movement, cinematic lighting, {vp[:200]}"
    ip = safe_get(row, "image_prompt", "scene_description")
    if ip: return f"Smooth cinematic motion, highly detailed face, consistent character appearance. {ip[:150]}"
    return "Smooth cinematic motion, highly detailed face, consistent character appearance"

def get_dialogue(row) -> str:
    return safe_get(row, "dialogue")[:180]

def make_tts(text: str, out_mp3: str) -> bool:
    if not text or len(text.strip()) < 2: return False
    if os.path.exists(out_mp3) and os.path.getsize(out_mp3) > 400: return True
    safe = text.replace('"', "'").replace("\n", " ")
    for voice in ["vi-VN-NamMinhNeural", "vi-VN-HoaiMyNeural"]:
        try:
            cmd = f'edge-tts --text "{safe}" --voice {voice} --rate=+5% --write-media "{out_mp3}"'
            subprocess.run(cmd, shell=True, timeout=90, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(out_mp3) and os.path.getsize(out_mp3) > 400:
                return True
        except: continue
    return False

def mix_video_audio(video_in, audio_mp3, video_out):
    if not (os.path.exists(audio_mp3) and os.path.getsize(audio_mp3) > 400):
        if video_in != video_out and os.path.exists(video_in):
            subprocess.run(f'cp -f "{video_in}" "{video_out}"', shell=True)
        return video_out if os.path.exists(video_out) else video_in
    cmd = f'ffmpeg -y -hide_banner -loglevel error -i "{video_in}" -i "{audio_mp3}" -c:v copy -c:a aac -b:a 96k -shortest -movflags +faststart "{video_out}"'
    subprocess.run(cmd, shell=True)
    return video_out if os.path.exists(video_out) and os.path.getsize(video_out) > 5000 else video_in

def post_process_video(raw_path, out_path):
    if not os.path.exists(raw_path): return raw_path
    cmd = f'ffmpeg -y -hide_banner -loglevel error -i "{raw_path}" -vf "scale={FINAL_WIDTH}:{FINAL_HEIGHT}:flags=lanczos,fps={FINAL_FPS}" -c:v libx264 -preset fast -crf 17 -pix_fmt yuv420p -movflags +faststart "{out_path}"'
    try:
        subprocess.run(cmd, shell=True, timeout=120)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
            return out_path
    except: pass
    return raw_path

def load_ltx_pipe():
    from diffusers import LTXImageToVideoPipeline
    from diffusers.utils import logging
    logging.set_verbosity_error()

    model_source = find_local_model_path()
    print(f"🧠 Load LTX Image-to-Video Model từ: {model_source}")

    pipe = LTXImageToVideoPipeline.from_pretrained(model_source, torch_dtype=torch.float16)

    if torch.cuda.is_available():
        pipe.enable_sequential_cpu_offload()
        print("✅ sequential_cpu_offload")
        try:
            pipe.vae.enable_slicing()
            print("✅ VAE slicing")
        except: pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✅ LTX Image-to-Video ready")
    return pipe

# ==========================================
# GIAI ĐOẠN 2: LTX IMAGE-TO-VIDEO + TTS
# ==========================================
def step_2_ltx_render():
    from diffusers.utils import export_to_video, load_image
    from PIL import Image

    print("\n" + "="*55)
    print(f"🎬 GIAI ĐOẠN 2: LTX Image-to-Video | {LTX_WIDTH}x{LTX_HEIGHT} | Khóa cố định nhân vật")
    print("="*55)

    pipe = load_ltx_pipe()
    df = get_sheet_csv(GID_SCENES)

    rendered = []
    t0 = time.time()

    for idx, row in df.iterrows():
        scene_num = int(row["scene_index"]) if pd.notna(row.get("scene_index")) else idx + 1
        if scene_num > TOTAL_SCENES: break

        img_path = f"{OUTPUT_DIR}/scene_{scene_num:03d}.png"
        raw_video = f"{VIDEO_DIR}/ltx_raw_{scene_num:03d}.mp4"
        upscaled = f"{VIDEO_DIR}/ltx_up_{scene_num:03d}.mp4"
        voice_mp3 = f"{VIDEO_DIR}/voice_{scene_num:03d}.mp3"
        final_scene = f"{VIDEO_DIR}/scene_{scene_num:03d}.mp4"

        if os.path.exists(final_scene) and os.path.getsize(final_scene) > 200000:
            print(f"⏩ [{scene_num}] Đã có sẵn local")
            rendered.append(final_scene)
            continue

        if not (os.path.exists(img_path) and os.path.getsize(img_path) > 2000):
            print(f"⚠️ [{scene_num}] Thiếu ảnh nhân vật gốc -> Bỏ qua")
            continue

        prompt = get_video_prompt(row)
        dialogue = get_dialogue(row)
        print(f"🎥 [{scene_num}/{TOTAL_SCENES}] {prompt[:70]}...")

        try:
            t1 = time.time()
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            # Load ảnh gốc làm Mỏ neo (Anchor) cố định gương mặt và trang phục
            image = load_image(img_path).convert("RGB").resize((LTX_WIDTH, LTX_HEIGHT), Image.Resampling.LANCZOS)
            
            # Cố định Seed dựa theo Scene ID để đảm bảo tính ổn định qua các khung hình
            generator = torch.Generator(device="cpu").manual_seed(42 + scene_num)

            with torch.inference_mode():
                res = pipe(
                    image=image,
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT_LOCK, # Áp dụng khóa Negative Prompt
                    height=LTX_HEIGHT, width=LTX_WIDTH,
                    num_frames=LTX_NUM_FRAMES,
                    num_inference_steps=LTX_STEPS,
                    generator=generator,
                )

            export_to_video(res.frames[0], raw_video, fps=LTX_FPS)
            post_process_video(raw_video, upscaled)
            make_tts(dialogue, voice_mp3)
            out_file = mix_video_audio(upscaled, voice_mp3, final_scene)

            sz = os.path.getsize(out_file) if os.path.exists(out_file) else 0
            print(f"✅ [{scene_num}] Render xong trong {time.time()-t1:.0f}s | {sz/1024:.0f}KB")

            if sz > 100000:
                rendered.append(out_file)

            for tmp in [raw_video, upscaled]:
                if os.path.exists(tmp) and tmp != out_file:
                    try: os.remove(tmp)
                    except: pass

            del res, image
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Lỗi cảnh [{scene_num}]: {e}")
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    print(f"\nRender xong {len(rendered)} cảnh / tổng thời gian: {(time.time()-t0)/60:.1f} phút")

    valid = [v for v in sorted(rendered) if os.path.exists(v) and os.path.getsize(v) > 100000]
    if not valid:
        print("❌ Không có video hợp lệ")
        return False

    with open("concat_list.txt", "w") as f:
        for v in valid:
            f.write(f"file '{os.path.abspath(v)}'\n")

    subprocess.run(f'ffmpeg -y -hide_banner -loglevel error -f concat -safe 0 -i concat_list.txt -c copy -movflags +faststart "{FINAL_VIDEO}"', shell=True)

    if os.path.exists(FINAL_VIDEO):
        print(f"🎉 Phim hoàn chỉnh: {FINAL_VIDEO} ({os.path.getsize(FINAL_VIDEO)/1024/1024:.1f} MB)")
        return True
    return False

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    try:
        print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    except: pass

    if step_1_ensure_images():
        step_2_ltx_render()
        print("\n🎉 HOÀN TẤT PIPELINE!")
    else:
        print("\n❌ Dừng pipeline: Không lấy được ảnh nhân vật từ Sheet")
