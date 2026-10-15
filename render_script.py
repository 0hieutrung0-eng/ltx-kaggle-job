# ========== FULL PIPELINE 1 GPU T4: Optimized Text-to-Video AI → TTS → Drive Backup → Merge ==========
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

def install_all_deps():
    print("📦 Đang kiểm tra và cài đặt dependency...")
    try:
        import ftfy
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "ftfy"], check=False)

    try:
        from diffusers import LTXPipeline
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

install_all_deps()

import torch
import ftfy
from diffusers.utils import export_to_video

TOTAL_SCENES = 100
VIDEO_DIR = "./output_videos"
FINAL_VIDEO = "final_full_movie.mp4"
LOCAL_LTX_DIR = "/kaggle/working/ltx-video-model"

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(LOCAL_LTX_DIR, exist_ok=True)

DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID", "").strip()
CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET", "")
REFRESH_TOKEN = os.environ.get("OAUTH_REFRESH_TOKEN", "")
SHEET_ID = os.environ.get("SHEET_ID", "1DmA-yuPwDl1riceSMGzWPXhuxrL4y987lOZ6Af351l8")
GID_CHARACTERS = os.environ.get("GID_CHARACTERS", "1382939846")
GID_SCENES = os.environ.get("GID_SCENES", "0")

DEFAULT_MODEL_ID = "Lightricks/LTX-Video"
LTX_WIDTH = 512
LTX_HEIGHT = 512
LTX_NUM_FRAMES = 81  # Giảm số frame xuống 33 để tiết kiệm VRAM T4, tránh lỗi tràn bộ nhớ sinh video trắng
LTX_STEPS = 25
LTX_FPS = 24
FINAL_FPS = 24
FINAL_WIDTH = 768
FINAL_HEIGHT = 512

NEGATIVE_PROMPT_LOCK = (
    "black screen, blank video, static image, no motion, worst quality, low quality, blurry, "
    "distorted face, deformed eyes, morphed limbs, extra limbs, bad anatomy, 2d, chibi"
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:32"

def get_google_access_token():
    if not all([CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN]):
        return None
    try:
        res = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "refresh_token": REFRESH_TOKEN,
                "grant_type": "refresh_token",
            },
            timeout=20,
        ).json()
        return res.get("access_token")
    except:
        return None

def upload_file_to_drive(file_path: str):
    token = get_google_access_token()
    if not token or not DRIVE_FOLDER_ID:
        return None
    if not os.path.exists(file_path) or os.path.getsize(file_path) < 50000: # Chặn không upload file dưới 50KB (file lỗi)
        print(f"⚠️ Bỏ qua upload vì file quá nhỏ hoặc lỗi: {file_path}")
        return None
    file_name = os.path.basename(file_path)
    metadata = {"name": file_name, "parents": [DRIVE_FOLDER_ID]}
    headers = {"Authorization": f"Bearer {token}"}
    try:
        with open(file_path, "rb") as f:
            files = {
                "data": ("metadata", json.dumps(metadata), "application/json; charset=UTF-8"),
                "file": (file_name, f, "application/octet-stream"),
            }
            res = requests.post(
                "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
                headers=headers, files=files, timeout=600,
            )
        if res.status_code in (200, 201):
            print(f"✅ Đã backup thành công lên Drive: {file_name}")
            return res.json()
    except Exception as e:
        print(f"❌ Lỗi backup Drive: {e}")
    return None

def download_ltx_dataset():
    if os.path.exists(LOCAL_LTX_DIR) and len(os.listdir(LOCAL_LTX_DIR)) > 3:
        return LOCAL_LTX_DIR
    return DEFAULT_MODEL_ID

def find_local_model_path():
    return download_ltx_dataset()

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
    except:
        return pd.DataFrame()

def load_character_profiles():
    df_char = get_sheet_csv(GID_CHARACTERS)
    if df_char.empty:
        return {}
    char_prompts = {}
    for idx, row in df_char.iterrows():
        char_id = safe_get(row, "character_id").lower()
        char_name = safe_get(row, "character_name")
        appearance = safe_get(row, "appearance", "design_detail", "costume")
        if char_id: char_prompts[char_id] = appearance
        if char_name: char_prompts[char_name.lower()] = appearance
    return char_prompts

def build_pure_text_prompt(row, char_prompts) -> str:
    scene_desc = safe_get(row, "video_prompt", "scene_description", "action")
    char_key = safe_get(row, "character_id", "character", "nhan_vat").lower()
    char_details = char_prompts.get(char_key, "3d chinese donghua style character")
    
    full_prompt = (
        f"Masterpiece, 3d chinese donghua style, unreal engine 5, cinematic neon noir lighting, highly detailed. "
        f"Character features: {char_details[:200]}. "
        f"Action sequence: {scene_desc[:250]}, smooth motion, dynamic camera angles."
    )
    return full_prompt

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

def load_ltx_text_pipe():
    from diffusers import LTXPipeline
    from diffusers.utils import logging
    logging.set_verbosity_error()

    model_source = find_local_model_path()
    pipe = LTXPipeline.from_pretrained(model_source, torch_dtype=torch.float16)

    if torch.cuda.is_available():
        pipe.enable_sequential_cpu_offload()
        try:
            pipe.vae.enable_slicing()
        except: pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pipe

def run_pipeline():
    char_prompts = load_character_profiles()
    df_scenes = get_sheet_csv(GID_SCENES)
    if df_scenes.empty:
        return False

    pipe = load_ltx_text_pipe()
    rendered = []
    t0 = time.time()

    for idx, row in df_scenes.iterrows():
        scene_num = int(row["scene_index"]) if pd.notna(row.get("scene_index")) else idx + 1
        if scene_num > TOTAL_SCENES: break

        raw_video = f"{VIDEO_DIR}/ltx_raw_{scene_num:03d}.mp4"
        upscaled = f"{VIDEO_DIR}/ltx_up_{scene_num:03d}.mp4"
        voice_mp3 = f"{VIDEO_DIR}/voice_{scene_num:03d}.mp3"
        final_scene = f"{VIDEO_DIR}/scene_{scene_num:03d}.mp4"

        if os.path.exists(final_scene) and os.path.getsize(final_scene) > 50000:
            rendered.append(final_scene)
            continue

        text_prompt = build_pure_text_prompt(row, char_prompts)
        dialogue = get_dialogue(row)
        
        print(f"🎥 [{scene_num}/{TOTAL_SCENES}] Render Text-to-Video...")

        try:
            t1 = time.time()
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            generator = torch.Generator(device="cpu").manual_seed(42 + scene_num)

            with torch.inference_mode():
                res = pipe(
                    prompt=text_prompt,
                    negative_prompt=NEGATIVE_PROMPT_LOCK,
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
            print(f"✅ [{scene_num}] Hoàn tất | Kích thước: {sz/1024:.0f}KB")

            if sz > 50000:
                upload_file_to_drive(out_file)
                rendered.append(out_file)

            for tmp in [raw_video, upscaled]:
                if os.path.exists(tmp) and tmp != out_file:
                    try: os.remove(tmp)
                    except: pass

            del res
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Lỗi cảnh [{scene_num}]: {e}")
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    valid = [v for v in sorted(rendered) if os.path.exists(v) and os.path.getsize(v) > 50000]
    if not valid:
        return False

    with open("concat_list.txt", "w") as f:
        for v in valid:
            f.write(f"file '{os.path.abspath(v)}'\n")

    subprocess.run(f'ffmpeg -y -hide_banner -loglevel error -f concat -safe 0 -i concat_list.txt -c copy -movflags +faststart "{FINAL_VIDEO}"', shell=True)

    if os.path.exists(FINAL_VIDEO):
        upload_file_to_drive(FINAL_VIDEO)
        return True
    return False

if __name__ == "__main__":
    run_pipeline()
