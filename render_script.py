# ========== FULL PIPELINE 1 GPU T4: Pure Text-to-Video AI (LTXPipeline) → TTS → Drive Backup → Merge ==========
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
        from diffusers import LTXPipeline
        print("✅ diffusers + LTXPipeline đã có")
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
from diffusers.utils import export_to_video

# ==========================================
# CẤU HÌNH PIPELINE
# ==========================================
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
LTX_NUM_FRAMES = 65
LTX_STEPS = 25
LTX_FPS = 24
FINAL_FPS = 24
FINAL_WIDTH = 768
FINAL_HEIGHT = 512

# Bộ Negative Prompt chuẩn chống méo mó, bảo toàn tỷ lệ và không gian
NEGATIVE_PROMPT_LOCK = (
    "worst quality, low quality, blurry, jittery, distorted face, "
    "deformed eyes, morphed limbs, extra limbs, bad anatomy, warped features, "
    "changing face shape, duplicate heads, unstable character design, 2d, flat color, chibi"
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:32"

# ==========================================
# GOOGLE OAUTH + DRIVE BACKUP UTILS
# ==========================================
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
    except Exception as e:
        print(f"❌ OAuth: {e}")
        return None

def upload_file_to_drive(file_path: str):
    token = get_google_access_token()
    if not token or not DRIVE_FOLDER_ID:
        return None
    if not os.path.exists(file_path) or os.path.getsize(file_path) < 1000:
        return None
    file_name = os.path.basename(file_path)
    metadata = {"name": file_name, "parents": [DRIVE_FOLDER_ID]}
    headers = {"Authorization": f"Bearer {token}"}
    print(f"🚀 Đang backup file {file_name} lên Google Drive...")
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
# GOOGLE SHEET UTILS
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
        print(f"❌ Sheet (GID {gid}): {e}")
        return pd.DataFrame()

# ==========================================
# ĐỌC THÔNG TIN NHÂN VẬT TỪ TAB CHARACTERS
# ==========================================
def load_character_profiles():
    print("\n" + "="*55)
    print("🔍 Đọc thông tin mô tả nhân vật từ Tab Characters")
    print("="*55)
    
    df_char = get_sheet_csv(GID_CHARACTERS)
    if df_char.empty:
        print("❌ Không đọc được bảng Characters từ Google Sheet!")
        return {}

    char_prompts = {}
    for idx, row in df_char.iterrows():
        char_id = safe_get(row, "character_id").lower()
        char_name = safe_get(row, "character_name")
        appearance = safe_get(row, "appearance", "design_detail", "costume")

        if char_id:
            char_prompts[char_id] = appearance
        if char_name:
            char_prompts[char_name.lower()] = appearance

    print(f"🎯 Đã nạp đặc điểm ngoại hình cho {len(char_prompts)} thực thể/nhân vật.")
    return char_prompts

# ==========================================
# XỬ LÝ PROMPT TỪNG CẢNH VÀ TTS
# ==========================================
def build_pure_text_prompt(row, char_prompts) -> str:
    scene_desc = safe_get(row, "video_prompt", "scene_description", "action")
    char_key = safe_get(row, "character_id", "character", "nhan_vat").lower()
    
    # Lấy mô tả ngoại hình nhân vật từ Sheet để gộp vào prompt giữ nguyên nhận diện
    char_details = char_prompts.get(char_key, "3d chinese donghua style character")
    
    # Prompt hoàn toàn bằng văn bản (Text-to-Video)
    full_prompt = (
        f"3d chinese donghua style, unreal engine 5, cinematic neon noir lighting, highly detailed. "
        f"Character: {char_details[:200]}. "
        f"Action & Movement: {scene_desc[:250]}, smooth motion, dynamic camera angles."
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
    print(f"🧠 Load LTX Text-to-Video Pipeline từ: {model_source}")

    pipe = LTXPipeline.from_pretrained(model_source, torch_dtype=torch.float16)

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
    print("✅ LTX Text-to-Video Pipeline ready")
    return pipe

# ==========================================
# GIAI ĐOẠN 2: RENDER TEXT-TO-VIDEO THUẦN TÚY
# ==========================================
def run_pipeline():
    char_prompts = load_character_profiles()

    print("\n" + "="*55)
    print(f"🎬 GIAI ĐOẠN 2: Chạy Text-to-Video AI theo kịch bản Scenes | {LTX_WIDTH}x{LTX_HEIGHT}")
    print("="*55)

    df_scenes = get_sheet_csv(GID_SCENES)
    if df_scenes.empty:
        print("❌ Không đọc được bảng kịch bản Scenes!")
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

        if os.path.exists(final_scene) and os.path.getsize(final_scene) > 200000:
            print(f"⏩ [{scene_num}] Đã có sẵn local")
            rendered.append(final_scene)
            continue

        text_prompt = build_pure_text_prompt(row, char_prompts)
        dialogue = get_dialogue(row)
        
        print(f"🎥 [{scene_num}/{TOTAL_SCENES}] Text-to-Video Prompt: {text_prompt[:80]}...")

        try:
            t1 = time.time()
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            generator = torch.Generator(device="cpu").manual_seed(42 + scene_num)

            with torch.inference_mode():
                # Pipeline thuần túy Text-to-Video: Tạo chuyển động và cảnh phim hoàn toàn bằng văn bản
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
            print(f"✅ [{scene_num}] Hoàn tất trong {time.time()-t1:.0f}s | {sz/1024:.0f}KB")

            if sz > 100000:
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
        upload_file_to_drive(FINAL_VIDEO)
        return True
    return False

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    try:
        print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    except: pass

    success = run_pipeline()
    if success:
        print("\n🎉 HOÀN TẤT PIPELINE TEXT-TO-VIDEO THÀNH CÔNG!")
    else:
        print("\n❌ PIPELINE DỪNG DO LỖI.")
