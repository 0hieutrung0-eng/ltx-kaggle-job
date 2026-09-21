import asyncio
import gc
import os
import subprocess
import sys
import time
from pathlib import Path

# -------------------------------------------------------------------
# 1. INSTALL REQUIREMENTS
# -------------------------------------------------------------------
def install_requirements():
    packages = [
        "nest_asyncio",
        "diffusers>=0.32.0",
        "transformers",
        "accelerate",
        "imageio-ffmpeg",
        "google-api-python-client",
        "google-auth-oauthlib",
        "huggingface_hub",
        "soundfile",
        "av",
        "edge-tts",
        "protobuf<6.0.0,>=3.20.2",
        "sentencepiece",
        "ftfy",
        "safetensors",
        "omegaconf",
        "einops",
        "opencv-python",
        "pandas",
        "requests"
    ]
    print("📦 Đang kiểm tra & cài đặt packages...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "--no-warn-script-location", "--disable-pip-version-check"
    ] + packages)

install_requirements()

import nest_asyncio
nest_asyncio.apply()

import torch
import numpy as np
import pandas as pd
import requests
import edge_tts
from PIL import Image
from diffusers import StableDiffusionXLPipeline, StableVideoDiffusionPipeline
from diffusers.utils import export_to_video, load_image
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

RUN_DATE = time.strftime("%Y%m%d_%H%M%S")
print(f"🚀 SDXL → SVD I2V PIPELINE - [{RUN_DATE}]")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

# Đồng bộ thời gian hệ thống 1 lần duy nhất ở đầu script
def sync_system_time_once():
    try:
        subprocess.run(
            "apt-get update -qq && apt-get install -y -qq ntpdate && ntpdate time.google.com",
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception:
        pass

sync_system_time_once()

# -------------------------------------------------------------------
# CONFIGURATION (LẤY TỪ MÔI TRƯỜNG ĐỂ BẢO MẬT)
# -------------------------------------------------------------------
N8N_WEBHOOK_URL = os.environ.get("N8N_WEBHOOK_URL", "https://n8n-latest-namx.onrender.com/webhook/kaggle-video-done")
SHEET_ID = os.environ.get("GOOGLE_SHEET_ID", "1DmA-yuPwDl1riceSMGzWPXhuxrL4y987lOZ6Af351l8")
GOOGLE_SHEET_CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID", "1oXS7LweDNK2fYsWonQay3U-hUmEIsgCF")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

VOICE_MAP = {"nam": "vi-VN-NamMinhNeural", "nu": "vi-VN-HoaiMyNeural"}

OAUTH_CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID", "")
OAUTH_CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET", "")
OAUTH_REFRESH_TOKEN = os.environ.get("OAUTH_REFRESH_TOKEN", "")
TOKEN_URI = "https://oauth2.googleapis.com/token"

def get_oauth_credentials():
    return Credentials(
        token=None,
        refresh_token=OAUTH_REFRESH_TOKEN,
        token_uri=TOKEN_URI,
        client_id=OAUTH_CLIENT_ID,
        client_secret=OAUTH_CLIENT_SECRET,
        scopes=["https://www.googleapis.com/auth/drive"]
    )

def get_media_duration(file_path):
    cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{file_path}"'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        return 3.0

def upload_file_to_drive_fresh(file_path, folder_id, retries=3):
    if not OAUTH_REFRESH_TOKEN:
        print("⚠️ Bỏ qua upload Drive: Chưa cấu hình OAUTH_REFRESH_TOKEN.")
        return None
        
    file_name = os.path.basename(file_path)
    for attempt in range(1, retries + 1):
        try:
            creds = get_oauth_credentials()
            service = build('drive', 'v3', credentials=creds, cache_discovery=False)
            file_metadata = {'name': file_name, 'parents': [folder_id]}
            media = MediaFileUpload(file_path, mimetype='video/mp4', resumable=True)
            uploaded_file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            print(f"☁️ Upload Google Drive thành công: {file_name}")
            return uploaded_file.get('id')
        except Exception as e:
            print(f"⚠️ Upload lần {attempt} thất bại: {e}")
            if attempt < retries:
                time.sleep(3)
            else:
                return None

def send_n8n_final_webhook(status, total_scenes, final_file=None, drive_file_id=None, error_message=None):
    payload = {
        "status": status,
        "total_scenes": total_scenes,
        "final_file": final_file,
        "drive_file_id": drive_file_id,
        "error_message": error_message,
        **project_info
    }
    try:
        res = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=60)
        print(f"📡 Webhook {status.upper()} → HTTP {res.status_code}")
    except Exception as e:
        print(f"❌ Lỗi gửi Webhook: {e}")

# -------------------------------------------------------------------
# 2. ĐỌC DỮ LIỆU TỪ GOOGLE SHEETS
# -------------------------------------------------------------------
print("\n📊 2. TẢI DỮ LIỆU TỪ GOOGLE SHEETS...")
try:
    df = pd.read_csv(GOOGLE_SHEET_CSV_URL)
    total_scenes = len(df)
    print(f"✅ Tìm thấy tổng cộng {total_scenes} cảnh.")
except Exception as e:
    print(f"❌ Lỗi đọc Google Sheet CSV: {e}")
    sys.exit(1)

project_info = {
    "title": "Chưa đặt tiêu đề",
    "genre": "Tiên Hiệp",
    "character_design": "",
    "world_setting": "",
    "visual_style": "",
    "gender": "nam"
}

if not df.empty:
    first_row = df.iloc[0]
    for key in project_info.keys():
        if key in df.columns:
            val = str(first_row[key]).strip()
            if val.lower() not in ["nan", "[empty]", ""]:
                project_info[key] = val

SELECTED_GENDER = project_info.get("gender", "nam").lower()
ACTIVE_VOICE = VOICE_MAP.get(SELECTED_GENDER, "vi-VN-NamMinhNeural")

# -------------------------------------------------------------------
# 3. QUẢN LÝ BỘ NHỚ MODEL (SDXL & SVD)
# -------------------------------------------------------------------
def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()

def load_sdxl_pipeline():
    print("\n🧠 Load SDXL Pipeline (Text → Image)...")
    sdxl = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        token=HF_TOKEN if HF_TOKEN.startswith("hf_") else None,
    )
    sdxl.enable_model_cpu_offload()
    sdxl.enable_vae_tiling()
    return sdxl

def load_svd_pipeline():
    print("\n🧠 Load Stable Video Diffusion Pipeline (Image → Video)...")
    svd = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16",
        token=HF_TOKEN if HF_TOKEN.startswith("hf_") else None,
    )
    svd.enable_model_cpu_offload()
    return svd

# -------------------------------------------------------------------
# 4. THỰC THI PIPELINE CHÍNH (IMAGE-TO-VIDEO)
# -------------------------------------------------------------------
async def process_video_pipeline():
    rendered_files = []

    STYLE_PREFIX = (
        "3d chinese donghua animation style, unreal engine 5 render, "
        "extremely detailed 3d face, anime style, cinematic lighting, masterpiece, best quality, "
    )
    DEFAULT_NEGATIVE = (
        "2d, flat drawing, realistic human, photorealistic, blurry, low quality, "
        "distorted face, morphing, text, watermark, stiff pose, deformed hands, extra limbs"
    )

    sdxl = load_sdxl_pipeline()
    generated_images = {}

    # PHASE 1: GENERATE ALL IMAGES (SDXL)
    print("\n🎨 --- GIAI ĐOẠN 1: TẠO ẢNH TĨNH TỪ IMAGE_PROMPT (SDXL) ---")
    for index, row in df.iterrows():
        scene_idx_val = row.get("scene_index")
        scene_index = int(scene_idx_val) if pd.notna(scene_idx_val) else index + 1

        image_prompt_raw = str(row.get("image_prompt", "")).strip()
        if not image_prompt_raw or image_prompt_raw.lower() in ["nan", "[empty]", "none", "null"]:
            image_prompt_raw = str(row.get("prompt", "")).strip()

        negative_prompt = str(row.get("negative_prompt", "")).strip()
        if not negative_prompt or negative_prompt.lower() in ["nan", "[empty]", "none", "null"]:
            negative_prompt = DEFAULT_NEGATIVE

        image_file = f"image_{scene_index:03d}_{RUN_DATE}.png"

        if not image_prompt_raw or image_prompt_raw.lower() in ["nan", "[empty]", "none"]:
            print(f"⚠️ Bỏ qua tạo ảnh Cảnh {scene_index}: prompt rỗng.")
            continue

        if os.path.exists(image_file):
            print(f"⏩ Ảnh cảnh {scene_index} đã tồn tại → dùng lại: {image_file}")
            generated_images[scene_index] = image_file
            continue

        final_prompt = f"{STYLE_PREFIX} {image_prompt_raw}"
        scene_seed = 42 + scene_index
        generator = torch.Generator(device="cuda").manual_seed(scene_seed)

        print(f"🖼️ [{scene_index}/{total_scenes}] Đang tạo ảnh bằng SDXL...")
        try:
            clear_vram()
            image = sdxl(
                prompt=final_prompt,
                negative_prompt=negative_prompt,
                width=1024,
                height=576,
                num_inference_steps=25,
                guidance_scale=7.0,
                generator=generator,
            ).images[0]

            image.save(image_file)
            generated_images[scene_index] = image_file
            print(f"   ✅ Đã lưu ảnh: {image_file}")
        except Exception as e:
            print(f"❌ Lỗi tạo ảnh ở Cảnh {scene_index}: {e}")

    del sdxl
    clear_vram()

    # PHASE 2: GENERATE VIDEO & AUDIO (SVD + Edge-TTS)
    print("\n🎬 --- GIAI ĐOẠN 2: CHUYỂN ĐỔI ẢNH THÀNH VIDEO (SVD) & GHÉP LỜI THOẠI ---")
    svd = load_svd_pipeline()

    for index, row in df.iterrows():
        scene_idx_val = row.get("scene_index")
        scene_index = int(scene_idx_val) if pd.notna(scene_idx_val) else index + 1
        location_val = str(row.get("location", "Location")).strip()

        scene_text = ""
        for col in ["dialogue", "scene_narration_vi", "narration_vi", "narration"]:
            if col in row and pd.notna(row[col]):
                val = str(row[col]).strip()
                if val and val.lower() not in ["nan", "[empty]", "none", "null"]:
                    scene_text = val
                    break

        image_file = generated_images.get(scene_index, f"image_{scene_index:03d}_{RUN_DATE}.png")
        raw_video_file = f"raw_scene_{scene_index:03d}_{RUN_DATE}.mp4"
        audio_scene_file = f"audio_scene_{scene_index:03d}_{RUN_DATE}.mp3"
        final_scene_file = f"scene_{scene_index:03d}_{RUN_DATE}.mp4"

        if os.path.exists(final_scene_file) and os.path.getsize(final_scene_file) > 10000:
            print(f"⏩ [{scene_index}/{total_scenes}] Video hoàn chỉnh đã tồn tại → Bỏ qua render.")
            rendered_files.append(final_scene_file)
            continue

        if not os.path.exists(image_file):
            print(f"⚠️ Không tìm thấy ảnh đầu vào cho Cảnh {scene_index}. Bỏ qua...")
            continue

        duration_sec = float(row.get("duration_sec", 3)) if pd.notna(row.get("duration_sec")) else 3.0
        num_frames = int(max(14, min(30, duration_sec * 7)))

        scene_seed = 42 + scene_index
        generator = torch.Generator(device="cuda").manual_seed(scene_seed)

        # 4.1 Sinh Video bằng SVD
        print(f"📹 [{scene_index}/{total_scenes}] Render video SVD ({location_val}) - Frame count: {num_frames}...")
        try:
            clear_vram()
            img_input = load_image(image_file).resize((1024, 576))

            frames = svd(
                img_input,
                decode_chunk_size=8,
                generator=generator,
                num_frames=num_frames,
                motion_bucket_id=127,
                noise_aug_strength=0.02,
            ).frames[0]

            export_to_video(frames, raw_video_file, fps=7)
            print(f"   ✅ Đã tạo video thô: {raw_video_file}")

            del frames
            clear_vram()
        except Exception as e:
            print(f"❌ Lỗi render SVD cảnh {scene_index}: {e}")
            continue

        # 4.2 Lồng thoại Edge-TTS
        if scene_text:
            print(f"🎙️ Tạo thoại cảnh {scene_index}: '{scene_text[:40]}...'")
            communicate = edge_tts.Communicate(text=scene_text, voice=ACTIVE_VOICE)
            await communicate.save(audio_scene_file)

            v_dur = get_media_duration(raw_video_file)
            a_dur = get_media_duration(audio_scene_file)
            pad_dur = max(0.0, a_dur - v_dur)

            if pad_dur > 0:
                mix_cmd = (
                    f'ffmpeg -y -i "{raw_video_file}" -i "{audio_scene_file}" '
                    f'-filter_complex "[0:v]tpad=stop_mode=clone:stop_duration={pad_dur:.3f}[v]" '
                    f'-map "[v]" -map 1:a:0 -c:v libx264 -pix_fmt yuv420p -r 7 '
                    f'-c:a aac -ar 44100 -ac 2 -b:a 192k "{final_scene_file}"'
                )
            else:
                mix_cmd = (
                    f'ffmpeg -y -i "{raw_video_file}" -i "{audio_scene_file}" '
                    f'-map 0:v:0 -map 1:a:0 -c:v libx264 -pix_fmt yuv420p -r 7 '
                    f'-c:a aac -ar 44100 -ac 2 -b:a 192k -shortest "{final_scene_file}"'
                )
            subprocess.run(mix_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            silent_cmd = (
                f'ffmpeg -y -i "{raw_video_file}" -f lavfi -i anullsrc=r=44100:cl=stereo '
                f'-c:v libx264 -pix_fmt yuv420p -r 7 -c:a aac -ar 44100 -ac 2 -b:a 192k '
                f'-shortest "{final_scene_file}"'
            )
            subprocess.run(silent_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        for f in [raw_video_file, audio_scene_file]:
            if os.path.exists(f):
                os.remove(f)

        print(f"💾 Cảnh {scene_index} đã tạo thành công!")
        upload_file_to_drive_fresh(final_scene_file, DRIVE_FOLDER_ID)
        rendered_files.append(final_scene_file)

    del svd
    clear_vram()

    # PHASE 3: CONCAT ALL SCENES
    print("\n🎞️ --- GIAI ĐOẠN 3: GHÉP TẤT CẢ CÁC CẢNH THÀNH PHIM HOÀN CHỈNH ---")
    if not rendered_files:
        send_n8n_final_webhook("failed", 0, error_message="Không render được cảnh nào.")
        sys.exit(1)

    with open("file_list.txt", "w") as f:
        for file in rendered_files:
            f.write(f"file '{file}'\n")

    concat_output = f"final_concat_{RUN_DATE}.mp4"
    final_output = f"final_movie_{RUN_DATE}.mp4"

    subprocess.run(f'ffmpeg -y -f concat -safe 0 -i file_list.txt -c copy "{concat_output}"', shell=True, check=True)

    bgm_file = "bgm_xianxia.mp3"
    if os.path.exists(bgm_file):
        bgm_cmd = (
            f'ffmpeg -y -i "{concat_output}" -stream_loop -1 -i "{bgm_file}" '
            f'-filter_complex "[0:a]volume=1.2[v_tts];[1:a]volume=0.15[v_bgm];'
            f'[v_tts][v_bgm]amix=inputs=2:duration=first[a]" '
            f'-map 0:v:0 -map "[a]" -c:v copy -c:a aac -ar 44100 -ac 2 -b:a 192k "{final_output}"'
        )
        try:
            subprocess.run(bgm_cmd, shell=True, check=True)
        except Exception:
            final_output = concat_output
    else:
        final_output = concat_output

    print("\n☁️ Tiến hành Upload Phim Hoàn Chỉnh lên Google Drive...")
    drive_file_id = upload_file_to_drive_fresh(final_output, DRIVE_FOLDER_ID)

    send_n8n_final_webhook(
        status="completed_all",
        total_scenes=len(rendered_files),
        final_file=final_output,
        drive_file_id=drive_file_id
    )
    print("\n🎉 HOÀN TẤT QUY TRÌNH TỰ ĐỘNG SDXL → SVD I2V!")

# -------------------------------------------------------------------
# KÍCH HOẠT VÒNG LẶP ASYNCIO TRÊN KAGGLE / JUPYTER
# -------------------------------------------------------------------
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    if loop.is_running():
        asyncio.ensure_future(process_video_pipeline())
    else:
        loop.run_until_complete(process_video_pipeline())
