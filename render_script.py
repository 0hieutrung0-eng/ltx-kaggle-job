import asyncio
import gc
import os
import subprocess
import sys
import time
from pathlib import Path

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
    ]
    print("📦 Đang kiểm tra và cài đặt packages...")
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
print(f"🚀 SDXL → SVD PIPELINE - [{RUN_DATE}]")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

def sync_system_time():
    try:
        subprocess.run(
            "apt-get update -qq && apt-get install -y -qq ntpdate && ntpdate time.google.com",
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception:
        pass

sync_system_time()

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
N8N_WEBHOOK_URL = "https://n8n-latest-namx.onrender.com/webhook/kaggle-video-done"
SHEET_ID = "1DmA-yuPwDl1riceSMGzWPXhuxrL4y987lOZ6Af351l8"
GOOGLE_SHEET_CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
DRIVE_FOLDER_ID = "1oXS7LweDNK2fYsWonQay3U-hUmEIsgCF"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

VOICE_MAP = {"nam": "vi-VN-NamMinhNeural", "nu": "vi-VN-HoaiMyNeural"}

OAUTH_CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID", "948179937421-o55enfl61lb8ou0ms2jmrr4dlf1fhgip.apps.googleusercontent.com")
OAUTH_CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET", "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")
OAUTH_REFRESH_TOKEN = os.environ.get("OAUTH_REFRESH_TOKEN", "1//06AsOeOfzvnpxCgYIARAAGAYSNwF-L9Ir-Yoj_gfy3CYDrDfXfUkE0z95bPruk8RMjNG3Y0F-SuDd-VFnFfIo0jRZ4T8J4oZzmek")
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
        return 2.0

# -------------------------------------------------------------------
# 2. ĐỌC VÀ CHUẨN HÓA SHEETS
# -------------------------------------------------------------------
print("\n📊 2. TẢI DỮ LIỆU TỪ GOOGLE SHEETS...")
try:
    df = pd.read_csv(GOOGLE_SHEET_CSV_URL)
    # Chuẩn hóa tên cột: xóa khoảng trắng và chuyển thành chữ thường
    df.columns = df.columns.str.strip().str.lower()
    total_scenes = len(df)
    print(f"✅ Tìm thấy {total_scenes} cảnh.")
    print(f"📋 Các cột phát hiện được: {df.columns.tolist()}")
except Exception as e:
    print(f"❌ Lỗi đọc Google Sheet: {e}")
    sys.exit(1)

project_info = {
    "title": "Chưa đặt tiêu đề",
    "genre": "Mặc định",
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

def upload_file_to_drive_fresh(file_path, folder_id, retries=3):
    if not os.path.exists(file_path) or not folder_id:
        print(f"⚠️ Bỏ qua upload: File {file_path} không tồn tại.")
        return None

    file_name = os.path.basename(file_path)
    mimetype = 'image/png' if file_name.endswith('.png') else 'video/mp4'

    for attempt in range(1, retries + 1):
        try:
            sync_system_time()
            creds = get_oauth_credentials()
            service = build('drive', 'v3', credentials=creds, cache_discovery=False)
            file_metadata = {'name': file_name, 'parents': [folder_id]}
            media = MediaFileUpload(file_path, mimetype=mimetype, resumable=True)
            uploaded_file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            print(f"☁️ Upload {file_name} lên Drive thành công! ID: {uploaded_file.get('id')}")
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
        print(f"📡 Webhook {status.upper()} → {res.status_code}")
    except Exception as e:
        print(f"❌ Lỗi webhook: {e}")

# -------------------------------------------------------------------
# 3. LOAD MODEL SDXL (Text → Image)
# -------------------------------------------------------------------
print("\n🧠 3. LOAD SDXL (Text → Image)...")
try:
    sdxl = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        token=HF_TOKEN if HF_TOKEN and HF_TOKEN.startswith("hf_") else None,
    )
    sdxl.enable_model_cpu_offload()
    sdxl.vae.enable_tiling()
    print("✅ Load SDXL thành công!")
except Exception as e:
    print(f"❌ Lỗi load SDXL: {e}")
    sys.exit(1)

# -------------------------------------------------------------------
# 4. LOAD MODEL SVD (Image → Video)
# -------------------------------------------------------------------
print("\n🧠 4. LOAD Stable Video Diffusion XT...")
try:
    svd = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        token=HF_TOKEN if HF_TOKEN and HF_TOKEN.startswith("hf_") else None,
    )
    svd.enable_model_cpu_offload()
    print("✅ Load SVD thành công!")
except Exception as e:
    print(f"❌ Lỗi load SVD: {e}")
    sys.exit(1)

# -------------------------------------------------------------------
# 5. PROCESS PIPELINE
# -------------------------------------------------------------------
async def process_video_pipeline():
    rendered_files = []

    DEFAULT_NEGATIVE = (
        "2d, flat drawing, realistic human, photorealistic, blurry, low quality, "
        "distorted face, text, watermark, deformed structures"
    )

    for index, row in df.iterrows():
        scene_idx_val = row.get("scene_index")
        scene_index = int(scene_idx_val) if pd.notna(scene_idx_val) else index + 1

        # Lấy Prompt ảnh chính xác theo các tên cột có thể có
        raw_prompt = ""
        for p_col in ["image_prompt", "prompt", "visual_prompt"]:
            if p_col in row and pd.notna(row[p_col]):
                val = str(row[p_col]).strip()
                if val and val.lower() not in ["nan", "[empty]", "none", "null"]:
                    raw_prompt = val
                    break

        # Lấy Negative Prompt
        raw_negative = str(row.get("negative_prompt", "")).strip()

        # Lấy câu đọc thoại (Thuyết minh hoặc Hội thoại)
        scene_text = ""
        for col in ["narration_vi", "dialogue", "scene_narration_vi", "narration"]:
            if col in row and pd.notna(row[col]):
                val = str(row[col]).strip()
                if val and val.lower() not in ["nan", "[empty]", "none", "null"]:
                    scene_text = val
                    break

        image_file = f"image_{scene_index:03d}_{RUN_DATE}.png"
        raw_video_file = f"raw_scene_{scene_index:03d}_{RUN_DATE}.mp4"
        audio_scene_file = f"audio_scene_{scene_index:03d}_{RUN_DATE}.mp3"
        final_scene_file = f"scene_{scene_index:03d}_{RUN_DATE}.mp4"

        if not raw_prompt:
            print(f"⚠️ Bỏ qua cảnh {scene_index} do không tìm thấy prompt ảnh")
            continue

        if os.path.exists(final_scene_file) and os.path.getsize(final_scene_file) > 10000:
            print(f"⏩ [{scene_index}/{total_scenes}] Đã tồn tại → bỏ qua")
            rendered_files.append(final_scene_file)
            continue

        final_prompt = raw_prompt
        final_negative = raw_negative if (raw_negative and raw_negative.lower() not in ["nan", "[empty]", "none"]) else DEFAULT_NEGATIVE

        scene_seed = 42 + scene_index
        generator = torch.Generator(device="cuda").manual_seed(scene_seed)

        # ---------- 5.1 Text → Image (SDXL) ----------
        print(f"\n🖼️ [{scene_index}/{total_scenes}] Tạo ảnh bằng SDXL...")
        try:
            gc.collect()
            torch.cuda.empty_cache()

            image = sdxl(
                prompt=final_prompt,
                negative_prompt=final_negative,
                width=1024,
                height=576,
                num_inference_steps=25,
                guidance_scale=7.0,
                generator=generator,
            ).images[0]

            image.save(image_file)
            print(f"   ✅ Đã lưu ảnh cục bộ: {image_file}")

            # Upload ngay ảnh PNG lên Google Drive
            upload_file_to_drive_fresh(image_file, DRIVE_FOLDER_ID)

        except Exception as e:
            print(f"❌ Lỗi tạo ảnh cảnh {scene_index}: {e}")
            continue

        # ---------- 5.2 Image → Video (SVD) ----------
        print(f"🎬 [{scene_index}/{total_scenes}] Tạo video từ ảnh bằng SVD...")
        try:
            gc.collect()
            torch.cuda.empty_cache()

            image_input = load_image(image_file).resize((1024, 576))

            frames = svd(
                image_input,
                decode_chunk_size=8,
                generator=generator,
                num_frames=25,
                motion_bucket_id=127,
                noise_aug_strength=0.02,
            ).frames[0]

            export_to_video(frames, raw_video_file, fps=7)
            print(f"   ✅ Đã tạo video thô: {raw_video_file}")

            del frames
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Lỗi tạo video cảnh {scene_index}: {e}")
            if os.path.exists(image_file): os.remove(image_file)
            continue

        # ---------- 5.3 Tạo Voice & Lắp ráp MP4 ----------
        if scene_text:
            print(f"🎙️ Voice cảnh {scene_index}: {scene_text[:50]}...")
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

        # Dọn dẹp tệp tạm
        for f in [image_file, raw_video_file, audio_scene_file]:
            if os.path.exists(f):
                os.remove(f)

        print(f"💾 Hoàn tất cảnh {scene_index}")

        # Upload video MP4 của cảnh lên Google Drive
        upload_file_to_drive_fresh(final_scene_file, DRIVE_FOLDER_ID)
        rendered_files.append(final_scene_file)

    # -------------------------------------------------------------------
    # 6. GỘP CẢNH + BGM + HOÀN TẤT
    # -------------------------------------------------------------------
    print("\n🎞️ Gộp tất cả các cảnh thành phim...")
    if not rendered_files:
        send_n8n_final_webhook("failed", 0, error_message="Không render được cảnh nào.")
        sys.exit(1)

    with open("file_list.txt", "w") as f:
        for file in rendered_files:
            f.write(f"file '{file}'\n")

    concat_output = f"final_concat_{RUN_DATE}.mp4"
    final_output = f"final_movie_{RUN_DATE}.mp4"

    subprocess.run(f"ffmpeg -y -f concat -safe 0 -i file_list.txt -c copy {concat_output}", shell=True, check=True)

    bgm_file = "bgm_xianxia.mp3"
    if os.path.exists(bgm_file):
        bgm_cmd = (
            f'ffmpeg -y -i {concat_output} -stream_loop -1 -i {bgm_file} '
            f'-filter_complex "[0:a]volume=1.2[v_tts];[1:a]volume=0.15[v_bgm];'
            f'[v_tts][v_bgm]amix=inputs=2:duration=first[a]" '
            f'-map 0:v:0 -map "[a]" -c:v copy -c:a aac -ar 44100 -ac 2 -b:a 192k {final_output}'
        )
        try:
            subprocess.run(bgm_cmd, shell=True, check=True)
        except Exception:
            final_output = concat_output
    else:
        final_output = concat_output

    print(f"\n☁️ Upload phim hoàn chỉnh...")
    drive_file_id = upload_file_to_drive_fresh(final_output, DRIVE_FOLDER_ID)

    send_n8n_final_webhook(
        status="completed_all",
        total_scenes=len(rendered_files),
        final_file=final_output,
        drive_file_id=drive_file_id
    )
    print("\n🎉 HOÀN TẤT PIPELINE!")

try:
    asyncio.run(process_video_pipeline())
except RuntimeError:
    loop = asyncio.get_event_loop()
    loop.run_until_complete(process_video_pipeline())
