import gc
import os
import subprocess
import sys
import time
import asyncio
from pathlib import Path

RUN_DATE = time.strftime("%Y%m%d")
print(f"🚀 1. KHỞI TẠO PHIÊN RENDER LTX-2.3 + TTS TIẾNG VIỆT - NGÀY [{RUN_DATE}]")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def sync_system_time():
    try:
        subprocess.run(
            "apt-get update -qq && apt-get install -y -qq ntpdate && ntpdate time.google.com",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("⏰ Đã đồng bộ giờ hệ thống thành công!")
    except Exception as e:
        print(f"⚠️ Không thể đồng bộ giờ: {e}")

sync_system_time()

def install_requirements():
    packages = [
        "git+https://github.com/huggingface/diffusers",
        "git+https://github.com/huggingface/transformers",   # Bắt buộc để có Gemma4ForConditionalGeneration
        "imageio-ffmpeg",
        "google-api-python-client",
        "google-auth-oauthlib",
        "huggingface_hub",
        "soundfile",
        "av",
        "edge-tts",
        "accelerate",
        "sentencepiece",
        "protobuf",
    ]
    print("📦 Đang cài đặt packages (bao gồm transformers mới nhất)...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "--upgrade"] + packages
    )

install_requirements()

import torch
import pandas as pd
import requests
import edge_tts
from diffusers import LTX2Pipeline
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT, DISTILLED_SIGMA_VALUES
from diffusers.utils import encode_video
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

print("✅ Cài đặt môi trường thành công!")

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
N8N_WEBHOOK_URL = "https://n8n-latest-namx.onrender.com/webhook/kaggle-video-done"
SHEET_ID = "1DmA-yuPwDl1riceSMGzWPXhuxrL4y987lOZ6Af351l8"
GOOGLE_SHEET_CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
DRIVE_FOLDER_ID = "1oXS7LweDNK2fYsWonQay3U-hUmEIsgCF"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

TTS_VOICE = "vi-VN-NamMinhNeural"   # Nam
# TTS_VOICE = "vi-VN-HoaiMyNeural"  # Nữ

OAUTH_CLIENT_ID = os.environ.get(
    "OAUTH_CLIENT_ID",
    "948179937421-o55enfl61lb8ou0ms2jmrr4dlf1fhgip.apps.googleusercontent.com",
)
OAUTH_CLIENT_SECRET = os.environ.get(
    "OAUTH_CLIENT_SECRET", "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"
)
OAUTH_REFRESH_TOKEN = os.environ.get(
    "OAUTH_REFRESH_TOKEN",
    "1//06AsOeOfzvnpxCgYIARAAGAYSNwF-L9Ir-Yoj_gfy3CYDrDfXfUkE0z95bPruk8RMjNG3Y0F-SuDd-VFnFfIo0jRZ4T8J4oZzmek",
)
TOKEN_URI = "https://oauth2.googleapis.com/token"

def get_oauth_credentials():
    return Credentials(
        token=None,
        refresh_token=OAUTH_REFRESH_TOKEN,
        token_uri=TOKEN_URI,
        client_id=OAUTH_CLIENT_ID,
        client_secret=OAUTH_CLIENT_SECRET,
        scopes=["https://www.googleapis.com/auth/drive"],
    )

# -------------------------------------------------------------------
# 2. ĐỌC GOOGLE SHEETS
# -------------------------------------------------------------------
print("\n📊 2. TẢI DỮ LIỆU TỪ GOOGLE SHEETS...")
try:
    df = pd.read_csv(GOOGLE_SHEET_CSV_URL)
    total_scenes = len(df)
    print(f"✅ Tìm thấy {total_scenes} cảnh từ Google Sheets.")
except Exception as e:
    print(f"❌ Lỗi đọc Google Sheet: {str(e)}")
    sys.exit(1)

project_info = {
    "title": "Chưa đặt tiêu đề",
    "genre": "Mặc định",
    "character_design": "",
    "world_setting": "",
    "visual_style": "",
}

if not df.empty:
    first_row = df.iloc[0]
    for key in project_info.keys():
        if key in df.columns:
            val = str(first_row[key]).strip()
            if val.lower() != "nan" and val != "":
                project_info[key] = val

# === LẤY LỜI KỂ TỪ GOOGLE SHEET ===
NARRATION_TEXT = ""
if "narration" in df.columns:
    for val in df["narration"]:
        text = str(val).strip()
        if text and text.lower() != "nan":
            NARRATION_TEXT = text
            break

if not NARRATION_TEXT:
    NARRATION_TEXT = f"Đây là câu chuyện thuộc thể loại {project_info.get('genre', '')}."

print("==================================================")
print("🎬 THÔNG TIN DỰ ÁN:")
print(f"📌 Tiêu đề             : {project_info['title']}")
print(f"🏷️ Thể loại           : {project_info['genre']}")
print(f"🗣️ Lời kể (narration) : {NARRATION_TEXT[:150]}...")
print("==================================================")

def upload_file_to_drive_fresh(file_path, folder_id, retries=3):
    file_name = os.path.basename(file_path)
    for attempt in range(1, retries + 1):
        try:
            sync_system_time()
            creds = get_oauth_credentials()
            service = build("drive", "v3", credentials=creds, cache_discovery=False)
            file_metadata = {"name": file_name, "parents": [folder_id]}
            media = MediaFileUpload(file_path, mimetype="video/mp4", resumable=True)
            uploaded_file = (
                service.files()
                .create(body=file_metadata, media_body=media, fields="id")
                .execute()
            )
            print(f"☁️ Đã đẩy thành công {file_name}")
            return uploaded_file.get("id")
        except Exception as e:
            print(f"⚠️ [Lần {attempt}/{retries}] Upload thất bại: {str(e)}")
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
        **project_info,
    }
    try:
        res = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=60)
        print(f"📡 [FINAL WEBHOOK {status.upper()}] HTTP {res.status_code}")
    except Exception as e:
        print(f"❌ Lỗi gửi Webhook: {str(e)}")

# -------------------------------------------------------------------
# 3. LOAD MODEL LTX-2.3
# -------------------------------------------------------------------
print("\n🧠 3. KHỞI TẠO MODEL LTX-2.3 DISTILLED...")
try:
    MODEL_ID = "diffusers/LTX-2.3-Distilled-Diffusers"
    pipe = LTX2Pipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN if HF_TOKEN.startswith("hf_") else None,
    )
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    print("✅ Load Model thành công!")
except Exception as e:
    send_n8n_final_webhook("failed", 0, error_message=f"Lỗi load model: {str(e)}")
    sys.exit(1)

# -------------------------------------------------------------------
# 4. RENDER 80 CẢNH
# -------------------------------------------------------------------
rendered_files = []

for index, row in df.iterrows():
    scene_idx_val = row.get("scene_index")
    scene_index = int(scene_idx_val) if pd.notna(scene_idx_val) else index + 1

    prompt = str(row.get("prompt", "")).strip()
    negative_prompt = str(row.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)).strip()
    filename = f"scene_{scene_index:03d}_{RUN_DATE}.mp4"

    if not prompt or prompt.lower() == "nan":
        continue

    if os.path.exists(filename) and os.path.getsize(filename) > 10000:
        print(f"⏩ [{scene_index}/{total_scenes}] Đã tồn tại → bỏ qua")
        rendered_files.append(filename)
        continue

    scene_seed = 42 + scene_index
    generator = torch.Generator(device="cuda").manual_seed(scene_seed)

    print(f"\n🎬 [{scene_index}/{total_scenes}] {filename}")
    print(f"   Prompt: {prompt[:100]}...")

    try:
        video, audio = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=768,
            height=512,
            num_frames=97,
            frame_rate=24.0,
            sigmas=DISTILLED_SIGMA_VALUES,
            guidance_scale=1.0,
            audio_guidance_scale=1.0,
            generator=generator,
            output_type="np",
            return_dict=False,
        )

        encode_video(
            video[0],
            fps=24,
            audio=audio[0].float().cpu() if audio is not None else None,
            audio_sample_rate=48000,
            output_path=filename,
        )

        print(f"💾 Đã lưu: {filename}")
        upload_file_to_drive_fresh(filename, DRIVE_FOLDER_ID)
        rendered_files.append(filename)

    except Exception as e:
        print(f"❌ Lỗi cảnh {scene_index}: {str(e)}")
    finally:
        gc.collect()
        torch.cuda.empty_cache()

# -------------------------------------------------------------------
# 5. GỘP VIDEO
# -------------------------------------------------------------------
print("\n🎞️ 5. GỘP VIDEO...")
if not rendered_files:
    send_n8n_final_webhook("failed", 0, error_message="Không render được cảnh nào.")
    sys.exit(1)

with open("file_list.txt", "w") as f:
    for file in rendered_files:
        f.write(f"file '{file}'\n")

final_model = f"final_model_{RUN_DATE}.mp4"
subprocess.run(f"ffmpeg -f concat -safe 0 -i file_list.txt -c copy {final_model} -y", shell=True, check=True)
print(f"✅ Đã gộp: {final_model}")

# -------------------------------------------------------------------
# 6. TẠO LỜI KỂ TIẾNG VIỆT
# -------------------------------------------------------------------
print("\n🗣️ 6. TẠO LỜI KỂ TIẾNG VIỆT...")
narration_audio = f"narration_{RUN_DATE}.mp3"

async def generate_tts():
    communicate = edge_tts.Communicate(
        text=NARRATION_TEXT.strip(),
        voice=TTS_VOICE,
        rate="+0%",
        volume="+0%",
    )
    await communicate.save(narration_audio)

asyncio.run(generate_tts())
print(f"✅ Đã tạo giọng kể: {narration_audio}")

# -------------------------------------------------------------------
# 7. GHÉP LỜI KỂ + UPLOAD
# -------------------------------------------------------------------
print("\n🎧 7. GHÉP LỜI KỂ VÀO VIDEO...")
final_output = f"final_with_narration_{RUN_DATE}.mp4"

mix_cmd = (
    f'ffmpeg -y -i {final_model} -i {narration_audio} '
    f'-filter_complex "[0:a]volume=0.25[a0];[1:a]volume=1.4[a1];[a0][a1]amix=inputs=2:duration=first[a]" '
    f'-map 0:v -map "[a]" -c:v copy -c:a aac -b:a 192k {final_output}'
)
subprocess.run(mix_cmd, shell=True, check=True)

print(f"🎉 VIDEO HOÀN CHỈNH: {final_output}")

drive_file_id = upload_file_to_drive_fresh(final_output, DRIVE_FOLDER_ID)

send_n8n_final_webhook(
    status="completed_all",
    total_scenes=len(rendered_files),
    final_file=final_output,
    drive_file_id=drive_file_id,
)

print("\n✅ HOÀN TẤT TOÀN BỘ!")
