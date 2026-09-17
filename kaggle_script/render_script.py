import gc
import os
import subprocess
import sys
import time
import json
import base64
import pandas as pd
import requests

print("🚀 1. CÀI ĐẶT CÁC THƯ VIỆN CẦN THIẾT VÀ ĐỒNG BỘ GIỜ HỆ THỐNG...")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def sync_system_time():
    try:
        subprocess.run(
            "apt-get update -qq && apt-get install -y -qq ntpdate && ntpdate time.google.com",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("⏰ Đã đồng bộ giờ hệ thống với time.google.com thành công!")
    except Exception as e:
        print(f"⚠️ Không thể đồng bộ giờ: {e}")

sync_system_time()

def install_requirements():
    packages = [
        "diffusers",
        "transformers",
        "accelerate",
        "imageio-ffmpeg",
        "requests",
        "pandas",
        "torch",
        "google-api-python-client",
        "google-auth",
    ]
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q"] + packages
    )

install_requirements()

import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

print("✅ Cài đặt môi trường thành công!")

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
N8N_WEBHOOK_URL = "https://n8n-latest-namx.onrender.com/webhook/kaggle-video-done"
SHEET_ID = "1DmA-yuPwDl1riceSMGzWPXhuxrL4y987lOZ6Af351l8"
GOOGLE_SHEET_CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
DRIVE_FOLDER_ID = "1oXS7LweDNK2fYsWonQay3U-hUmEIsgCF"

DATASET_MODEL_PATH = "/kaggle/input/ltx-video-weights/LTX-Video-Local"
WORKING_MODEL_PATH = "/kaggle/working/LTX-Video-Local"

# -------------------------------------------------------------------
# LOAD SERVICE ACCOUNT (được inject bởi GitHub Actions - Base64)
# -------------------------------------------------------------------
SERVICE_ACCOUNT_B64 = "___GOOGLE_SERVICE_ACCOUNT_B64_PLACEHOLDER___"

def load_service_account():
    # Kiểm tra bằng độ dài (placeholder gốc rất ngắn < 100 ký tự)
    if len(SERVICE_ACCOUNT_B64) < 100:
        raise ValueError(
            "Service Account chưa được inject (placeholder còn nguyên). "
            "Hãy kiểm tra GitHub Actions."
        )
    try:
        json_str = base64.b64decode(SERVICE_ACCOUNT_B64).decode("utf-8")
        data = json.loads(json_str)
        print("✅ Đã load Service Account từ Base64 (injected bởi GitHub Actions)")
        return data
    except Exception as e:
        raise ValueError(f"Lỗi decode/parse Service Account: {e}")

SERVICE_ACCOUNT_INFO = load_service_account()

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

print("==================================================")
print("🎬 THÔNG TIN DỰ ÁN (Đọc từ Google Sheets):")
print(f"📌 Tiêu đề           : {project_info['title']}")
print(f"🏷️ Thể loại          : {project_info['genre']}")
print(f"👤 Thiết kế Nhân vật : {project_info['character_design']}")
print(f"🏰 Thiết kế Bối cảnh : {project_info['world_setting']}")
print(f"🎨 Phong cách        : {project_info['visual_style']}")
print("==================================================")

def upload_file_to_drive_fresh(file_path, folder_id, retries=3):
    file_name = os.path.basename(file_path)
    for attempt in range(1, retries + 1):
        try:
            sync_system_time()
            scopes = ["https://www.googleapis.com/auth/drive"]
            creds = Credentials.from_service_account_info(
                SERVICE_ACCOUNT_INFO, scopes=scopes
            )
            service = build("drive", "v3", credentials=creds)
            file_metadata = {"name": file_name, "parents": [folder_id]}
            media = MediaFileUpload(file_path, mimetype="video/mp4", resumable=True)
            uploaded_file = (
                service.files()
                .create(body=file_metadata, media_body=media, fields="id")
                .execute()
            )
            file_id = uploaded_file.get("id")
            print(f"☁️ Đã đẩy thành công {file_name} lên Drive (ID: {file_id})")
            return file_id
        except Exception as e:
            print(f"⚠️ [Lần {attempt}/{retries}] Upload {file_name} thất bại: {str(e)}")
            if attempt < retries:
                time.sleep(3)
            else:
                print(f"❌ Bỏ qua upload cho {file_name} sau {retries} lần thử thất bại.")
                return None

def send_n8n_final_webhook(
    status,
    total_scenes,
    final_file=None,
    drive_file_id=None,
    error_message=None,
):
    payload = {
        "status": status,
        "total_scenes": total_scenes,
        "final_file": final_file,
        "drive_file_id": drive_file_id,
        "error_message": error_message,
        "title": project_info["title"],
        "genre": project_info["genre"],
        "character_design": project_info["character_design"],
        "world_setting": project_info["world_setting"],
        "visual_style": project_info["visual_style"],
    }
    try:
        res = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=60)
        print(f"📡 [FINAL WEBHOOK {status.upper()}] HTTP {res.status_code}")
    except Exception as e:
        print(f"❌ Lỗi gửi Webhook về n8n: {str(e)}")

# -------------------------------------------------------------------
# 3. KHỞI TẠO MODEL LTX-VIDEO
# -------------------------------------------------------------------
print("\n🧠 3. KHỞI TẠO MODEL LTX-VIDEO...")

try:
    if os.path.exists(DATASET_MODEL_PATH):
        print(f"⚡ Tìm thấy Model từ Kaggle Dataset ({DATASET_MODEL_PATH}). Đang load siêu tốc...")
        model_source = DATASET_MODEL_PATH
    elif os.path.exists(WORKING_MODEL_PATH):
        print(f"⚡ Tìm thấy Model từ thư mục local ({WORKING_MODEL_PATH}). Đang load...")
        model_source = WORKING_MODEL_PATH
    else:
        print("⏳ Chưa thấy Dataset đính kèm. Đang tải trực tiếp từ HuggingFace (Lightricks/LTX-Video)...")
        model_source = "Lightricks/LTX-Video"

    pipe = LTXPipeline.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    if model_source == "Lightricks/LTX-Video":
        print(f"💾 Đang lưu bản backup vào '{WORKING_MODEL_PATH}'...")
        pipe.save_pretrained(WORKING_MODEL_PATH)

    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    print("✅ Load Model LTX-Video thành công!")
except Exception as e:
    send_n8n_final_webhook(
        "failed", 0, error_message=f"Lỗi khởi tạo Model: {str(e)}"
    )
    sys.exit(1)

# -------------------------------------------------------------------
# 4. RENDER VÀ LƯU TỪNG CẢNH
# -------------------------------------------------------------------
rendered_files = []

for index, row in df.iterrows():
    scene_idx_val = row.get("scene_index")
    if pd.notna(scene_idx_val):
        scene_index = int(scene_idx_val)
    else:
        scene_index = index + 1

    prompt = str(row.get("prompt", "")).strip()
    negative_prompt = str(
        row.get("negative_prompt", "worst quality, low quality, blurry")
    ).strip()
    filename = f"scene_{scene_index:03d}.mp4"

    if not prompt or prompt.lower() == "nan":
        continue

    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        print(f"\n⏩ [Cảnh {scene_index}/{total_scenes}] Đã tồn tại local, tiến hành upload lại lên Drive...")
        upload_file_to_drive_fresh(filename, DRIVE_FOLDER_ID)
        rendered_files.append(filename)
        continue

    print(f"\n🎬 [{scene_index}/{total_scenes}] Đang render cảnh {scene_index}...")

    try:
        video_frames = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=640,
            height=384,
            num_frames=65,
            num_inference_steps=15,
            guidance_scale=3.0,
        ).frames[0]

        export_to_video(video_frames, filename, fps=24)
        print(f"💾 Đã lưu local: {filename}")

        upload_file_to_drive_fresh(filename, DRIVE_FOLDER_ID)
        rendered_files.append(filename)

    except Exception as e:
        print(f"❌ Lỗi render cảnh {scene_index}: {str(e)}")
    finally:
        if "video_frames" in locals():
            del video_frames
        gc.collect()
        torch.cuda.empty_cache()

# -------------------------------------------------------------------
# 5. GỘP VIDEO FULL & GỬI WEBHOOK
# -------------------------------------------------------------------
print("\n🎞️ 5. BẮT ĐẦU GỘP TẤT CẢ CẢNH THÀNH VIDEO HOÀN CHỈNH...")

if rendered_files:
    with open("file_list.txt", "w") as f:
        for file in rendered_files:
            f.write(f"file '{file}'\n")

    final_output = "final_output_full.mp4"
    concat_cmd = f"ffmpeg -f concat -safe 0 -i file_list.txt -c copy {final_output} -y"
    subprocess.run(concat_cmd, shell=True, check=True)
    print(f"🎉 GỘP VIDEO THÀNH CÔNG: {final_output}")

    drive_file_id = upload_file_to_drive_fresh(final_output, DRIVE_FOLDER_ID)

    send_n8n_final_webhook(
        status="completed_all",
        total_scenes=len(rendered_files),
        final_file=final_output,
        drive_file_id=drive_file_id,
    )
else:
    send_n8n_final_webhook(
        status="failed",
        total_scenes=0,
        error_message="Không render thành công cảnh nào.",
    )
