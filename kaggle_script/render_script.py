import os
import sys
import gc
import json
import subprocess
import traceback
import requests
import pandas as pd

print("🚀 1. CÀI ĐẶT CÁC THƯ VIỆN CẦN THIẾT...")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
        "google-auth"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)

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

# Folder ID Google Drive của anh
DRIVE_FOLDER_ID = "1oXS7LweDNK2fYsWonQay3U-hUmEIsgCF"

# Lấy chuỗi JSON Service Account từ biến môi trường (hoặc dán trực tiếp chuỗi JSON vào đây)
SERVICE_ACCOUNT_JSON_STR = os.environ.get("GDRIVE_SERVICE_ACCOUNT_JSON", "")

def get_drive_service():
    if not SERVICE_ACCOUNT_JSON_STR:
        print("⚠️ Chưa tìm thấy cấu hình Service Account JSON.")
        return None
    scopes = ['https://www.googleapis.com/auth/drive']
    info = json.loads(SERVICE_ACCOUNT_JSON_STR)
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    return build('drive', 'v3', credentials=creds)

def upload_file_to_drive(service, file_path, folder_id):
    if not service:
        return None
    file_name = os.path.basename(file_path)
    file_metadata = {
        'name': file_name,
        'parents': [folder_id]
    }
    media = MediaFileUpload(file_path, mimetype='video/mp4', resumable=True)
    uploaded_file = service.files().create(
        body=file_metadata, media_body=media, fields='id'
    ).execute()
    print(f"☁️ Đã đẩy {file_name} lên Drive (ID: {uploaded_file.get('id')})")
    return uploaded_file.get('id')

def send_n8n_final_webhook(status, total_scenes, final_file=None, error_message=None):
    payload = {
        "status": status,
        "total_scenes": total_scenes,
        "final_file": final_file,
        "error_message": error_message
    }
    try:
        res = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=60)
        print(f"📡 [FINAL WEBHOOK {status.upper()}] HTTP {res.status_code}")
    except Exception as e:
        print(f"❌ Lỗi gửi Webhook về n8n: {str(e)}")

# -------------------------------------------------------------------
# 2. KHỞI TẠO MODEL LTX-VIDEO
# -------------------------------------------------------------------
print("🧠 2. TẢI MODEL LTX-VIDEO (TỐI ƯU SIÊU NHẸ)...")
try:
    pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    print("✅ Load Model thành công!")
except Exception as e:
    send_n8n_final_webhook("failed", 0, error_message=f"Lỗi khởi tạo Model: {str(e)}")
    sys.exit(1)

# -------------------------------------------------------------------
# 3. ĐỌC GOOGLE SHEETS & DRIVE API
# -------------------------------------------------------------------
try:
    df = pd.read_csv(GOOGLE_SHEET_CSV_URL)
    total_scenes = len(df)
    print(f"📊 Tìm thấy {total_scenes} cảnh từ Google Sheets.")
except Exception as e:
    send_n8n_final_webhook("failed", 0, error_message=f"Lỗi đọc Google Sheet: {str(e)}")
    sys.exit(1)

drive_service = None
try:
    drive_service = get_drive_service()
except Exception as e:
    print(f"⚠️ Không thể kết nối Google Drive API: {str(e)}")

# -------------------------------------------------------------------
# 4. RENDER, LƯU TỪNG CẢNH LÊN DRIVE & GIẢI PHÓNG BỘ NHỚ
# -------------------------------------------------------------------
rendered_files = []

for index, row in df.iterrows():
    scene_index = index + 1
    prompt = str(row.get("prompt", ""))
    negative_prompt = str(row.get("negative_prompt", "worst quality, low quality, blurry"))
    filename = f"scene_{scene_index:03d}.mp4"

    if not prompt or prompt == "nan":
        continue

    # Nếu file đã được render và upload từ trước thì dùng lại (Resume)
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        print(f"⏩ [Cảnh {scene_index}/{total_scenes}] Đã tồn tại local, bỏ qua...")
        rendered_files.append(filename)
        continue

    print(f"\n🎬 [{scene_index}/{total_scenes}] Đang render...")

    try:
        video_frames = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=640,
            height=384,
            num_frames=65,
            num_inference_steps=15,
            guidance_scale=3.0
        ).frames[0]

        export_to_video(video_frames, filename, fps=24)
        print(f"💾 Đã lưu local: {filename}")

        # Upload ngay lên thư mục Google Drive
        if drive_service:
            try:
                upload_file_to_drive(drive_service, filename, DRIVE_FOLDER_ID)
            except Exception as drive_err:
                print(f"⚠️ Lỗi upload Drive: {str(drive_err)}")

        rendered_files.append(filename)

    except Exception as e:
        print(f"❌ Lỗi render cảnh {scene_index}: {str(e)}")

    # Giải phóng VRAM/RAM triệt để sau mỗi cảnh
    gc.collect()
    torch.cuda.empty_cache()

# -------------------------------------------------------------------
# 5. GỘP CÁC CẢNH THÀNH 1 VIDEO FULL & BẮN WEBHOOK VỀ N8N
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

    if drive_service:
        try:
            upload_file_to_drive(drive_service, final_output, DRIVE_FOLDER_ID)
        except Exception as e:
            print(f"⚠️ Lỗi upload video Full lên Drive: {str(e)}")

    # BẮN WEBHOOK VỀ N8N THÔNG BÁO HOÀN THÀNH 1 LẦN DUY NHẤT
    send_n8n_final_webhook(
        status="completed_all",
        total_scenes=len(rendered_files),
        final_file=final_output
    )
else:
    send_n8n_final_webhook(
        status="failed",
        total_scenes=0,
        error_message="Không render thành công cảnh nào."
    )
