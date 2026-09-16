import gc
import json
import os
import subprocess
import sys
import time
import pandas as pd
import requests

print("🚀 1. CÀI ĐẶT CÁC THƯ VIỆN CẦN THIẾT VÀ ĐỒNG BỘ GIỜ HỆ THỐNG...")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# Đồng bộ giờ hệ thống Kaggle container tránh lỗi lệch JWT Timestamp của Google API
def sync_system_time():
  try:
    subprocess.run(
        "apt-get update -qq && apt-get install -y -qq ntpdate && ntpdate"
        " time.google.com",
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
# CONFIGURATION & PARSE THÔNG TIN DỰ ÁN TỪ N8N
# -------------------------------------------------------------------
N8N_WEBHOOK_URL = (
    "https://n8n-latest-namx.onrender.com/webhook-test/kaggle-video-done"
)
SHEET_ID = "1DmA-yuPwDl1riceSMGzWPXhuxrL4y987lOZ6Af351l8"
GOOGLE_SHEET_CSV_URL = (
    f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
)
DRIVE_FOLDER_ID = "1oXS7LweDNK2fYsWonQay3U-hUmEIsgCF"

project_info = {
    "title": "Chưa đặt tiêu đề",
    "genre": "Mặc định",
    "character_design": "",
    "world_setting": "",
    "visual_style": "",
}

if len(sys.argv) > 1:
  try:
    input_data = json.loads(sys.argv[1])
    project_info["title"] = input_data.get("title", project_info["title"])
    project_info["genre"] = input_data.get("genre", project_info["genre"])
    project_info["character_design"] = input_data.get(
        "character_design", project_info["character_design"]
    )
    project_info["world_setting"] = input_data.get(
        "world_setting", project_info["world_setting"]
    )
    project_info["visual_style"] = input_data.get(
        "visual_style", project_info["visual_style"]
    )
  except Exception as e:
    print(f"⚠️ Không thể đọc tham số đầu vào: {e}")

print("==================================================")
print("🎬 THÔNG TIN DỰ ÁN:")
print(f"📌 Tiêu đề           : {project_info['title']}")
print(f"🏷️ Thể loại          : {project_info['genre']}")
print(f"👤 Thiết kế Nhân vật : {project_info['character_design']}")
print(f"🏰 Thiết kế Bối cảnh : {project_info['world_setting']}")
print(f"🎨 Phong cách        : {project_info['visual_style']}")
print("==================================================")

SERVICE_ACCOUNT_INFO = {
    "type": "service_account",
    "project_id": "hieutrung",
    "private_key_id": "0b555f4f6a3d0d2e4f83bd60e1ce874b8dd01a20",
    "private_key": (
        "-----BEGIN PRIVATE"
        " KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDKEajn67fIvu2M\nzza5HddfDbHM1ziV70BKLGHtGtnY8ZQLQC0TZs9gLoyDWK5ok/z2H5Pz1we1ITDU\nOhzV76zs8jZhnzeVBFnpdfY5IMkhfJd4RSn0mGrlROm8MadBFq1XHYQ/x00drce/\nj2yqyK9cdLs73aoYJTGgtit0va7xPeHMgwOuxnCDh2W+6xDp6hczr4RSIqjSpYAP\nWE2dkUWzM3ewaAuVlAvsRGhZWAKgb5xlVrZ/isSOBabmSDh79IenTC3xSewWlV4K\nPw3DtgtYHBazj7P9fD8TjpLkbnk8kgxb0hiYKm2ZbxraDovTVjiB/UEHFlqH4GFv\nGu+NDxshAgMBAAECggEAU4wzsxiSF41huLugW6/L8cA+yHwgKFYQ1do97wQQGJPh\n6zjwqjny+kikzlXnXtP5XmY2DTbWN/zuLIGOlKIRdLK862YiXBm9dzrPwFUe9BqI\nojCupTQz1nHE1owNJGtU5lUM7jXgW6oTkc+iVYa+gtK864a+IleWimVn2E/pOlLo\nRzEI3SVRgy/6ILj2wBxeFZHSQObZe4XOW64boJZPAE5bjX+Z5siOIfoxBNuPom2+\nmVb9ijrOAuOY3AyE67G/pWhPODAs7Xv3Nt0d7Yd8r6qXYBnzY0EYzsHfHi89rjZY\njElqQ0uyzrG6OcPmcQoxWgvNzUxpPxPCTNvUkO4y5QKBgQDyfenUqH0WIcgSgUaq\nxlM8i3u2xpRPFbchOGAKko9D1ihQ8icWh4XHW7VJDj9Dj4t2cu5t4XkX74IslsBL\n7g35kmoYldX2yJX6LRyfjN49gAyRyiGo2UNFCX15DW+gaJQx1E9z7trSiSZDn/T6\nGVGJwVExBWnBIgGul839LTsHZwKBgQDVU0p08bBtz4XtqeaTFIw0dN2FK48CG8qJ\nfaONZeTPDud/Znc6xiyuJTaKyXiPBBE7TCwdCKv89sa1dw5OX0LfhwAaDrKJol/m\nSekMUnkeaUHa9BfSFfsE58Z3b6LR3Gi8ZSjPkLz/cKT25l/lEF9Mu21hKPIUqjHJ\nQRLxvlzcNwKBgQDhh9wnrkEQiYDEPToVcPlPcUcxukWLvF2jZwRkMOVQKWk7x8w0\n9vykaxYTiU2rr2D9XG2HAtKWQWsnv1nABPs4aEWG8iybJvneQYDCn8i/GE4Ydg+S\nM+eN2QK6yJVOcpWKNrVi1P7uGyLceHPm/A9K+OJjnm46cz9vO78Yvq2M9wKBgBI3\nFnh92q7FtY3hoAqXCpHAGNoyKffoH5c13y1Hsg3sG+BJA41FNrN4Afw/z8eGdWI2\n0t13zBfBip4cGGCgybkEcgHHl38hGkczsG6Y7DaojjL//Li3n8N/dvbj1WdOBrNv\nf9iZZ0n4eF2Mtkt85mZK6sANGv6gubeRkuiJdKxpAoGAMbZnGb9gTM+NYOUUG/Y2\ngH9BTiHEuXpLdGi47B/2YVzmmxI2UNs5DB56SZAiIoRRWjwq95cDtEOWPm9LUeUb\niYYQ+yuU4+6EG4w6A1jBS8RYAmO0NY4ic9syFkLv9ecmikqH2cJW1MKbhvJ2URm4\nYLz7Qq3TrM5I2qNmnZbm+28=\n-----END"
        " PRIVATE KEY-----\n"
    ),
    "client_email": "n8n-youtube@hieutrung.iam.gserviceaccount.com",
    "client_id": "102538454054650566316",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": (
        "https://www.googleapis.com/robot/v1/metadata/x509/n8n-youtube%40hieutrung.iam.gserviceaccount.com"
    ),
    "universe_domain": "googleapis.com",
}


def upload_file_to_drive_fresh(file_path, folder_id, retries=3):
  file_name = os.path.basename(file_path)

  for attempt in range(1, retries + 1):
    try:
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
      print(
          f"⚠️ [Lần {attempt}/{retries}] Upload {file_name} thất bại: {str(e)}"
      )
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
    print(f"📡 [FINAL WEBHOOK TEST {status.upper()}] HTTP {res.status_code}")
  except Exception as e:
    print(f"❌ Lỗi gửi Webhook về n8n: {str(e)}")


# -------------------------------------------------------------------
# 2. KHỞI TẠO MODEL LTX-VIDEO
# -------------------------------------------------------------------
print("🧠 2. TẢI MODEL LTX-VIDEO...")
try:
  pipe = LTXPipeline.from_pretrained(
      "Lightricks/LTX-Video",
      torch_dtype=torch.bfloat16,
      low_cpu_mem_usage=True,
  )
  pipe.enable_sequential_cpu_offload()
  pipe.vae.enable_tiling()
  pipe.vae.enable_slicing()
  print("✅ Load Model thành công!")
except Exception as e:
  send_n8n_final_webhook(
      "failed", 0, error_message=f"Lỗi khởi tạo Model: {str(e)}"
  )
  sys.exit(1)

# -------------------------------------------------------------------
# 3. ĐỌC GOOGLE SHEETS
# -------------------------------------------------------------------
try:
  df = pd.read_csv(GOOGLE_SHEET_CSV_URL)
  total_scenes = len(df)
  print(f"📊 Tìm thấy {total_scenes} cảnh từ Google Sheets.")
except Exception as e:
  send_n8n_final_webhook(
      "failed", 0, error_message=f"Lỗi đọc Google Sheet: {str(e)}"
  )
  sys.exit(1)

# -------------------------------------------------------------------
# 4. RENDER VÀ LƯU TỪNG CẢNH
# -------------------------------------------------------------------
rendered_files = []

for index, row in df.iterrows():
  scene_index = index + 1
  prompt = str(row.get("prompt", "")).strip()
  negative_prompt = str(
      row.get("negative_prompt", "worst quality, low quality, blurry")
  ).strip()
  filename = f"scene_{scene_index:03d}.mp4"

  if not prompt or prompt.lower() == "nan":
    continue

  if os.path.exists(filename) and os.path.getsize(filename) > 0:
    print(
        f"⏩ [Cảnh {scene_index}/{total_scenes}] Đã tồn tại local, bỏ qua..."
    )
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
# 5. GỘP VIDEO FULL & BẮN THÔNG TIN HOÀN CHỈNH VỀ N8N
# -------------------------------------------------------------------
print("\n🎞️ 5. BẮT ĐẦU GỘP TẤT CẢ CẢNH THÀNH VIDEO HOÀN CHỈNH...")

if rendered_files:
  with open("file_list.txt", "w") as f:
    for file in rendered_files:
      f.write(f"file '{file}'\n")

  final_output = "final_output_full.mp4"
  concat_cmd = (
      f"ffmpeg -f concat -safe 0 -i file_list.txt -c copy {final_output} -y"
  )
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
