import os
import sys
import gc
import subprocess
import traceback

print("🚀 1. BẮT ĐẦU CÀI ĐẶT THƯ VIỆN CẦN THIẾT...")

# Cấu hình tránh phân mảnh bộ nhớ VRAM PyTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def install_requirements():
    packages = [
        "diffusers",
        "transformers",
        "accelerate",
        "imageio-ffmpeg",
        "requests",
        "pandas",
        "torch"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)

install_requirements()

import torch
import requests
import pandas as pd
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

print("✅ Cài đặt môi trường thành công!")

# -------------------------------------------------------------------
# CONFIGURATION (CẤU HÌNH THÔNG TIN)
# -------------------------------------------------------------------
N8N_WEBHOOK_URL = "https://n8n-latest-namx.onrender.com/webhook/kaggle-video-done"

SHEET_ID = "1DmA-yuPwDl1riceSMGzWPXhuxrL4y987lOZ6Af351l8"
GOOGLE_SHEET_CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

def send_n8n_webhook(status, scene_index, prompt, file_name=None, error_message=None):
    payload = {
        "status": status,
        "scene_index": scene_index,
        "prompt": prompt,
        "file_name": file_name,
        "error_message": error_message
    }
    try:
        res = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=30)
        print(f"📡 [Webhook {status.upper()}] Bắn tin về n8n (HTTP {res.status_code})")
    except Exception as e:
        print(f"❌ Lỗi gửi Webhook về n8n: {str(e)}")

# -------------------------------------------------------------------
# 2. KHỞI TẠO MODEL LTX-VIDEO (TỐI ƯU VRAM CHO KAGGLE GPU T4)
# -------------------------------------------------------------------
print("🧠 2. ĐANG TẢI MODEL LTX-VIDEO (BFLOAT16 & CPU OFFLOAD)...")

try:
    # 1. Chuyển sang bfloat16 để tiết kiệm VRAM hơn float16
    pipe = LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video", 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    
    # 2. Bật Offload tự động (Không dùng .to("cuda") để tránh tràn VRAM)
    pipe.enable_model_cpu_offload()
    
    # 3. Tối ưu VAE decode theo mảng nhỏ
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    
    print("✅ Đã load Model LTX-Video thành công!")
except Exception as e:
    err_str = f"Lỗi khởi tạo Model LTX-Video: {str(e)}"
    print(f"❌ {err_str}")
    send_n8n_webhook(
        status="failed",
        scene_index=0,
        prompt="Model Initialization",
        error_message=err_str
    )
    sys.exit(1)

# -------------------------------------------------------------------
# 3. LẤY DỮ LIỆU TỪ GOOGLE SHEET
# -------------------------------------------------------------------
print("📊 3. ĐANG ĐỌC DỮ LIỆU TỪ GOOGLE SHEETS...")

try:
    df = pd.read_csv(GOOGLE_SHEET_CSV_URL)
    print(f"✅ Đã tải thành công danh sách gồm {len(df)} cảnh từ Google Sheets!")
except Exception as e:
    err_str = f"Không đọc được Google Sheets CSV: {str(e)}"
    print(f"❌ {err_str}")
    send_n8n_webhook(
        status="failed",
        scene_index=0,
        prompt="Fetch Google Sheets",
        error_message=err_str
    )
    sys.exit(1)

# -------------------------------------------------------------------
# 4. LẶP QUA TỪNG CẢNH ĐỂ RENDER & THÔNG BÁO VỀ N8N
# -------------------------------------------------------------------
print("⚙️ 4. BẮT ĐẦU VÒNG LẶP RENDER VIDEO...")

for index, row in df.iterrows():
    scene_index = row.get("scene_index", index + 1)
    prompt = str(row.get("prompt", ""))
    negative_prompt = str(row.get("negative_prompt", "worst quality, low quality, blurry, distorted"))
    output_filename = f"scene_{scene_index:03d}.mp4"

    if not prompt or prompt == "nan":
        print(f"⚠️ Cảnh {scene_index} bỏ qua do Prompt rỗng.")
        continue

    print(f"\n🎬 [Cảnh {scene_index}/{len(df)}] Đang render...")
    print(f"   Prompt: {prompt[:80]}...")

    try:
        # Render video cấu hình chuẩn GPU T4: 704x480, 121 frames (~5s)
        video_frames = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=704,
            height=480,
            num_frames=121,         # Mức frames an toàn tránh OOM
            num_inference_steps=30, # Số bước sinh ảnh tối ưu
            guidance_scale=3.0
        ).frames[0]

        export_to_video(video_frames, output_filename, fps=24)
        print(f"🎉 Render hoàn tất: {output_filename}")

        send_n8n_webhook(
            status="success",
            scene_index=scene_index,
            prompt=prompt,
            file_name=output_filename
        )

    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"❌ [LỖI CẢNH {scene_index}] {str(e)}")

        send_n8n_webhook(
            status="failed",
            scene_index=scene_index,
            prompt=prompt,
            error_message=str(e)
        )

    # Giải phóng hoàn toàn cache VRAM PyTorch sau từng cảnh
    gc.collect()
    torch.cuda.empty_cache()

print("\n🏁 TẤT CẢ CÁC CẢNH ĐÃ ĐƯỢC XỬ LÝ XONG!")
