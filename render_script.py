import asyncio
import gc
import os
import subprocess
import sys
import time
from pathlib import Path

# -------------------------------------------------------------------
# 1. CÀI ĐẶT PACKAGES & CẤU HÌNH HỆ THỐNG
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
        "requests",
        "Pillow"
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
from diffusers import FluxPipeline, LTXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from huggingface_hub import login

RUN_DATE = time.strftime("%Y%m%d_%H%M%S")
print(f"🚀 FLUX.1 + LTX-VIDEO PIPELINE (RAM SAFE EDITION) - [{RUN_DATE}]")

# Cấu hình PyTorch quản lý bộ nhớ chống phân mảnh VRAM
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
# CONFIG VÀ THÔNG TIN DỰ ÁN
# -------------------------------------------------------------------
N8N_WEBHOOK_URL = "https://n8n-latest-namx.onrender.com/webhook/kaggle-video-done"
SHEET_ID = "1DmA-yuPwDl1riceSMGzWPXhuxrL4y987lOZ6Af351l8"
GOOGLE_SHEET_CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
DRIVE_FOLDER_ID = "1oXS7LweDNK2fYsWonQay3U-hUmEIsgCF"

# 🔑 Đọc Hugging Face Token an toàn từ biến môi trường (GitHub/Kaggle Secrets)
HF_TOKEN = os.environ.get("HF_TOKEN", "")

if HF_TOKEN and HF_TOKEN.startswith("hf_"):
    try:
        login(token=HF_TOKEN)
        print("🔑 Đã xác thực thành công Hugging Face Token.")
    except Exception as e:
        print(f"⚠️ Không thể login Hugging Face: {e}")
else:
    print("⚠️ CẢNH BÁO: HF_TOKEN không tìm thấy hoặc không hợp lệ trong biến môi trường.")

VOICE_MAP = {
    "nam": "vi-VN-NamMinhNeural",
    "nu": "vi-VN-HoaiMyNeural"
}

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
# 2. ĐỌC DỮ LIỆU TỪ GOOGLE SHEETS
# -------------------------------------------------------------------
print("\n📊 2. TẢI DỮ LIỆU TỪ GOOGLE SHEETS...")
try:
    df = pd.read_csv(GOOGLE_SHEET_CSV_URL)
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
    "visual_style": ""
}

if not df.empty:
    first_row = df.iloc[0]
    for key in project_info.keys():
        if key in df.columns:
            val = str(first_row[key]).strip()
            if val.lower() not in ["nan", "[empty]", ""]:
                project_info[key] = val

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
# 3. QUY TRÌNH XỬ LÝ CHÍNH (2-PASS CHỐNG TRÀN RAM/VRAM)
# -------------------------------------------------------------------
async def process_video_pipeline():
    rendered_files = []
    image_paths = {}

    # ===============================================================
    # GIAI ĐOẠN 1: TẠO TOÀN BỘ ẢNH TĨNH BẰNG FLUX.1
    # ===============================================================
    print("\n🧠 [GIAI ĐOẠN 1] Khởi chạy FLUX.1 (Text → Image)...")
    try:
        flux_pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
            token=HF_TOKEN,
        )
        flux_pipe.enable_sequential_cpu_offload() # Tối ưu hóa RAM/VRAM cấp độ cao nhất
    except Exception as e:
        print(f"❌ Lỗi load FLUX.1: {e}")
        sys.exit(1)

    for index, row in df.iterrows():
        scene_idx_val = row.get("scene_index")
        scene_index = int(scene_idx_val) if pd.notna(scene_idx_val) else index + 1
        img_prompt = str(row.get("image_prompt", "")).strip()

        if not img_prompt or img_prompt.lower() in ["nan", "none", "null"]:
            print(f"⚠️ Bỏ qua cảnh {scene_index}: Không tìm thấy image_prompt")
            continue

        image_file = f"image_{scene_index:03d}_{RUN_DATE}.png"
        image_paths[scene_index] = image_file

        if os.path.exists(image_file):
            print(f"⏩ Ảnh cảnh {scene_index} đã tồn tại → Bỏ qua sinh ảnh")
            continue

        print(f"🖼️ [{scene_index}/{total_scenes}] Đang sinh ảnh FLUX.1...")
        try:
            gc.collect()
            torch.cuda.empty_cache()

            generator = torch.Generator(device="cpu").manual_seed(42 + scene_index)
            image = flux_pipe(
                prompt=img_prompt,
                width=1024,
                height=576,
                num_inference_steps=4,
                guidance_scale=0.0,
                generator=generator,
                max_sequence_length=256,
            ).images[0]

            image.save(image_file)
            print(f"   ✅ Đã lưu ảnh: {image_file}")
            upload_file_to_drive_fresh(image_file, DRIVE_FOLDER_ID)
        except Exception as e:
            print(f"❌ Lỗi sinh ảnh cảnh {scene_index}: {e}")

    # UNLOAD FLUX HOÀN TOÀN KHỎI BỘ NHỚ TRƯỚC KHI CHUYỂN SANG GIAI ĐOẠN 2
    print("\n🧹 Xóa FLUX.1 khỏi RAM & VRAM...")
    del flux_pipe
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)

    # ===============================================================
    # GIAI ĐOẠN 2: TẠO VIDEO BẰNG LTX-VIDEO & LẮP RÁP AUDIO
    # ===============================================================
    print("\n🧠 [GIAI ĐOẠN 2] Khởi chạy LTX-Video (Image → Video)...")
    try:
        ltx_pipe = LTXImageToVideoPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            torch_dtype=torch.bfloat16,
            token=HF_TOKEN,
        )
        ltx_pipe.enable_model_cpu_offload()
    except Exception as e:
        print(f"❌ Lỗi load LTX-Video: {e}")
        sys.exit(1)

    for index, row in df.iterrows():
        scene_idx_val = row.get("scene_index")
        scene_index = int(scene_idx_val) if pd.notna(scene_idx_val) else index + 1

        image_file = image_paths.get(scene_index)
        if not image_file or not os.path.exists(image_file):
            print(f"⚠️ Thiếu ảnh đầu vào cho cảnh {scene_index} → Bỏ qua")
            continue

        vid_prompt = str(row.get("video_prompt", "")).strip()
        neg_prompt = str(row.get("negative_prompt", "")).strip()
        dialogue_text = str(row.get("dialogue", "")).strip()
        char_name = str(row.get("character_name", "")).strip()

        if dialogue_text.lower() in ["nan", "[empty]", "none", "null"]:
            dialogue_text = ""

        # Tự động phân tách giọng Nam/Nữ
        voice_to_use = VOICE_MAP["nam"]
        if any(kw in char_name.lower() for kw in ["nữ", "chị", "muội", "cô"]):
            voice_to_use = VOICE_MAP["nu"]

        raw_video_file = f"raw_scene_{scene_index:03d}_{RUN_DATE}.mp4"
        audio_scene_file = f"audio_scene_{scene_index:03d}_{RUN_DATE}.mp3"
        final_scene_file = f"scene_{scene_index:03d}_{RUN_DATE}.mp4"

        if os.path.exists(final_scene_file) and os.path.getsize(final_scene_file) > 10000:
            print(f"⏩ Video cảnh {scene_index} đã tồn tại → Bỏ qua")
            rendered_files.append(final_scene_file)
            continue

        print(f"\n🎬 [{scene_index}/{total_scenes}] Đang tạo chuyển động LTX-Video...")
        try:
            gc.collect()
            torch.cuda.empty_cache()

            image_input = load_image(image_file).resize((1024, 576))
            scene_seed = 42 + scene_index

            motion_prompt = vid_prompt if vid_prompt and vid_prompt.lower() not in ["nan", "none"] else "smooth character movement, cinematic lighting"

            video_frames = ltx_pipe(
                image=image_input,
                prompt=motion_prompt,
                negative_prompt=neg_prompt if neg_prompt else None,
                width=768,
                height=432,
                num_frames=20,
                num_inference_steps=20,
                generator=torch.Generator(device="cuda").manual_seed(scene_seed),
            ).frames[0]

            export_to_video(video_frames, raw_video_file, fps=25)
            print(f"   ✅ Đã tạo video thô: {raw_video_file}")

            del video_frames
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Lỗi sinh video LTX cảnh {scene_index}: {e}")
            if os.path.exists(image_file): os.remove(image_file)
            continue

        # ---------- Tạo Voice & Lắp ráp FFmpeg ----------
        if dialogue_text:
            print(f"🎙️ Voice [{char_name}]: {dialogue_text}")
            communicate = edge_tts.Communicate(text=dialogue_text, voice=voice_to_use)
            await communicate.save(audio_scene_file)

            v_dur = get_media_duration(raw_video_file)
            a_dur = get_media_duration(audio_scene_file)
            pad_dur = max(0.0, a_dur - v_dur)

            if pad_dur > 0:
                mix_cmd = (
                    f'ffmpeg -y -i "{raw_video_file}" -i "{audio_scene_file}" '
                    f'-filter_complex "[0:v]tpad=stop_mode=clone:stop_duration={pad_dur:.3f}[v]" '
                    f'-map "[v]" -map 1:a:0 -c:v libx264 -pix_fmt yuv420p -r 25 '
                    f'-c:a aac -ar 44100 -ac 2 -b:a 192k "{final_scene_file}"'
                )
            else:
                mix_cmd = (
                    f'ffmpeg -y -i "{raw_video_file}" -i "{audio_scene_file}" '
                    f'-map 0:v:0 -map 1:a:0 -c:v libx264 -pix_fmt yuv420p -r 25 '
                    f'-c:a aac -ar 44100 -ac 2 -b:a 192k -shortest "{final_scene_file}"'
                )
            subprocess.run(mix_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            silent_cmd = (
                f'ffmpeg -y -i "{raw_video_file}" -f lavfi -i anullsrc=r=44100:cl=stereo '
                f'-c:v libx264 -pix_fmt yuv420p -r 25 -c:a aac -ar 44100 -ac 2 -b:a 192k '
                f'-shortest "{final_scene_file}"'
            )
            subprocess.run(silent_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Xóa các file trung gian
        for f in [image_file, raw_video_file, audio_scene_file]:
            if os.path.exists(f): 
                os.remove(f)

        upload_file_to_drive_fresh(final_scene_file, DRIVE_FOLDER_ID)
        rendered_files.append(final_scene_file)

    # UNLOAD LTX-VIDEO KHỎI BỘ NHỚ
    del ltx_pipe
    gc.collect()
    torch.cuda.empty_cache()

    # ===============================================================
    # GIAI ĐOẠN 3: GỘP TẤT CẢ CẢNH VÀ THÊM BGM
    # ===============================================================
    print("\n🎞️ Gộp tất cả các cảnh thành bộ phim hoàn chỉnh...")
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

    print(f"\n☁️ Upload phim hoàn chỉnh lên Google Drive...")
    drive_file_id = upload_file_to_drive_fresh(final_output, DRIVE_FOLDER_ID)

    send_n8n_final_webhook(
        status="completed_all",
        total_scenes=len(rendered_files),
        final_file=final_output,
        drive_file_id=drive_file_id
    )
    print("\n🎉 HOÀN TẤT PIPELINE FLUX + LTX CHỐNG TRÀN RAM!")

# -------------------------------------------------------------------
# 4. CHẠY PIPELINE
# -------------------------------------------------------------------
try:
    asyncio.run(process_video_pipeline())
except RuntimeError:
    loop = asyncio.get_event_loop()
    loop.run_until_complete(process_video_pipeline())
