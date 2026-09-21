import asyncio
import gc
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


# -------------------------------------------------------------------
# 1. CÀI ĐẶT PACKAGE (ĐÃ TỐI ƯU DEPENDENCY - KHÔNG ÉP NUMPY CŨ)
# -------------------------------------------------------------------
def install_requirements():
    packages = [
        "nest_asyncio",
        "diffusers>=0.31.0",
        "transformers",
        "imageio-ffmpeg",
        "google-api-python-client",
        "google-auth-oauthlib",
        "huggingface_hub",
        "soundfile",
        "av",
        "edge-tts",
        "accelerate",
        "protobuf<6.0.0,>=3.20.2",
    ]
    print("📦 Đang kiểm tra và đồng bộ Packages...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-warn-script-location",
            "--disable-pip-version-check",
        ]
        + packages
    )


install_requirements()

import edge_tts
import nest_asyncio

nest_asyncio.apply()

from diffusers import LTXPipeline
from diffusers.utils import export_to_video
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import numpy as np
import pandas as pd
import requests
import torch

RUN_DATE = time.strftime("%Y%m%d_%H%M%S")
print(f"🚀 KHỞI TẠO PIPELINE LTX-VIDEO TỐI ƯU VRAM - PHIÊN RUN [{RUN_DATE}]")

# Tối ưu phân bổ bộ nhớ PyTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True,max_split_size_mb:128"
)


def sync_system_time():
    try:
        subprocess.run(
            "apt-get update -qq && apt-get install -y -qq ntpdate && ntpdate"
            " time.google.com",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


sync_system_time()

# -------------------------------------------------------------------
# CONFIG & AUTH
# -------------------------------------------------------------------
N8N_WEBHOOK_URL = (
    "https://n8n-latest-namx.onrender.com/webhook/kaggle-video-done"
)
SHEET_ID = "1DmA-yuPwDl1riceSMGzWPXhuxrL4y987lOZ6Af351l8"
GOOGLE_SHEET_CSV_URL = (
    f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
)
DRIVE_FOLDER_ID = "1oXS7LweDNK2fYsWonQay3U-hUmEIsgCF"

VOICE_MAP = {"nam": "vi-VN-NamMinhNeural", "nu": "vi-VN-HoaiMyNeural"}

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


def get_media_duration(file_path):
    cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{file_path}"'
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, check=True
        )
        return float(result.stdout.strip())
    except Exception:
        return 2.0


# -------------------------------------------------------------------
# 2. ĐỌC VÀ XỬ LÝ DỮ LIỆU SHEETS
# -------------------------------------------------------------------
print("\n📊 2. TẢI DỮ LIỆU TỪ GOOGLE SHEETS...")
try:
    df = pd.read_csv(GOOGLE_SHEET_CSV_URL)
    total_scenes = len(df)
    print(f"✅ Tìm thấy {total_scenes} cảnh trong Google Sheet.")
except Exception as e:
    print(f"❌ Lỗi đọc Google Sheet: {str(e)}")
    sys.exit(1)

project_info = {
    "title": "Chưa đặt tiêu đề",
    "genre": "Mặc định",
    "character_design": "",
    "world_setting": "",
    "visual_style": "",
    "gender": "nam",
}

if not df.empty:
    first_row = df.iloc[0]
    for key in project_info.keys():
        if key in df.columns:
            val = str(first_row[key]).strip()
            if val.lower() not in ["nan", "[empty]", ""] :
                project_info[key] = val

SELECTED_GENDER = project_info.get("gender", "nam").lower()
ACTIVE_VOICE = VOICE_MAP.get(SELECTED_GENDER, "vi-VN-NamMinhNeural")


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
            print(f"☁️ Đã tải lên Google Drive thành công: {file_name}")
            return uploaded_file.get("id")
        except Exception as e:
            print(f"⚠️ [Lần {attempt}/{retries}] Upload Drive thất bại: {str(e)}")
            if attempt < retries:
                time.sleep(3)
            else:
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
        **project_info,
    }
    try:
        res = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=60)
        print(f"📡 [FINAL WEBHOOK {status.upper()}] HTTP {res.status_code}")
    except Exception as e:
        print(f"❌ Lỗi gửi Webhook: {str(e)}")


# -------------------------------------------------------------------
# 3. LOAD MODEL VỚI CÁC TỐI ƯU VRAM NGHIÊM NGẶT
# -------------------------------------------------------------------
print("\n🧠 3. KHỞI TẠO MODEL LTX-VIDEO (TỐI ƯU BỘ NHỚ)...")
try:
    MODEL_ID = "Lightricks/LTX-Video"
    pipe = LTXPipeline.from_pretrained(MODEL_ID, dtype=torch.bfloat16)

    # Offload từng layer sang CPU & Tiling VAE để chống OOM VRAM
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    print("✅ Load Model LTX-Video & Bật Offload VRAM thành công!")
except Exception as e:
    send_n8n_final_webhook(
        "failed", 0, error_message=f"Lỗi load model: {str(e)}"
    )
    sys.exit(1)


# -------------------------------------------------------------------
# 4. PROCESS PIPELINE
# -------------------------------------------------------------------
async def process_video_pipeline():
    rendered_files = []

    STYLE_3D_PREFIX = (
        "3d chinese donghua animation style, unreal engine 5 render, extremely"
        " detailed 3d face,"
    )
    DEFAULT_NEGATIVE = (
        "2d, flat drawing, realistic human, photorealistic, blurry, low quality,"
        " distorted face, morphing, text, watermark, stiff pose, deformed hands,"
        " missing fingers, extra limbs"
    )

    for index, row in df.iterrows():
        scene_idx_val = row.get("scene_index")
        scene_index = int(scene_idx_val) if pd.notna(scene_idx_val) else index + 1

        raw_prompt = str(row.get("prompt", "")).strip()
        raw_negative = str(row.get("negative_prompt", "")).strip()

        # Bổ sung bộ lọc thoại tránh nhận chuỗi rác [empty], nan, null
        scene_text = ""
        for col in ["dialogue", "scene_narration_vi", "narration_vi", "narration"]:
            if col in row and pd.notna(row[col]):
                val = str(row[col]).strip()
                if val and val.lower() not in ["nan", "[empty]", "none", "null"]:
                    scene_text = val
                    break

        raw_video_file = f"raw_scene_{scene_index:03d}_{RUN_DATE}.mp4"
        audio_scene_file = f"audio_scene_{scene_index:03d}_{RUN_DATE}.mp3"
        final_scene_file = f"scene_{scene_index:03d}_{RUN_DATE}.mp4"

        if not raw_prompt or raw_prompt.lower() in ["nan", "[empty]", "none", "null"]:
            print(f"⚠️ Bỏ qua cảnh {scene_index} do không có Prompt.")
            continue

        if (
            os.path.exists(final_scene_file)
            and os.path.getsize(final_scene_file) > 10000
        ):
            print(f"⏩ [{scene_index}/{total_scenes}] Đã tồn tại file -> bỏ qua")
            rendered_files.append(final_scene_file)
            continue

        # 4.1 Sinh Video LTX
        final_prompt = f"{STYLE_3D_PREFIX} {raw_prompt}"
        final_negative = (
            raw_negative
            if (raw_negative and raw_negative.lower() not in ["nan", "[empty]", "none"])
            else DEFAULT_NEGATIVE
        )

        scene_seed = 42 + scene_index
        generator = torch.Generator(device="cpu").manual_seed(scene_seed)

        print(f"\n🎬 [{scene_index}/{total_scenes}] Render Video LTX...")
        try:
            gc.collect()
            torch.cuda.empty_cache()

            with torch.inference_mode():
                output = pipe(
                    prompt=final_prompt,
                    negative_prompt=final_negative,
                    width=704,
                    height=384,
                    num_frames=49,
                    frame_rate=24.0,
                    num_inference_steps=20,
                    guidance_scale=3.0,
                    generator=generator,
                    output_type="pt",
                )
                video_tensor = output.frames[0]

            if video_tensor.min() < 0:
                video_tensor = (video_tensor + 1.0) / 2.0
            video_tensor = torch.clamp(video_tensor, 0.0, 1.0)

            if video_tensor.ndim == 4 and video_tensor.shape[1] == 3:
                video_tensor = video_tensor.permute(0, 2, 3, 1)
            elif video_tensor.ndim == 4 and video_tensor.shape[0] == 3:
                video_tensor = video_tensor.permute(1, 2, 3, 0)

            video_frames = (video_tensor * 255.0).cpu().numpy().astype(np.uint8)
            export_to_video(video_frames, raw_video_file, fps=24)

            del output, video_tensor, video_frames
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Lỗi render video cảnh {scene_index}: {str(e)}")
            continue

        # 4.2 Xử lý Voice Thuyết Minh & Trộn Âm thanh
        if scene_text:
            character_name = str(row.get("character_name", "")).strip()
            char_prefix = (
                f"[{character_name}]: "
                if character_name and character_name.lower() not in ["nan", "[empty]", "none"]
                else ""
            )
            print(f"🎙️ Tạo voice cảnh {scene_index} {char_prefix}'{scene_text}'")

            communicate = edge_tts.Communicate(text=scene_text, voice=ACTIVE_VOICE)
            await communicate.save(audio_scene_file)

            v_dur = get_media_duration(raw_video_file)
            a_dur = get_media_duration(audio_scene_file)
            pad_dur = max(0.0, a_dur - v_dur)

            if pad_dur > 0:
                mix_scene_cmd = (
                    f'ffmpeg -y -i "{raw_video_file}" -i "{audio_scene_file}" '
                    f'-filter_complex "[0:v]tpad=stop_mode=clone:stop_duration={pad_dur:.3f}[v]" '
                    f'-map "[v]" -map 1:a:0 -c:v libx264 -pix_fmt yuv420p -r 24 -c:a aac'
                    f' -ar 44100 -ac 2 -b:a 192k "{final_scene_file}"'
                )
            else:
                mix_scene_cmd = (
                    f'ffmpeg -y -i "{raw_video_file}" -i "{audio_scene_file}" '
                    f'-map 0:v:0 -map 1:a:0 -c:v libx264 -pix_fmt yuv420p -r 24 -c:a aac'
                    f' -ar 44100 -ac 2 -b:a 192k -shortest "{final_scene_file}"'
                )

            subprocess.run(
                mix_scene_cmd,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            # Tạo audio câm (Silent Audio) cho cảnh không có lời thoại
            silent_cmd = (
                f'ffmpeg -y -i "{raw_video_file}" -f lavfi -i'
                " anullsrc=r=44100:cl=stereo -c:v libx264 -pix_fmt yuv420p -r 24 -c:a"
                f' aac -ar 44100 -ac 2 -b:a 192k -shortest "{final_scene_file}"'
            )
            subprocess.run(
                silent_cmd,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        # Dọn dẹp file tạm
        if os.path.exists(raw_video_file):
            os.remove(raw_video_file)
        if os.path.exists(audio_scene_file):
            os.remove(audio_scene_file)

        print(f"💾 Hoàn tất Cảnh {scene_index}: {final_scene_file}")
        upload_file_to_drive_fresh(final_scene_file, DRIVE_FOLDER_ID)
        rendered_files.append(final_scene_file)

    # -------------------------------------------------------------------
    # 5. GỘP TOÀN BỘ CẢNH THÀNH PHIM HOÀN CHỈNH
    # -------------------------------------------------------------------
    print("\n🎞️ 5. TIẾN HÀNH GỘP TOÀN BỘ CÁC CẢNH...")
    if not rendered_files:
        send_n8n_final_webhook(
            "failed", 0, error_message="Không render được cảnh nào."
        )
        sys.exit(1)

    with open("file_list.txt", "w") as f:
        for file in rendered_files:
            f.write(f"file '{file}'\n")

    concat_output = f"final_concat_{RUN_DATE}.mp4"
    final_output = f"final_movie_{RUN_DATE}.mp4"

    subprocess.run(
        f"ffmpeg -y -f concat -safe 0 -i file_list.txt -c copy {concat_output}",
        shell=True,
        check=True,
    )

    # -------------------------------------------------------------------
    # 6. THÊM NHẠC NỀN (BGM)
    # -------------------------------------------------------------------
    bgm_file = "bgm_xianxia.mp3"
    if os.path.exists(bgm_file):
        print("🎵 Đang hòa âm Nhạc nền BGM cho phim...")
        bgm_cmd = (
            f'ffmpeg -y -i {concat_output} -stream_loop -1 -i {bgm_file}'
            ' -filter_complex "[0:a]volume=1.2[v_tts];[1:a]volume=0.15[v_bgm];[v_tts][v_bgm]amix=inputs=2:duration=first[a]"'
            ' -map 0:v:0 -map "[a]" -c:v copy -c:a aac -ar 44100 -ac 2 -b:a 192k'
            f" {final_output}"
        )
        try:
            subprocess.run(bgm_cmd, shell=True, check=True)
        except Exception:
            final_output = concat_output
    else:
        final_output = concat_output

    # -------------------------------------------------------------------
    # 7. UPLOAD PHIM HOÀN CHỈNH & WEBHOOK THÔNG BÁO
    # -------------------------------------------------------------------
    print(f"\n☁️ 7. ĐANG TẢI PHIM HOÀN CHỈNH {final_output} LÊN GOOGLE DRIVE...")
    drive_file_id = upload_file_to_drive_fresh(final_output, DRIVE_FOLDER_ID)

    send_n8n_final_webhook(
        status="completed_all",
        total_scenes=len(rendered_files),
        final_file=final_output,
        drive_file_id=drive_file_id,
    )

    print("\n🎉 HOÀN TẤT TOÀN BỘ TIẾN TRÌNH RENDER PHIM 3D DONGHUA!")


# Thực thi Async
try:
    asyncio.run(process_video_pipeline())
except RuntimeError:
    loop = asyncio.get_event_loop()
    loop.run_until_complete(process_video_pipeline())
