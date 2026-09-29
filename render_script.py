# ============================================================
# PIPELINE KAGGLE - TẠO ẢNH NHÂN VẬT → ẢNH CẢNH → VIDEO
# Đọc Characters + Scenes từ Google Sheet
# 1. Tạo ảnh nhân vật (reference)
# 2. Tạo ảnh cảnh (dùng ảnh nhân vật làm reference)
# 3. LTX → TTS → SFX → BGM → Gộp video
# ============================================================
import asyncio
import gc
import os
import subprocess
import sys
import time
from pathlib import Path

# -------------------------------------------------------------------
# 1. CÀI PACKAGE THÔNG MINH
# -------------------------------------------------------------------
def install_requirements():
    required = {
        "nest_asyncio": "nest_asyncio",
        "diffusers": "diffusers",
        "transformers": "transformers",
        "accelerate": "accelerate",
        "imageio": "imageio",
        "googleapiclient": "google-api-python-client",
        "google_auth_oauthlib": "google-auth-oauthlib",
        "huggingface_hub": "huggingface_hub",
        "soundfile": "soundfile",
        "scipy": "scipy",
        "av": "av",
        "edge_tts": "edge-tts",
        "sentencepiece": "sentencepiece",
        "ftfy": "ftfy",
        "safetensors": "safetensors",
        "omegaconf": "omegaconf",
        "einops": "einops",
        "cv2": "opencv-python",
        "pandas": "pandas",
        "requests": "requests",
        "PIL": "Pillow",
    }
    missing = []
    for import_name, pip_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)
    if not missing:
        print("✅ Tất cả package đã có sẵn → bỏ qua cài đặt")
        return
    print(f"📦 Đang cài {len(missing)} package còn thiếu...")
    packages = missing + ["protobuf<6.0.0,>=3.20.2", "imageio-ffmpeg"]
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "--no-warn-script-location", "--disable-pip-version-check",
    ] + packages)
    print("✅ Cài đặt xong")

install_requirements()

import nest_asyncio
nest_asyncio.apply()

import torch
import pandas as pd
import requests
import scipy.io.wavfile as wavfile
import edge_tts
from PIL import Image
from diffusers import (
    LTXImageToVideoPipeline,
    AudioLDMPipeline,
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
)
from diffusers.utils import export_to_video, load_image
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from huggingface_hub import login

RUN_DATE = time.strftime("%Y%m%d_%H%M%S")
print(f"🚀 LTX + TTS + BGM PIPELINE (có tạo ảnh nhân vật + cảnh) - [{RUN_DATE}]")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------------------
# 2. CONFIG
# -------------------------------------------------------------------
N8N_WEBHOOK_URL = "https://n8n-latest-namx.onrender.com/webhook/kaggle-video-done"
SHEET_ID = "1DmA-yuPwDl1riceSMGzWPXhuxrL4y987lOZ6Af351l8"
DRIVE_FOLDER_ID = "1oXS7LweDNK2fYsWonQay3U-hUmEIsgCF"

# Tên sheet trong Google Spreadsheet
CHARACTERS_SHEET = "Characters"
SCENES_SHEET = "Scenes"

TOKEN_PART1 = os.environ.get("HF_TOKEN_PART1", "")
TOKEN_PART2 = os.environ.get("HF_TOKEN_PART2", "")
COMBINED_HF_TOKEN = f"{TOKEN_PART1.strip()}{TOKEN_PART2.strip()}".strip()
hf_token_to_pass = None
if COMBINED_HF_TOKEN.startswith("hf_"):
    try:
        login(token=COMBINED_HF_TOKEN)
        hf_token_to_pass = COMBINED_HF_TOKEN
        print("🔑 HF login OK")
    except Exception as e:
        print(f"⚠️ HF login fail: {e}")

OAUTH_CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID", "")
OAUTH_CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET", "")
OAUTH_REFRESH_TOKEN = os.environ.get("OAUTH_REFRESH_TOKEN", "")
TOKEN_URI = "https://oauth2.googleapis.com/token"

VOICE_MAP = {
    "nam": "vi-VN-NamMinhNeural",
    "nu": "vi-VN-HoaiMyNeural",
}
BGM_FILE = f"bgm_generated_{RUN_DATE}.mp3"

# Model tạo ảnh (nhẹ + nhanh trên Kaggle)
IMAGE_MODEL_ID = "black-forest-labs/FLUX.1-schnell"   # hoặc "stabilityai/sdxl-turbo"

# -------------------------------------------------------------------
# 3. HELPERS
# -------------------------------------------------------------------
def get_oauth_credentials():
    return Credentials(
        token=None,
        refresh_token=OAUTH_REFRESH_TOKEN,
        token_uri=TOKEN_URI,
        client_id=OAUTH_CLIENT_ID,
        client_secret=OAUTH_CLIENT_SECRET,
        scopes=[
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/spreadsheets.readonly",
        ],
    )

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

def get_media_duration(file_path):
    cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{file_path}"'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        return 2.0

def list_files_in_folder(folder_id):
    creds = get_oauth_credentials()
    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    files = []
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id, name, mimeType)",
            pageSize=200,
            pageToken=page_token,
        ).execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files

def download_drive_file(file_id, save_path):
    creds = get_oauth_credentials()
    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    request = service.files().get_media(fileId=file_id)
    with open(save_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
    return save_path

def upload_file_to_drive(file_path, folder_id, retries=3):
    if not os.path.exists(file_path) or not folder_id:
        return None
    file_name = os.path.basename(file_path)
    mimetype = "image/png" if file_name.endswith((".png", ".jpg", ".jpeg", ".webp")) else "video/mp4"
    if file_name.endswith(".mp3"):
        mimetype = "audio/mpeg"
    for attempt in range(1, retries + 1):
        try:
            creds = get_oauth_credentials()
            service = build("drive", "v3", credentials=creds, cache_discovery=False)
            meta = {"name": file_name, "parents": [folder_id]}
            media = MediaFileUpload(file_path, mimetype=mimetype, resumable=True)
            uploaded = service.files().create(body=meta, media_body=media, fields="id").execute()
            print(f"☁️ Upload OK: {file_name}")
            return uploaded.get("id")
        except Exception as e:
            print(f"⚠️ Upload lần {attempt} lỗi: {e}")
            time.sleep(2)
    return None

def send_n8n_webhook(status, total_scenes=0, final_file=None, drive_file_id=None, error_message=None, extra=None):
    payload = {
        "status": status,
        "total_scenes": total_scenes,
        "final_file": final_file,
        "drive_file_id": drive_file_id,
        "error_message": error_message,
        "run_date": RUN_DATE,
    }
    if extra:
        payload.update(extra)
    try:
        res = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=60)
        print(f"📡 Webhook [{status}] → {res.status_code}")
    except Exception as e:
        print(f"❌ Webhook lỗi: {e}")

def find_file_in_drive(drive_files, keywords, extensions=(".png", ".jpg", ".jpeg", ".webp")):
    keywords = [k.lower() for k in keywords]
    for f in drive_files:
        name = f["name"].lower()
        if any(k in name for k in keywords) and name.endswith(extensions):
            return f
    return None

def pick_voice(char_name: str) -> str:
    name = (char_name or "").lower()
    if any(k in name for k in ["nữ", "cô", "chị", "muội", "my", "linh", "sera", "eva"]):
        return VOICE_MAP["nu"]
    return VOICE_MAP["nam"]

def safe_get(row, *keys, default=""):
    for key in keys:
        val = row.get(key)
        if pd.notna(val) and str(val).strip().lower() not in ["nan", "none", "null", "", "[empty]"]:
            return str(val).strip()
    return default

def get_sheet_as_df(sheet_name: str) -> pd.DataFrame:
    """Đọc một sheet cụ thể từ Google Spreadsheet"""
    try:
        creds = get_oauth_credentials()
        service = build("sheets", "v4", credentials=creds, cache_discovery=False)
        result = service.spreadsheets().values().get(
            spreadsheetId=SHEET_ID,
            range=f"{sheet_name}!A:Z"
        ).execute()
        values = result.get("values", [])
        if not values:
            print(f"⚠️ Sheet '{sheet_name}' trống")
            return pd.DataFrame()
        header = [str(h).strip().lower() for h in values[0]]
        data = values[1:]
        # Đồng bộ số cột
        max_cols = len(header)
        data = [row + [""] * (max_cols - len(row)) for row in data]
        df = pd.DataFrame(data, columns=header)
        print(f"✅ Đọc sheet '{sheet_name}': {len(df)} dòng | Cột: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"❌ Lỗi đọc sheet '{sheet_name}': {e}")
        return pd.DataFrame()

# -------------------------------------------------------------------
# 3.1 AI MUSIC & SFX
# -------------------------------------------------------------------
def generate_ai_bgm(prompt_text: str, duration_sec: int, output_path: str):
    print(f"🎵 [AI BGM] Đang tạo: '{prompt_text}'...")
    try:
        clear_memory()
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(DEVICE)
        inputs = processor(text=[prompt_text], padding=True, return_tensors="pt").to(DEVICE)
        max_tokens = min(int(duration_sec * 50), 1500)
        audio = model.generate(**inputs, max_new_tokens=max_tokens)
        sr = model.config.audio_encoder.sampling_rate
        wav_path = output_path.replace(".mp3", ".wav")
        wavfile.write(wav_path, rate=sr, data=audio[0, 0].cpu().numpy())
        subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-acodec", "libmp3lame", output_path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(wav_path):
            os.remove(wav_path)
        del processor, model
        clear_memory()
        print(f"✅ [AI BGM] Xong")
        return True
    except Exception as e:
        print(f"⚠️ [AI BGM] Lỗi: {e}")
        return False

def generate_ai_sfx(prompt_text: str, duration_sec: float, output_path: str):
    print(f"🔊 [AI SFX] {prompt_text}...")
    try:
        clear_memory()
        pipe = AudioLDMPipeline.from_pretrained(
            "cvssp/audioldm-m-full",
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        ).to(DEVICE)
        audio = pipe(prompt_text, num_inference_steps=15,
                     audio_length_in_s=max(1.0, min(duration_sec, 8.0))).audios[0]
        wavfile.write(output_path, rate=16000, data=audio)
        del pipe
        clear_memory()
        print("✅ [AI SFX] Xong")
        return True
    except Exception as e:
        print(f"⚠️ [AI SFX] Lỗi: {e}")
        return False

# -------------------------------------------------------------------
# 3.2 TẠO ẢNH (NHÂN VẬT + CẢNH)
# -------------------------------------------------------------------
def load_image_pipeline(task="text2img"):
    """Load pipeline tạo ảnh, tự offload để tiết kiệm VRAM"""
    clear_memory()
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    if task == "text2img":
        pipe = AutoPipelineForText2Image.from_pretrained(
            IMAGE_MODEL_ID,
            torch_dtype=dtype,
            token=hf_token_to_pass,
        )
    else:
        pipe = AutoPipelineForImage2Image.from_pretrained(
            IMAGE_MODEL_ID,
            torch_dtype=dtype,
            token=hf_token_to_pass,
        )
    if DEVICE == "cuda":
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pipe = pipe.to(DEVICE)
    print(f"✅ Loaded image pipeline ({task})")
    return pipe

def generate_character_image(prompt: str, output_path: str, width=1280, height=720):
    """Tạo ảnh nhân vật (4-view hoặc full body)"""
    print(f"🎨 [Character] {os.path.basename(output_path)}")
    try:
        pipe = load_image_pipeline("text2img")
        image = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=4 if "schnell" in IMAGE_MODEL_ID else 20,
            guidance_scale=0.0 if "schnell" in IMAGE_MODEL_ID else 7.5,
            generator=torch.Generator("cpu").manual_seed(42),
        ).images[0]
        image.save(output_path)
        del pipe
        clear_memory()
        print(f"   ✅ Đã tạo: {output_path}")
        return True
    except Exception as e:
        print(f"   ❌ Lỗi tạo ảnh nhân vật: {e}")
        clear_memory()
        return False

def generate_scene_image(prompt: str, output_path: str, ref_image_path: str = None, width=1280, height=720):
    """Tạo ảnh cảnh. Nếu có ref_image_path (1 nhân vật) thì dùng img2img để giữ consistency"""
    print(f"🖼️  [Scene] {os.path.basename(output_path)}")
    try:
        if ref_image_path and os.path.exists(ref_image_path):
            # img2img với ảnh nhân vật làm reference
            pipe = load_image_pipeline("img2img")
            init_image = load_image(ref_image_path).resize((width, height))
            image = pipe(
                prompt=prompt,
                image=init_image,
                strength=0.65,          # giữ khuôn mặt + trang phục
                num_inference_steps=4 if "schnell" in IMAGE_MODEL_ID else 20,
                guidance_scale=0.0 if "schnell" in IMAGE_MODEL_ID else 7.5,
                generator=torch.Generator("cpu").manual_seed(42),
            ).images[0]
        else:
            # text2img thuần
            pipe = load_image_pipeline("text2img")
            image = pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=4 if "schnell" in IMAGE_MODEL_ID else 20,
                guidance_scale=0.0 if "schnell" in IMAGE_MODEL_ID else 7.5,
                generator=torch.Generator("cpu").manual_seed(42),
            ).images[0]
        image.save(output_path)
        del pipe
        clear_memory()
        print(f"   ✅ Đã tạo: {output_path}")
        return True
    except Exception as e:
        print(f"   ❌ Lỗi tạo ảnh cảnh: {e}")
        clear_memory()
        return False

# -------------------------------------------------------------------
# 4. PIPELINE CHÍNH
# -------------------------------------------------------------------
async def process_video_pipeline():
    rendered_files = []
    character_images = {}   # {character_name_lower: local_path}
    scene_image_paths = {}

    # ========== 0. ĐỌC GOOGLE SHEET ==========
    print("\n📊 Đọc Google Sheet...")
    df_chars = get_sheet_as_df(CHARACTERS_SHEET)
    df_scenes = get_sheet_as_df(SCENES_SHEET)

    if df_scenes.empty:
        send_n8n_webhook("failed", error_message="Sheet Scenes trống hoặc không đọc được")
        sys.exit(1)

    total_scenes = len(df_scenes)
    print(f"✅ {total_scenes} cảnh cần xử lý")

    # Lấy thông tin project (nếu có)
    project_info = {}
    if not df_scenes.empty:
        first = df_scenes.iloc[0]
        for key in ["title", "genre", "visual_style", "world_setting", "bgm_prompt", "project_title"]:
            val = safe_get(first, key)
            if val:
                project_info[key] = val

    # ========== 1. TẢI DANH SÁCH FILE TRÊN DRIVE ==========
    print("\n📥 [1] Lấy danh sách file trên Drive...")
    try:
        drive_files = list_files_in_folder(DRIVE_FOLDER_ID)
        print(f"✅ {len(drive_files)} file trên Drive")
    except Exception as e:
        print(f"❌ Lỗi list Drive: {e}")
        send_n8n_webhook("failed", error_message=str(e))
        sys.exit(1)

    # ========== 2. TẠO / TẢI ẢNH NHÂN VẬT ==========
    print("\n🧑‍🎨 [2] Xử lý ảnh nhân vật...")
    if not df_chars.empty:
        for idx, row in df_chars.iterrows():
            char_id = safe_get(row, "character_id", default=f"char_{idx+1:02d}")
            char_name = safe_get(row, "character_name", default=f"Character_{idx+1}")
            design = safe_get(row, "design_detail", "appearance", "design")
            image_prompt = safe_get(row, "image_prompt", "design_detail")
            file_name = safe_get(row, "file_name", default=f"character_{idx+1:02d}_{char_name.lower().replace(' ', '_')}.png")

            local_path = f"char_{char_id}_{RUN_DATE}.png"
            name_key = char_name.lower().strip()

            # Đã có local?
            existing_local = list(Path(".").glob(f"char_{char_id}_*.png")) + list(Path(".").glob(f"*{file_name}"))
            if existing_local:
                character_images[name_key] = str(existing_local[0])
                print(f"⏩ {char_name}: dùng ảnh local")
                continue

            # Tìm trên Drive
            found = find_file_in_drive(drive_files, [char_id, file_name.replace(".png", ""), char_name.lower().replace(" ", "_")])
            if found:
                download_drive_file(found["id"], local_path)
                character_images[name_key] = local_path
                print(f"✅ {char_name}: tải từ Drive")
                continue

            # Chưa có → tạo mới
            if not image_prompt:
                image_prompt = f"full body character reference sheet of {design}, 3d chinese donghua style, unreal engine 5, highly detailed"

            success = generate_character_image(image_prompt, local_path)
            if success:
                upload_file_to_drive(local_path, DRIVE_FOLDER_ID)
                character_images[name_key] = local_path
            else:
                print(f"⚠️ {char_name}: tạo ảnh thất bại")
    else:
        print("⚠️ Không có dữ liệu Characters → bỏ qua bước tạo ảnh nhân vật")

    print(f"✅ Có {len(character_images)} ảnh nhân vật sẵn sàng")

    # ========== 3. TẠO / TẢI ẢNH CẢNH ==========
    print("\n🖼️  [3] Xử lý ảnh cảnh...")
    for index, row in df_scenes.iterrows():
        scene_index = int(row.get("scene_index")) if pd.notna(row.get("scene_index")) else index + 1
        local_img = f"image_{scene_index:03d}_{RUN_DATE}.png"

        # Đã có local?
        existing = list(Path(".").glob(f"image_{scene_index:03d}_*.png"))
        if existing:
            scene_image_paths[scene_index] = str(existing[0])
            print(f"⏩ Cảnh {scene_index}: ảnh local đã có")
            continue

        # Tìm trên Drive
        found = find_file_in_drive(drive_files, [f"scene_{scene_index:03d}", f"image_{scene_index:03d}", f"scene_{scene_index}"])
        if found:
            download_drive_file(found["id"], local_img)
            scene_image_paths[scene_index] = local_img
            print(f"✅ Cảnh {scene_index}: tải từ Drive")
            continue

        # Chưa có → tạo mới
        image_prompt = safe_get(row, "image_prompt", "scene_description", "prompt")
        char_name_raw = safe_get(row, "character_name")
        neg_prompt = safe_get(row, "negative_prompt")

        # Tìm ảnh nhân vật reference (ưu tiên nhân vật đầu tiên)
        ref_path = None
        if char_name_raw:
            for name in char_name_raw.split(","):
                key = name.strip().lower()
                if key in character_images:
                    ref_path = character_images[key]
                    break
                # fuzzy
                for ck, cp in character_images.items():
                    if key in ck or ck in key:
                        ref_path = cp
                        break
                if ref_path:
                    break

        if not image_prompt:
            image_prompt = "3d chinese donghua style scene, unreal engine 5, cinematic lighting"

        success = generate_scene_image(
            prompt=image_prompt,
            output_path=local_img,
            ref_image_path=ref_path,
        )
        if success:
            upload_file_to_drive(local_img, DRIVE_FOLDER_ID)
            scene_image_paths[scene_index] = local_img
        else:
            print(f"⚠️ Cảnh {scene_index}: tạo ảnh thất bại → bỏ qua video")

    print(f"✅ Có {len(scene_image_paths)} ảnh cảnh sẵn sàng")

    # ========== 4. TẠO BGM ==========
    bgm_desc = project_info.get("bgm_prompt") or f"{project_info.get('genre', 'dramatic')} cinematic background music"
    generate_ai_bgm(bgm_desc, duration_sec=total_scenes * 5, output_path=BGM_FILE)

    # ========== 5. LOAD LTX ==========
    print("\n🎬 [5] Load LTX...")
    try:
        ltx_pipe = LTXImageToVideoPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            torch_dtype=torch.bfloat16,
            token=hf_token_to_pass,
        )
        try:
            from diffusers.hooks import apply_group_offloading
            ltx_pipe = apply_group_offloading(ltx_pipe, offload_type="block_level")
            print("✅ apply_group_offloading")
        except Exception:
            ltx_pipe.enable_model_cpu_offload()
            print("✅ enable_model_cpu_offload")
        try:
            ltx_pipe.vae.enable_tiling()
            print("✅ vae.enable_tiling()")
        except Exception:
            pass
        print("✅ LTX ready")
    except Exception as e:
        print(f"❌ Load LTX lỗi: {e}")
        send_n8n_webhook("failed", error_message=str(e))
        sys.exit(1)

    # ========== 6. TẠO VIDEO TỪNG CẢNH ==========
    for index, row in df_scenes.iterrows():
        scene_start = time.time()
        scene_index = int(row.get("scene_index")) if pd.notna(row.get("scene_index")) else index + 1
        image_file = scene_image_paths.get(scene_index)

        if not image_file or not os.path.exists(image_file):
            print(f"⚠️ Cảnh {scene_index}: thiếu ảnh → bỏ qua")
            continue

        final_scene_file = f"scene_{scene_index:03d}_{RUN_DATE}.mp4"
        existing = list(Path(".").glob(f"scene_{scene_index:03d}_*.mp4"))
        if existing and os.path.getsize(str(existing[0])) > 15000:
            rendered_files.append(str(existing[0]))
            print(f"⏩ Cảnh {scene_index}: video đã có")
            continue

        vid_prompt = safe_get(row, "video_prompt", "camera_motion", default="smooth cinematic movement")
        neg_prompt = safe_get(row, "negative_prompt")
        dialogue_text = safe_get(row, "dialogue")
        char_name = safe_get(row, "character_name")
        sfx_prompt = safe_get(row, "sfx_type", "sound_effect", "sfx_prompt")

        raw_video_file = f"raw_scene_{scene_index:03d}_{RUN_DATE}.mp4"
        audio_tts_file = f"tts_scene_{scene_index:03d}_{RUN_DATE}.mp3"
        audio_sfx_file = f"sfx_scene_{scene_index:03d}_{RUN_DATE}.wav"

        print(f"\n🎬 [{scene_index}/{total_scenes}] LTX | {char_name[:40]}")

        # --- LTX ---
        try:
            clear_memory()
            image_input = load_image(image_file).resize((640, 360))
            video_frames = ltx_pipe(
                image=image_input,
                prompt=vid_prompt,
                negative_prompt=neg_prompt if neg_prompt else None,
                width=640,
                height=360,
                num_frames=25,
                num_inference_steps=20,
                generator=torch.Generator("cpu").manual_seed(42 + scene_index),
            ).frames[0]
            export_to_video(video_frames, raw_video_file, fps=16)
            del video_frames
            clear_memory()
            print(f"   ✅ Video thô xong")
        except Exception as e:
            print(f"❌ Lỗi LTX cảnh {scene_index}: {e}")
            clear_memory()
            continue

        # --- TTS ---
        a_dur = 2.0
        has_tts = False
        if dialogue_text:
            voice = pick_voice(char_name)
            print(f"🎙️ TTS [{char_name}] ({voice}): {dialogue_text[:50]}...")
            communicate = edge_tts.Communicate(text=dialogue_text, voice=voice)
            await communicate.save(audio_tts_file)
            a_dur = get_media_duration(audio_tts_file)
            has_tts = True

        # --- SFX ---
        has_sfx = False
        if sfx_prompt and sfx_prompt.lower() not in ["none", "nan", ""]:
            has_sfx = generate_ai_sfx(sfx_prompt, duration_sec=a_dur, output_path=audio_sfx_file)

        # --- Mix ---
        v_dur = get_media_duration(raw_video_file)
        pad_dur = max(0.0, a_dur - v_dur) if has_tts else 0.0

        inputs = [f'-i "{raw_video_file}"']
        filter_parts = []
        map_v = "0:v:0"
        map_a = "1:a:0"

        if pad_dur > 0.05:
            filter_parts.append(f"[0:v]tpad=stop_mode=clone:stop_duration={pad_dur:.3f}[v]")
            map_v = "[v]"

        if has_tts and has_sfx:
            inputs += [f'-i "{audio_tts_file}"', f'-i "{audio_sfx_file}"']
            filter_parts.append("[1:a]volume=1.0[tts];[2:a]volume=0.4[sfx];[tts][sfx]amix=inputs=2:duration=first[a]")
            map_a = "[a]"
        elif has_tts:
            inputs.append(f'-i "{audio_tts_file}"')
            map_a = "1:a:0"
        elif has_sfx:
            inputs.append(f'-i "{audio_sfx_file}"')
            map_a = "1:a:0"
        else:
            inputs.append('-f lavfi -i anullsrc=r=44100:cl=stereo')

        filter_str = f'-filter_complex "{";".join(filter_parts)}"' if filter_parts else ""
        mix_cmd = (
            f'ffmpeg -y {" ".join(inputs)} {filter_str} '
            f'-map {map_v} -map {map_a} -c:v libx264 -pix_fmt yuv420p -r 16 '
            f'-c:a aac -ar 44100 -ac 2 -b:a 192k -shortest "{final_scene_file}"'
        )
        subprocess.run(mix_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        for f in [raw_video_file, audio_tts_file, audio_sfx_file]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass

        if os.path.exists(final_scene_file) and os.path.getsize(final_scene_file) > 15000:
            upload_file_to_drive(final_scene_file, DRIVE_FOLDER_ID)
            rendered_files.append(final_scene_file)
            print(f"✅ Cảnh {scene_index} xong ({time.time() - scene_start:.1f}s)")
        else:
            print(f"⚠️ Cảnh {scene_index}: file lỗi")

    del ltx_pipe
    clear_memory()

    # ========== 7. GỘP + BGM ==========
    print("\n🎞️ [7] Gộp video...")
    if not rendered_files:
        send_n8n_webhook("failed", error_message="Không render được cảnh nào", extra=project_info)
        sys.exit(1)

    rendered_files = sorted(rendered_files)
    with open("file_list.txt", "w", encoding="utf-8") as f:
        for file in rendered_files:
            f.write(f"file '{file}'\n")

    concat_output = f"final_concat_{RUN_DATE}.mp4"
    final_output = f"final_movie_{RUN_DATE}.mp4"

    subprocess.run(f'ffmpeg -y -f concat -safe 0 -i file_list.txt -c copy "{concat_output}"', shell=True, check=True)

    if os.path.exists(BGM_FILE):
        print("🎵 Ghép BGM...")
        bgm_cmd = (
            f'ffmpeg -y -i "{concat_output}" -stream_loop -1 -i "{BGM_FILE}" '
            f'-filter_complex "[0:a]volume=1.0[a1];[1:a]volume=0.12[a2];[a1][a2]amix=inputs=2:duration=first[a]" '
            f'-map 0:v:0 -map "[a]" -c:v copy -c:a aac -ar 44100 -ac 2 -b:a 192k "{final_output}"'
        )
        try:
            subprocess.run(bgm_cmd, shell=True, check=True)
        except Exception:
            final_output = concat_output
    else:
        final_output = concat_output

    print(f"\n☁️ Upload phim: {final_output}")
    drive_file_id = upload_file_to_drive(final_output, DRIVE_FOLDER_ID)

    send_n8n_webhook(
        status="completed_all",
        total_scenes=len(rendered_files),
        final_file=final_output,
        drive_file_id=drive_file_id,
        extra=project_info,
    )
    print("\n🎉 HOÀN TẤT PIPELINE!")

# -------------------------------------------------------------------
# 8. RUN
# -------------------------------------------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(process_video_pipeline())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(process_video_pipeline())
