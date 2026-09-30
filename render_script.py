# ============================================================
# PIPELINE KAGGLE
# - Đọc dữ liệu từ Google Sheet
# - Ảnh nhân vật: tải từ image_url trong Sheet
# - Tạo ảnh cảnh → upload Drive
# - Tạo video + TTS + SFX + BGM
# ============================================================
import asyncio
import gc
import os
import subprocess
import sys
import time
from pathlib import Path

# -------------------------------------------------------------------
# 1. CÀI PACKAGE
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
        print("✅ Tất cả package đã có sẵn")
        return
    print(f"📦 Đang cài {len(missing)} package...")
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
print(f"🚀 PIPELINE - [{RUN_DATE}]")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Device: {DEVICE}")

# -------------------------------------------------------------------
# 2. CONFIG
# -------------------------------------------------------------------
N8N_WEBHOOK_URL = "https://n8n-latest-namx.onrender.com/webhook/kaggle-video-done"
SHEET_ID = "1DmA-yuPwDl1riceSMGzWPXhuxrL4y987lOZ6Af351l8"
DRIVE_FOLDER_ID = "1oXS7LweDNK2fYsWonQay3U-hUmEIsgCF"

GID_SCENES = "0"
GID_CHARACTERS = "1382939846"

# HF Token
HF1 = "hf_GJeIMPtqGWNZJInJn"
HF2 = "TCVlxljdWnUzudVQH"
HF_TOKEN = HF1 + HF2

hf_token_to_pass = None
if HF_TOKEN.startswith("hf_"):
    try:
        login(token=HF_TOKEN)
        hf_token_to_pass = HF_TOKEN
        print("🔑 HF login OK")
    except Exception as e:
        print(f"⚠️ HF login fail: {e}")

if not hf_token_to_pass:
    raise SystemExit("Missing Hugging Face token")

# OAuth
OAUTH_CLIENT_ID     = "948179937421-o55enfl61lb8ou0ms2jmrr4dlf1fhgip.apps.googleusercontent.com"
OAUTH_CLIENT_SECRET = "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"
OAUTH_REFRESH_TOKEN = "1//06GnOlI9wdLJ-CgYIARAAGAYSNwF-L9Ir5sxbZNKU6xqWnjWPP2jFwNaI8UnENzUrHgdc52RO-QIDl3NG8RQA6J_fzGe-vAR3zgA"
TOKEN_URI = "https://oauth2.googleapis.com/token"

print("\n🔐 Kiểm tra OAuth:")
print(f"   CLIENT_ID     : {'✅' if OAUTH_CLIENT_ID else '❌'}")
print(f"   CLIENT_SECRET : {'✅' if OAUTH_CLIENT_SECRET else '❌'}")
print(f"   REFRESH_TOKEN : {'✅' if OAUTH_REFRESH_TOKEN else '❌'}")

USE_DRIVE = bool(OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET and OAUTH_REFRESH_TOKEN)
if not USE_DRIVE:
    print("⚠️ Thiếu OAuth → chạy LOCAL")
else:
    print("✅ OAuth OK → sẽ dùng Drive")

VOICE_MAP = {
    "nam": "vi-VN-NamMinhNeural",
    "nu": "vi-VN-HoaiMyNeural",
}
BGM_FILE = f"bgm_generated_{RUN_DATE}.mp3"
IMAGE_MODEL_ID = "black-forest-labs/FLUX.1-schnell"

# -------------------------------------------------------------------
# 3. HELPERS
# -------------------------------------------------------------------
def get_oauth_credentials():
    if not USE_DRIVE:
        return None
    return Credentials(
        token=None,
        refresh_token=OAUTH_REFRESH_TOKEN,
        token_uri=TOKEN_URI,
        client_id=OAUTH_CLIENT_ID,
        client_secret=OAUTH_CLIENT_SECRET,
        scopes=["https://www.googleapis.com/auth/drive"],
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
    if not USE_DRIVE:
        return []
    try:
        creds = get_oauth_credentials()
        if creds is None:
            return []
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
    except Exception as e:
        print(f"⚠️ list Drive lỗi: {e}")
        return []

def download_drive_file(file_id, save_path):
    if not USE_DRIVE:
        return None
    try:
        creds = get_oauth_credentials()
        if creds is None:
            return None
        service = build("drive", "v3", credentials=creds, cache_discovery=False)
        request = service.files().get_media(fileId=file_id)
        with open(save_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
        return save_path
    except Exception as e:
        print(f"⚠️ download lỗi: {e}")
        return None

def upload_file_to_drive(file_path, folder_id, retries=2):
    if not USE_DRIVE or not os.path.exists(file_path):
        return None
    file_name = os.path.basename(file_path)
    mimetype = "image/png" if file_name.endswith((".png", ".jpg", ".jpeg", ".webp")) else "video/mp4"
    if file_name.endswith(".mp3"):
        mimetype = "audio/mpeg"
    for attempt in range(1, retries + 1):
        try:
            creds = get_oauth_credentials()
            if creds is None:
                return None
            service = build("drive", "v3", credentials=creds, cache_discovery=False)
            meta = {"name": file_name, "parents": [folder_id]}
            media = MediaFileUpload(file_path, mimetype=mimetype, resumable=True)
            uploaded = service.files().create(body=meta, media_body=media, fields="id").execute()
            print(f"☁️ Upload OK: {file_name}")
            return uploaded.get("id")
        except Exception as e:
            print(f"⚠️ Upload lần {attempt} lỗi: {e}")
            time.sleep(1)
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
        res = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=30)
        print(f"📡 Webhook [{status}] → {res.status_code}")
    except Exception as e:
        print(f"❌ Webhook lỗi: {e}")

def find_file_in_drive(drive_files, keywords, extensions=(".png", ".jpg", ".jpeg", ".webp")):
    keywords = [k.lower() for k in keywords if k]
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

def get_sheet_csv(gid: str) -> pd.DataFrame:
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={gid}"
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip().str.lower()
        df = df.dropna(how="all")
        return df
    except Exception as e:
        print(f"❌ Lỗi đọc CSV (gid={gid}): {e}")
        return pd.DataFrame()

# -------------------------------------------------------------------
# 3.1 AI MUSIC & SFX
# -------------------------------------------------------------------
def generate_ai_bgm(prompt_text: str, duration_sec: int, output_path: str):
    print(f"🎵 [AI BGM] '{prompt_text[:50]}...'")
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
        print("✅ [AI BGM] Xong")
        return True
    except Exception as e:
        print(f"⚠️ [AI BGM] Lỗi: {e}")
        return False

def generate_ai_sfx(prompt_text: str, duration_sec: float, output_path: str):
    print(f"🔊 [AI SFX] {prompt_text}")
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
# 3.2 TẠO ẢNH CẢNH
# -------------------------------------------------------------------
def load_image_pipeline(task="text2img"):
    clear_memory()
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    kwargs = {"torch_dtype": dtype, "token": hf_token_to_pass}
    if task == "text2img":
        pipe = AutoPipelineForText2Image.from_pretrained(IMAGE_MODEL_ID, **kwargs)
    else:
        pipe = AutoPipelineForImage2Image.from_pretrained(IMAGE_MODEL_ID, **kwargs)
    if DEVICE == "cuda":
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pipe = pipe.to(DEVICE)
    return pipe

def generate_scene_image(prompt: str, output_path: str, ref_image_path: str = None, width=1280, height=720):
    print(f"🖼️  [Scene] {os.path.basename(output_path)}")
    try:
        if ref_image_path and os.path.exists(ref_image_path):
            pipe = load_image_pipeline("img2img")
            init_image = load_image(ref_image_path).resize((width, height))
            image = pipe(
                prompt=prompt,
                image=init_image,
                strength=0.65,
                num_inference_steps=4 if "schnell" in IMAGE_MODEL_ID else 20,
                guidance_scale=0.0 if "schnell" in IMAGE_MODEL_ID else 7.5,
                generator=torch.Generator("cpu").manual_seed(42),
            ).images[0]
        else:
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
        print(f"   ✅ {output_path}")
        return True
    except Exception as e:
        print(f"   ❌ Lỗi: {e}")
        clear_memory()
        return False

# -------------------------------------------------------------------
# 4. PIPELINE CHÍNH
# -------------------------------------------------------------------
async def process_video_pipeline():
    rendered_files = []
    character_images = {}
    scene_image_paths = {}

    print("\n📊 Đọc Google Sheet...")
    df_scenes = get_sheet_csv(GID_SCENES)
    df_chars  = get_sheet_csv(GID_CHARACTERS)

    if df_scenes.empty:
        send_n8n_webhook("failed", error_message="Không đọc được Scenes")
        return

    total_scenes = len(df_scenes)
    print(f"✅ Scenes: {total_scenes} cảnh")
    print(f"   Cột: {df_scenes.columns.tolist()}")

    if not df_chars.empty:
        print(f"✅ Characters: {len(df_chars)} nhân vật")
        print(f"   Cột: {df_chars.columns.tolist()}")
    else:
        print("⚠️ Không đọc được Characters")

    project_info = {}
    if not df_scenes.empty:
        first = df_scenes.iloc[0]
        for key in ["title", "genre", "visual_style", "world_setting", "bgm_prompt", "project_title"]:
            val = safe_get(first, key)
            if val:
                project_info[key] = val

    # ========== 1. LẤY DANH SÁCH FILE TRÊN DRIVE ==========
    drive_files = []
    if USE_DRIVE:
        print("\n📥 [1] Lấy danh sách file trên Drive...")
        drive_files = list_files_in_folder(DRIVE_FOLDER_ID)
        print(f"✅ {len(drive_files)} file trên Drive")
    else:
        print("\n⚠️ [1] Chạy LOCAL (không dùng Drive)")

    # ========== 2. ẢNH NHÂN VẬT - TẢI TỪ image_url TRONG SHEET ==========
    print("\n🧑‍🎨 [2] Tải ảnh nhân vật từ URL trong Google Sheet...")
    if not df_chars.empty:
        for idx, row in df_chars.iterrows():
            char_id = safe_get(row, "character_id", default=f"char_{idx+1:02d}")
            char_name = safe_get(row, "character_name", default=f"Character_{idx+1}")
            image_url = safe_get(row, "image_url")
            file_name = safe_get(row, "file_name", default=f"char_{char_id}.png")

            if not file_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                file_name = file_name + ".png"
            local_path = file_name
            name_key = char_name.lower().strip()

            if not image_url:
                print(f"⚠️ {char_name}: không có image_url → bỏ qua")
                continue

            if os.path.exists(local_path):
                character_images[name_key] = local_path
                print(f"⏩ {char_name}: đã có local → {local_path}")
                continue

            try:
                print(f"⬇️  {char_name}: đang tải từ URL...")
                resp = requests.get(image_url, timeout=60)
                if resp.status_code == 200:
                    with open(local_path, "wb") as f:
                        f.write(resp.content)
                    character_images[name_key] = local_path
                    print(f"✅ {char_name}: tải thành công → {local_path}")
                else:
                    print(f"⚠️ {char_name}: tải lỗi HTTP {resp.status_code}")
            except Exception as e:
                print(f"⚠️ {char_name}: lỗi tải → {e}")
    else:
        print("⚠️ Không có dữ liệu Characters")

    print(f"✅ Có {len(character_images)} ảnh nhân vật")

    # ========== 3. TẠO ẢNH CẢNH + UPLOAD DRIVE ==========
    print("\n🖼️  [3] Tạo ảnh cảnh...")
    for index, row in df_scenes.iterrows():
        scene_index = int(row.get("scene_index")) if pd.notna(row.get("scene_index")) else index + 1
        local_img = f"image_{scene_index:03d}_{RUN_DATE}.png"

        existing = list(Path(".").glob(f"image_{scene_index:03d}_*.png"))
        if existing:
            scene_image_paths[scene_index] = str(existing[0])
            print(f"⏩ Cảnh {scene_index}: local")
            continue

        found = find_file_in_drive(drive_files, [f"scene_{scene_index:03d}", f"image_{scene_index:03d}"])
        if found:
            download_drive_file(found["id"], local_img)
            scene_image_paths[scene_index] = local_img
            print(f"✅ Cảnh {scene_index}: từ Drive")
            continue

        image_prompt = safe_get(row, "image_prompt", "scene_description", "prompt")
        char_name_raw = safe_get(row, "character_name")
        ref_path = None
        if char_name_raw:
            for name in char_name_raw.split(","):
                key = name.strip().lower()
                if key in character_images:
                    ref_path = character_images[key]
                    break
                for ck, cp in character_images.items():
                    if key in ck or ck in key:
                        ref_path = cp
                        break
                if ref_path:
                    break

        if not image_prompt:
            image_prompt = "3d chinese donghua style scene, unreal engine 5, cinematic lighting"

        if generate_scene_image(image_prompt, local_img, ref_path):
            upload_file_to_drive(local_img, DRIVE_FOLDER_ID)
            scene_image_paths[scene_index] = local_img
        else:
            print(f"⚠️ Cảnh {scene_index}: tạo thất bại")

    print(f"✅ Có {len(scene_image_paths)} ảnh cảnh")

    # ========== 4. BGM ==========
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
        except Exception:
            ltx_pipe.enable_model_cpu_offload()
        try:
            ltx_pipe.vae.enable_tiling()
        except Exception:
            pass
        print("✅ LTX ready")
    except Exception as e:
        print(f"❌ Load LTX lỗi: {e}")
        send_n8n_webhook("failed", error_message=str(e))
        return

    # ========== 6. TẠO VIDEO ==========
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

        print(f"\n🎬 [{scene_index}/{total_scenes}] {char_name[:40] if char_name else ''}")

        try:
            clear_memory()
            image_input = load_image(image_file).resize((640, 360))
            video_frames = ltx_pipe(
                image=image_input,
                prompt=vid_prompt,
                negative_prompt=neg_prompt if neg_prompt else None,
                width=640, height=360,
                num_frames=25,
                num_inference_steps=20,
                generator=torch.Generator("cpu").manual_seed(42 + scene_index),
            ).frames[0]
            export_to_video(video_frames, raw_video_file, fps=16)
            del video_frames
            clear_memory()
            print("   ✅ Video thô")
        except Exception as e:
            print(f"❌ LTX lỗi: {e}")
            clear_memory()
            continue

        a_dur = 2.0
        has_tts = False
        if dialogue_text:
            voice = pick_voice(char_name)
            print(f"🎙️ TTS: {dialogue_text[:40]}...")
            communicate = edge_tts.Communicate(text=dialogue_text, voice=voice)
            await communicate.save(audio_tts_file)
            a_dur = get_media_duration(audio_tts_file)
            has_tts = True

        has_sfx = False
        if sfx_prompt and sfx_prompt.lower() not in ["none", "nan", ""]:
            has_sfx = generate_ai_sfx(sfx_prompt, duration_sec=a_dur, output_path=audio_sfx_file)

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
        elif has_sfx:
            inputs.append(f'-i "{audio_sfx_file}"')
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
                try: os.remove(f)
                except: pass

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
        return

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

    print(f"\n☁️ Upload phim cuối: {final_output}")
    drive_file_id = upload_file_to_drive(final_output, DRIVE_FOLDER_ID)

    send_n8n_webhook(
        status="completed_all",
        total_scenes=len(rendered_files),
        final_file=final_output,
        drive_file_id=drive_file_id,
        extra=project_info,
    )
    print("\n🎉 HOÀN TẤT!")

# -------------------------------------------------------------------
# RUN
# -------------------------------------------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(process_video_pipeline())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(process_video_pipeline())
