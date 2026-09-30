# ============================================================
# PIPELINE KAGGLE
# - Ảnh NV: tải từ image_url (Sheet) — KHÔNG tạo lại
# - Ảnh cảnh: áp ảnh NV tương ứng → nhất quán nhân vật
# - CPU offload + slicing | Upload Drive
# ============================================================
import asyncio
import gc
import os
import subprocess
import sys
import time
from pathlib import Path

def install_requirements():
    required = {
        "nest_asyncio": "nest_asyncio", "diffusers": "diffusers",
        "transformers": "transformers", "accelerate": "accelerate",
        "imageio": "imageio", "googleapiclient": "google-api-python-client",
        "google_auth_oauthlib": "google-auth-oauthlib",
        "huggingface_hub": "huggingface_hub", "soundfile": "soundfile",
        "scipy": "scipy", "av": "av", "edge_tts": "edge-tts",
        "sentencepiece": "sentencepiece", "ftfy": "ftfy",
        "safetensors": "safetensors", "omegaconf": "omegaconf",
        "einops": "einops", "cv2": "opencv-python", "pandas": "pandas",
        "requests": "requests", "PIL": "Pillow",
    }
    missing = []
    for imp, pip in required.items():
        try:
            __import__(imp)
        except ImportError:
            missing.append(pip)
    if not missing:
        print("✅ Package OK")
        return
    print(f"📦 Cài {len(missing)} package...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "--no-warn-script-location", "--disable-pip-version-check",
    ] + missing + ["protobuf<6.0.0,>=3.20.2", "imageio-ffmpeg"])
    print("✅ Cài xong")

install_requirements()

import nest_asyncio
nest_asyncio.apply()

import torch
import pandas as pd
import requests
import scipy.io.wavfile as wavfile
import edge_tts
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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  {DEVICE}")

# ----- CONFIG -----
N8N_WEBHOOK_URL = "https://n8n-latest-namx.onrender.com/webhook/kaggle-video-done"
SHEET_ID = "1DmA-yuPwDl1riceSMGzWPXhuxrL4y987lOZ6Af351l8"
DRIVE_FOLDER_ID = "1oXS7LweDNK2fYsWonQay3U-hUmEIsgCF"
GID_SCENES = "0"
GID_CHARACTERS = "1382939846"

HF_TOKEN = "hf_GJeIMPtqGWNZJInJn" + "TCVlxljdWnUzudVQH"
hf_token_to_pass = None
if HF_TOKEN.startswith("hf_"):
    try:
        login(token=HF_TOKEN)
        hf_token_to_pass = HF_TOKEN
        print("🔑 HF OK")
    except Exception as e:
        print(f"⚠️ HF: {e}")
if not hf_token_to_pass:
    raise SystemExit("Missing HF token")

OAUTH_CLIENT_ID = "948179937421-o55enfl61lb8ou0ms2jmrr4dlf1fhgip.apps.googleusercontent.com"
OAUTH_CLIENT_SECRET = "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"
OAUTH_REFRESH_TOKEN = "1//06GnOlI9wdLJ-CgYIARAAGAYSNwF-L9Ir5sxbZNKU6xqWnjWPP2jFwNaI8UnENzUrHgdc52RO-QIDl3NG8RQA6J_fzGe-vAR3zgA"
TOKEN_URI = "https://oauth2.googleapis.com/token"
USE_DRIVE = bool(OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET and OAUTH_REFRESH_TOKEN)
print("✅ Drive ON" if USE_DRIVE else "⚠️ LOCAL")

VOICE_MAP = {"nam": "vi-VN-NamMinhNeural", "nu": "vi-VN-HoaiMyNeural"}
BGM_FILE = f"bgm_generated_{RUN_DATE}.mp3"
IMAGE_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
SCENE_W, SCENE_H = 640, 360
VIDEO_W, VIDEO_H = 640, 360

_scene_pipe_t2i = None
_scene_pipe_i2i = None
_ltx_pipe = None

# ----- HELPERS -----
def get_oauth_credentials():
    if not USE_DRIVE:
        return None
    return Credentials(
        token=None, refresh_token=OAUTH_REFRESH_TOKEN, token_uri=TOKEN_URI,
        client_id=OAUTH_CLIENT_ID, client_secret=OAUTH_CLIENT_SECRET,
        scopes=["https://www.googleapis.com/auth/drive"],
    )

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

def prepare_offload_pipe(pipe):
    if DEVICE != "cuda":
        return pipe.to(DEVICE)
    try:
        pipe.enable_sequential_cpu_offload()
        print("   → sequential_cpu_offload ON")
    except Exception:
        try:
            pipe.enable_model_cpu_offload()
            print("   → model_cpu_offload ON")
        except Exception:
            pipe = pipe.to(DEVICE)
    try:
        pipe.enable_attention_slicing("max")
    except Exception:
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
    try:
        if getattr(pipe, "vae", None) is not None:
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
    except Exception:
        pass
    return pipe

def get_media_duration(path):
    try:
        r = subprocess.run(
            f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{path}"',
            shell=True, capture_output=True, text=True, check=True,
        )
        return float(r.stdout.strip())
    except Exception:
        return 2.0

def list_files_in_folder(folder_id):
    if not USE_DRIVE:
        return []
    try:
        service = build("drive", "v3", credentials=get_oauth_credentials(), cache_discovery=False)
        files, token = [], None
        while True:
            resp = service.files().list(
                q=f"'{folder_id}' in parents and trashed=false",
                fields="nextPageToken, files(id, name, mimeType)",
                pageSize=200, pageToken=token,
            ).execute()
            files.extend(resp.get("files", []))
            token = resp.get("nextPageToken")
            if not token:
                break
        return files
    except Exception as e:
        print(f"⚠️ list Drive: {e}")
        return []

def download_drive_file(file_id, save_path):
    if not USE_DRIVE:
        return None
    try:
        service = build("drive", "v3", credentials=get_oauth_credentials(), cache_discovery=False)
        req = service.files().get_media(fileId=file_id)
        with open(save_path, "wb") as f:
            dl = MediaIoBaseDownload(f, req)
            done = False
            while not done:
                _, done = dl.next_chunk()
        return save_path
    except Exception as e:
        print(f"⚠️ download: {e}")
        return None

def upload_file_to_drive(file_path, folder_id, retries=2):
    if not USE_DRIVE or not os.path.exists(file_path):
        return None
    name = os.path.basename(file_path)
    mime = "image/png" if name.endswith((".png", ".jpg", ".jpeg", ".webp")) else "video/mp4"
    if name.endswith(".mp3"):
        mime = "audio/mpeg"
    for i in range(1, retries + 1):
        try:
            service = build("drive", "v3", credentials=get_oauth_credentials(), cache_discovery=False)
            media = MediaFileUpload(file_path, mimetype=mime, resumable=True)
            up = service.files().create(
                body={"name": name, "parents": [folder_id]}, media_body=media, fields="id",
            ).execute()
            print(f"☁️ {name}")
            return up.get("id")
        except Exception as e:
            print(f"⚠️ upload {i}: {e}")
            time.sleep(1)
    return None

def send_n8n_webhook(status, total_scenes=0, final_file=None, drive_file_id=None, error_message=None, extra=None):
    payload = {
        "status": status, "total_scenes": total_scenes,
        "final_file": final_file, "drive_file_id": drive_file_id,
        "error_message": error_message, "run_date": RUN_DATE,
    }
    if extra:
        payload.update(extra)
    try:
        r = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=30)
        print(f"📡 [{status}] → {r.status_code}")
    except Exception as e:
        print(f"❌ webhook: {e}")

def find_file_in_drive(drive_files, keywords, extensions=(".png", ".jpg", ".jpeg", ".webp")):
    keywords = [k.lower() for k in keywords if k]
    for f in drive_files:
        name = f["name"].lower()
        if any(k in name for k in keywords) and name.endswith(extensions):
            return f
    return None

def pick_voice(char_name: str) -> str:
    name = (char_name or "").lower()
    if any(k in name for k in ["nữ", "cô", "chị", "muội", "my", "linh", "sera", "eva", "nhi"]):
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
        return df.dropna(how="all")
    except Exception as e:
        print(f"❌ CSV gid={gid}: {e}")
        return pd.DataFrame()

def find_character_ref(char_name_raw: str, character_images: dict):
    """Map tên trong scene → đường dẫn ảnh NV (nhất quán)."""
    if not char_name_raw or not character_images:
        return None
    for name in char_name_raw.split(","):
        key = name.strip().lower()
        if not key:
            continue
        if key in character_images:
            return character_images[key]
        for ck, path in character_images.items():
            if key in ck or ck in key:
                return path
    return None

# ----- AUDIO -----
def generate_ai_bgm(prompt_text, duration_sec, output_path):
    print(f"🎵 BGM...")
    try:
        clear_memory()
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
        if DEVICE == "cuda":
            try:
                model.enable_model_cpu_offload()
            except Exception:
                model = model.to(DEVICE)
        else:
            model = model.to(DEVICE)
        inputs = processor(text=[prompt_text], padding=True, return_tensors="pt")
        if DEVICE == "cuda":
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        audio = model.generate(**inputs, max_new_tokens=min(int(duration_sec * 50), 1500))
        sr = model.config.audio_encoder.sampling_rate
        wav = output_path.replace(".mp3", ".wav")
        wavfile.write(wav, rate=sr, data=audio[0, 0].cpu().numpy())
        subprocess.run(["ffmpeg", "-y", "-i", wav, "-acodec", "libmp3lame", output_path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(wav):
            os.remove(wav)
        del processor, model, audio
        clear_memory()
        print("✅ BGM")
        return True
    except Exception as e:
        print(f"⚠️ BGM: {e}")
        clear_memory()
        return False

def generate_ai_sfx(prompt_text, duration_sec, output_path):
    print(f"🔊 SFX: {prompt_text[:40]}")
    try:
        clear_memory()
        pipe = AudioLDMPipeline.from_pretrained(
            "cvssp/audioldm-m-full",
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        )
        pipe = prepare_offload_pipe(pipe)
        audio = pipe(prompt_text, num_inference_steps=15,
                     audio_length_in_s=max(1.0, min(duration_sec, 8.0))).audios[0]
        wavfile.write(output_path, rate=16000, data=audio)
        del pipe, audio
        clear_memory()
        return True
    except Exception as e:
        print(f"⚠️ SFX: {e}")
        clear_memory()
        return False

# ----- ẢNH CẢNH (có ref NV = img2img để nhất quán) -----
def get_scene_pipe(task="text2img"):
    global _scene_pipe_t2i, _scene_pipe_i2i
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    kwargs = {"torch_dtype": dtype, "token": hf_token_to_pass, "low_cpu_mem_usage": True}
    if task == "img2img":
        if _scene_pipe_i2i is None:
            print("📦 Load FLUX img2img (1 lần)...")
            clear_memory()
            _scene_pipe_i2i = AutoPipelineForImage2Image.from_pretrained(IMAGE_MODEL_ID, **kwargs)
            _scene_pipe_i2i = prepare_offload_pipe(_scene_pipe_i2i)
            print("✅ img2img ready")
        return _scene_pipe_i2i
    if _scene_pipe_t2i is None:
        print("📦 Load FLUX text2img (1 lần)...")
        clear_memory()
        _scene_pipe_t2i = AutoPipelineForText2Image.from_pretrained(IMAGE_MODEL_ID, **kwargs)
        _scene_pipe_t2i = prepare_offload_pipe(_scene_pipe_t2i)
        print("✅ text2img ready")
    return _scene_pipe_t2i

def generate_scene_image(prompt, output_path, ref_image_path=None, width=SCENE_W, height=SCENE_H):
    mode = "img2img" if (ref_image_path and os.path.exists(ref_image_path)) else "text2img"
    print(f"🖼️  {os.path.basename(output_path)} [{mode}] ref={os.path.basename(ref_image_path) if ref_image_path else 'none'}")
    try:
        clear_memory()
        if mode == "img2img":
            pipe = get_scene_pipe("img2img")
            init = load_image(ref_image_path).resize((width, height))
            # strength 0.55–0.70: giữ khuôn mặt/trang phục NV, đổi bối cảnh
            out = pipe(
                prompt=prompt, image=init, strength=0.62,
                num_inference_steps=4, guidance_scale=0.0,
                generator=torch.Generator("cpu").manual_seed(42),
            )
            image = out.images[0]
            del init, out
        else:
            pipe = get_scene_pipe("text2img")
            out = pipe(
                prompt=prompt, width=width, height=height,
                num_inference_steps=4, guidance_scale=0.0,
                generator=torch.Generator("cpu").manual_seed(42),
            )
            image = out.images[0]
            del out
        image.save(output_path)
        del image
        clear_memory()
        print(f"   ✅ {output_path}")
        return True
    except Exception as e:
        print(f"   ❌ {e}")
        clear_memory()
        return False

def get_ltx_pipe():
    global _ltx_pipe
    if _ltx_pipe is None:
        print("📦 Load LTX...")
        clear_memory()
        _ltx_pipe = LTXImageToVideoPipeline.from_pretrained(
            "Lightricks/LTX-Video", torch_dtype=torch.bfloat16,
            token=hf_token_to_pass, low_cpu_mem_usage=True,
        )
        if DEVICE == "cuda":
            try:
                from diffusers.hooks import apply_group_offloading
                _ltx_pipe = apply_group_offloading(_ltx_pipe, offload_type="block_level")
            except Exception:
                _ltx_pipe = prepare_offload_pipe(_ltx_pipe)
            try:
                _ltx_pipe.vae.enable_tiling()
                _ltx_pipe.vae.enable_slicing()
            except Exception:
                pass
        print("✅ LTX ready")
    return _ltx_pipe

# ----- MAIN -----
async def process_video_pipeline():
    global _scene_pipe_t2i, _scene_pipe_i2i, _ltx_pipe
    rendered_files = []
    character_images = {}   # name_lower -> local path
    scene_image_paths = {}

    print("\n📊 Đọc Sheet...")
    df_scenes = get_sheet_csv(GID_SCENES)
    df_chars = get_sheet_csv(GID_CHARACTERS)
    if df_scenes.empty:
        send_n8n_webhook("failed", error_message="Không đọc được Scenes")
        return

    total_scenes = len(df_scenes)
    print(f"✅ Scenes: {total_scenes} | cột: {df_scenes.columns.tolist()}")
    if not df_chars.empty:
        print(f"✅ Characters: {len(df_chars)} | cột: {df_chars.columns.tolist()}")

    project_info = {}
    first = df_scenes.iloc[0]
    for key in ["title", "genre", "visual_style", "world_setting", "bgm_prompt", "project_title"]:
        val = safe_get(first, key)
        if val:
            project_info[key] = val

    drive_files = list_files_in_folder(DRIVE_FOLDER_ID) if USE_DRIVE else []
    print(f"📥 Drive: {len(drive_files)} file")

    # ===== 1. ẢNH NHÂN VẬT TỪ SHEET (image_url) — KHÔNG tạo FLUX =====
    print("\n🧑‍🎨 [1] Tải ảnh NV từ Google Sheet (image_url)...")
    if not df_chars.empty:
        for idx, row in df_chars.iterrows():
            char_id = safe_get(row, "character_id", default=f"char_{idx+1:02d}")
            char_name = safe_get(row, "character_name", default=f"Character_{idx+1}")
            image_url = safe_get(row, "image_url")
            file_name = safe_get(row, "file_name", default=f"char_{char_id}.png")
            if not file_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                file_name += ".png"
            local_path = file_name
            name_key = char_name.lower().strip()

            # Bỏ qua weapon nếu không cần làm ref cảnh (vẫn tải được nếu có url)
            if not image_url:
                print(f"⚠️ {char_name}: thiếu image_url")
                continue
            if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
                character_images[name_key] = local_path
                print(f"⏩ {char_name}: local")
                continue
            try:
                print(f"⬇️  {char_name}...")
                resp = requests.get(image_url, timeout=60)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    with open(local_path, "wb") as f:
                        f.write(resp.content)
                    character_images[name_key] = local_path
                    print(f"✅ {char_name} → {local_path}")
                else:
                    print(f"⚠️ {char_name}: HTTP {resp.status_code}")
            except Exception as e:
                print(f"⚠️ {char_name}: {e}")
    print(f"✅ Map NV: {list(character_images.keys())}")

    # ===== 2. ẢNH CẢNH — áp đúng ảnh NV của cảnh =====
    print("\n🖼️  [2] Tạo ảnh cảnh (ref NV để nhất quán)...")
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
            if os.path.exists(local_img):
                scene_image_paths[scene_index] = local_img
                print(f"✅ Cảnh {scene_index}: Drive")
                continue

        image_prompt = safe_get(row, "image_prompt", "scene_description", "prompt")
        char_name_raw = safe_get(row, "character_name")
        ref_path = find_character_ref(char_name_raw, character_images)

        if ref_path:
            print(f"   → Cảnh {scene_index} dùng NV: {char_name_raw} ({os.path.basename(ref_path)})")
        else:
            print(f"   → Cảnh {scene_index}: không match NV → text2img")

        if not image_prompt:
            image_prompt = "3d chinese donghua style scene, unreal engine 5, cinematic lighting"

        # Nhắc giữ nhân vật trong prompt khi có ref
        if ref_path and char_name_raw:
            image_prompt = f"{image_prompt}, consistent character design of {char_name_raw}, same face and costume"

        if generate_scene_image(image_prompt, local_img, ref_path):
            upload_file_to_drive(local_img, DRIVE_FOLDER_ID)
            scene_image_paths[scene_index] = local_img
        else:
            print(f"⚠️ Cảnh {scene_index}: tạo fail")

    print(f"✅ {len(scene_image_paths)} ảnh cảnh")

    # Giải phóng FLUX trước LTX
    _scene_pipe_t2i = None
    _scene_pipe_i2i = None
    clear_memory()

    bgm_desc = project_info.get("bgm_prompt") or f"{project_info.get('genre', 'dramatic')} cinematic background music"
    generate_ai_bgm(bgm_desc, duration_sec=max(total_scenes * 5, 30), output_path=BGM_FILE)
    clear_memory()

    print("\n🎬 [3] Load LTX...")
    try:
        ltx_pipe = get_ltx_pipe()
    except Exception as e:
        print(f"❌ LTX: {e}")
        send_n8n_webhook("failed", error_message=str(e))
        return

    # ===== 4. VIDEO =====
    for index, row in df_scenes.iterrows():
        t0 = time.time()
        scene_index = int(row.get("scene_index")) if pd.notna(row.get("scene_index")) else index + 1
        image_file = scene_image_paths.get(scene_index)
        if not image_file or not os.path.exists(image_file):
            print(f"⚠️ Cảnh {scene_index}: thiếu ảnh")
            continue

        final_scene_file = f"scene_{scene_index:03d}_{RUN_DATE}.mp4"
        existing = list(Path(".").glob(f"scene_{scene_index:03d}_*.mp4"))
        if existing and os.path.getsize(str(existing[0])) > 15000:
            rendered_files.append(str(existing[0]))
            print(f"⏩ Cảnh {scene_index}: video có sẵn")
            continue

        vid_prompt = safe_get(row, "video_prompt", "camera_motion", default="smooth cinematic movement")
        neg_prompt = safe_get(row, "negative_prompt")
        dialogue_text = safe_get(row, "dialogue")
        char_name = safe_get(row, "character_name")
        sfx_prompt = safe_get(row, "sfx_type", "sound_effect", "sfx_prompt")

        raw_video = f"raw_scene_{scene_index:03d}_{RUN_DATE}.mp4"
        tts_file = f"tts_scene_{scene_index:03d}_{RUN_DATE}.mp3"
        sfx_file = f"sfx_scene_{scene_index:03d}_{RUN_DATE}.wav"

        print(f"\n🎬 [{scene_index}/{total_scenes}] {char_name[:40] if char_name else ''}")

        try:
            clear_memory()
            img = load_image(image_file).resize((VIDEO_W, VIDEO_H))
            frames = ltx_pipe(
                image=img, prompt=vid_prompt,
                negative_prompt=neg_prompt or None,
                width=VIDEO_W, height=VIDEO_H,
                num_frames=25, num_inference_steps=20,
                generator=torch.Generator("cpu").manual_seed(42 + scene_index),
            ).frames[0]
            export_to_video(frames, raw_video, fps=16)
            del frames, img
            clear_memory()
            print("   ✅ video thô")
        except Exception as e:
            print(f"❌ LTX: {e}")
            clear_memory()
            continue

        a_dur, has_tts, has_sfx = 2.0, False, False
        if dialogue_text:
            await edge_tts.Communicate(text=dialogue_text, voice=pick_voice(char_name)).save(tts_file)
            a_dur = get_media_duration(tts_file)
            has_tts = True
        if sfx_prompt and sfx_prompt.lower() not in ["none", "nan", ""]:
            has_sfx = generate_ai_sfx(sfx_prompt, a_dur, sfx_file)

        v_dur = get_media_duration(raw_video)
        pad = max(0.0, a_dur - v_dur) if has_tts else 0.0
        inputs = [f'-i "{raw_video}"']
        filters, map_v, map_a = [], "0:v:0", "1:a:0"
        if pad > 0.05:
            filters.append(f"[0:v]tpad=stop_mode=clone:stop_duration={pad:.3f}[v]")
            map_v = "[v]"
        if has_tts and has_sfx:
            inputs += [f'-i "{tts_file}"', f'-i "{sfx_file}"']
            filters.append("[1:a]volume=1.0[tts];[2:a]volume=0.4[sfx];[tts][sfx]amix=inputs=2:duration=first[a]")
            map_a = "[a]"
        elif has_tts:
            inputs.append(f'-i "{tts_file}"')
        elif has_sfx:
            inputs.append(f'-i "{sfx_file}"')
        else:
            inputs.append("-f lavfi -i anullsrc=r=44100:cl=stereo")

        fc = f'-filter_complex "{";".join(filters)}"' if filters else ""
        subprocess.run(
            f'ffmpeg -y {" ".join(inputs)} {fc} -map {map_v} -map {map_a} '
            f'-c:v libx264 -pix_fmt yuv420p -r 16 -c:a aac -ar 44100 -ac 2 -b:a 192k -shortest "{final_scene_file}"',
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for f in [raw_video, tts_file, sfx_file]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

        if os.path.exists(final_scene_file) and os.path.getsize(final_scene_file) > 15000:
            upload_file_to_drive(final_scene_file, DRIVE_FOLDER_ID)
            rendered_files.append(final_scene_file)
            print(f"✅ Cảnh {scene_index} ({time.time()-t0:.1f}s)")
        else:
            print(f"⚠️ Cảnh {scene_index}: file lỗi")

    _ltx_pipe = None
    clear_memory()

    print("\n🎞️ [5] Gộp...")
    if not rendered_files:
        send_n8n_webhook("failed", error_message="Không có video", extra=project_info)
        return

    rendered_files = sorted(rendered_files)
    with open("file_list.txt", "w", encoding="utf-8") as f:
        for p in rendered_files:
            f.write(f"file '{p}'\n")

    concat_out = f"final_concat_{RUN_DATE}.mp4"
    final_out = f"final_movie_{RUN_DATE}.mp4"
    subprocess.run(f'ffmpeg -y -f concat -safe 0 -i file_list.txt -c copy "{concat_out}"', shell=True, check=True)

    if os.path.exists(BGM_FILE):
        try:
            subprocess.run(
                f'ffmpeg -y -i "{concat_out}" -stream_loop -1 -i "{BGM_FILE}" '
                f'-filter_complex "[0:a]volume=1.0[a1];[1:a]volume=0.12[a2];[a1][a2]amix=inputs=2:duration=first[a]" '
                f'-map 0:v:0 -map "[a]" -c:v copy -c:a aac -ar 44100 -ac 2 -b:a 192k "{final_out}"',
                shell=True, check=True,
            )
        except Exception:
            final_out = concat_out
    else:
        final_out = concat_out

    drive_id = upload_file_to_drive(final_out, DRIVE_FOLDER_ID)
    send_n8n_webhook("completed_all", total_scenes=len(rendered_files),
                     final_file=final_out, drive_file_id=drive_id, extra=project_info)
    print("\n🎉 HOÀN TẤT!")

if __name__ == "__main__":
    try:
        asyncio.run(process_video_pipeline())
    except RuntimeError:
        asyncio.get_event_loop().run_until_complete(process_video_pipeline())
