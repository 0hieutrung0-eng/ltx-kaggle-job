# ============================================================
# PIPELINE HOÀN CHỈNH:
# Sheet + Drive ảnh → LTX I2V → Lồng tiếng VI → AI BGM & SFX → 1 video
# Upload Drive + Webhook n8n
# ============================================================

import asyncio
import gc
import io
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# -------------------------------------------------------------------
# 1. CÀI PACKAGE
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
        "google-auth-httplib2",
        "huggingface_hub",
        "soundfile",
        "scipy",
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
        "Pillow",
    ]
    print("📦 Cài packages...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "--no-warn-script-location", "--disable-pip-version-check",
    ] + packages)

install_requirements()

import nest_asyncio
nest_asyncio.apply()

import torch
import pandas as pd
import requests
import scipy.io.wavfile as wavfile
import edge_tts
from PIL import Image
from diffusers import LTXImageToVideoPipeline, AudioLDMPipeline
from diffusers.utils import export_to_video, load_image
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from huggingface_hub import login

RUN_DATE = time.strftime("%Y%m%d_%H%M%S")
print(f"🚀 LTX + TTS + AI BGM/SFX PIPELINE - [{RUN_DATE}]")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------------------
# 2. CONFIG (đổi theo của bạn)
# -------------------------------------------------------------------
N8N_WEBHOOK_URL = "https://n8n-latest-namx.onrender.com/webhook/kaggle-video-done"
SHEET_ID = "1DmA-yuPwDl1riceSMGzWPXhuxrL4y987lOZ6Af351l8"
GOOGLE_SHEET_CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
DRIVE_FOLDER_ID = "1oXS7LweDNK2fYsWonQay3U-hUmEIsgCF"

# Hugging Face (nên để Secrets, không hard-code)
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

# Google OAuth (nên để Secrets)
OAUTH_CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID", "")
OAUTH_CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET", "")
OAUTH_REFRESH_TOKEN = os.environ.get("OAUTH_REFRESH_TOKEN", "")
TOKEN_URI = "https://oauth2.googleapis.com/token"

VOICE_MAP = {
    "nam": "vi-VN-NamMinhNeural",
    "nu": "vi-VN-HoaiMyNeural",
}

BGM_FILE = f"bgm_generated_{RUN_DATE}.mp3"

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
    mimetype = "image/png" if file_name.endswith(".png") else "video/mp4"
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

def find_scene_image(drive_files, scene_index):
    keys = [
        f"scene_{scene_index:03d}",
        f"image_{scene_index:03d}",
        f"scene_{scene_index}",
    ]
    for f in drive_files:
        name = f["name"].lower()
        if any(k in name for k in keys) and name.endswith((".png", ".jpg", ".jpeg", ".webp")):
            return f
    return None

def pick_voice(char_name: str) -> str:
    name = (char_name or "").lower()
    if any(k in name for k in ["nữ", "cô", "chị", "muội", "evo", "my", "linh"]):
        return VOICE_MAP["nu"]
    return VOICE_MAP["nam"]

# -------------------------------------------------------------------
# 3.1 AI MUSIC & AUDIO GENERATION HELPERS
# -------------------------------------------------------------------
def generate_ai_bgm(prompt_text: str, duration_sec: int, output_path: str):
    """Tự động tạo nhạc nền BGM dựa vào phong cách câu chuyện"""
    print(f"🎵 [AI BGM] Đang tạo BGM theo mô tả: '{prompt_text}' ({duration_sec}s)...")
    try:
        clear_memory()
        processor_bgm = AutoProcessor.from_pretrained("facebook/musicgen-small")
        model_bgm = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(DEVICE)
        
        inputs = processor_bgm(text=[prompt_text], padding=True, return_tensors="pt").to(DEVICE)
        max_tokens = min(int(duration_sec * 50), 1500) # Giới hạn tối đa ~30s sample (có thể loop bằng FFmpeg)
        
        audio_outputs = model_bgm.generate(**inputs, max_new_tokens=max_tokens)
        sampling_rate = model_bgm.config.audio_encoder.sampling_rate
        
        wav_path = output_path.replace(".mp3", ".wav")
        wavfile.write(wav_path, rate=sampling_rate, data=audio_outputs[0, 0].cpu().numpy())
        
        # Convert WAV -> MP3
        subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-acodec", "libmp3lame", output_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(wav_path):
            os.remove(wav_path)
            
        del processor_bgm, model_bgm
        clear_memory()
        print(f"✅ [AI BGM] Đã sinh nhạc nền: {output_path}")
        return True
    except Exception as e:
        print(f"⚠️ [AI BGM] Không tạo được BGM: {e}")
        return False

def generate_ai_sfx(prompt_text: str, duration_sec: float, output_path: str):
    """Tự động tạo hiệu ứng âm thanh SFX cho từng cảnh"""
    print(f"🔊 [AI SFX] Đang sinh âm thanh: '{prompt_text}'...")
    try:
        clear_memory()
        pipe_sfx = AudioLDMPipeline.from_pretrained("cvssp/audioldm-m-full", torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32)
        pipe_sfx = pipe_sfx.to(DEVICE)
        
        audio = pipe_sfx(prompt_text, num_inference_steps=20, audio_length_in_s=max(1.0, duration_sec)).audios[0]
        wavfile.write(output_path, rate=16000, data=audio)
        
        del pipe_sfx
        clear_memory()
        print(f"✅ [AI SFX] Đã tạo SFX: {output_path}")
        return True
    except Exception as e:
        print(f"⚠️ [AI SFX] Lỗi tạo SFX: {e}")
        return False

# -------------------------------------------------------------------
# 4. ĐỌC GOOGLE SHEET
# -------------------------------------------------------------------
print("\n📊 Đọc Google Sheet...")
try:
    df = pd.read_csv(GOOGLE_SHEET_CSV_URL)
    df.columns = df.columns.str.strip().str.lower()
    total_scenes = len(df)
    print(f"✅ {total_scenes} cảnh | Cột: {df.columns.tolist()}")
except Exception as e:
    print(f"❌ Lỗi Sheet: {e}")
    send_n8n_webhook("failed", error_message=str(e))
    sys.exit(1)

project_info = {}
if not df.empty:
    first = df.iloc[0]
    for key in ["title", "genre", "visual_style", "world_setting", "bgm_prompt"]:
        if key in df.columns:
            val = str(first[key]).strip()
            if val.lower() not in ["nan", "", "[empty]"]:
                project_info[key] = val

# -------------------------------------------------------------------
# 5. PIPELINE
# -------------------------------------------------------------------
async def process_video_pipeline():
    rendered_files = []
    image_paths = {}

    # ========== GIAI ĐOẠN 1: TẢI ẢNH TỪ DRIVE ==========
    print("\n📥 [1] Tải ảnh cảnh từ Google Drive...")
    try:
        drive_files = list_files_in_folder(DRIVE_FOLDER_ID)
        print(f"✅ {len(drive_files)} file trên Drive")
    except Exception as e:
        print(f"❌ Lỗi list Drive: {e}")
        send_n8n_webhook("failed", error_message=str(e))
        sys.exit(1)

    for index, row in df.iterrows():
        scene_idx_val = row.get("scene_index")
        scene_index = int(scene_idx_val) if pd.notna(scene_idx_val) else index + 1
        local_img = f"image_{scene_index:03d}_{RUN_DATE}.png"

        existing = list(Path(".").glob(f"image_{scene_index:03d}_*.png"))
        if existing:
            image_paths[scene_index] = str(existing[0])
            print(f"⏩ Cảnh {scene_index}: ảnh local đã có")
            continue

        found = find_scene_image(drive_files, scene_index)
        if found:
            download_drive_file(found["id"], local_img)
            image_paths[scene_index] = local_img
            print(f"✅ Cảnh {scene_index}: tải {found['name']}")
        else:
            print(f"⚠️ Cảnh {scene_index}: không tìm thấy ảnh trên Drive")

    # ========== GIAI ĐOẠN 1.5: TỰ ĐỘNG TẠO BGM CHO TOÀN BỘ PHIM/CHƯƠNG ==========
    bgm_desc = project_info.get("bgm_prompt", f"{project_info.get('genre', 'dramatic')} cinematic background music")
    estimated_total_duration = total_scenes * 6  # Ước tính 6s mỗi cảnh
    generate_ai_bgm(bgm_desc, duration_sec=estimated_total_duration, output_path=BGM_FILE)

    # ========== GIAI ĐOẠN 2: LTX IMAGE → VIDEO ==========
    print("\n🎬 [2] Load LTX Image-to-Video...")
    try:
        ltx_pipe = LTXImageToVideoPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            torch_dtype=torch.bfloat16,
            token=hf_token_to_pass,
        )
        ltx_pipe.enable_model_cpu_offload()
        print("✅ LTX ready")
    except Exception as e:
        print(f"❌ Load LTX lỗi: {e}")
        send_n8n_webhook("failed", error_message=str(e))
        sys.exit(1)

    for index, row in df.iterrows():
        scene_start = time.time()
        scene_idx_val = row.get("scene_index")
        scene_index = int(scene_idx_val) if pd.notna(scene_idx_val) else index + 1

        image_file = image_paths.get(scene_index)
        if not image_file or not os.path.exists(image_file):
            print(f"⚠️ Cảnh {scene_index}: thiếu ảnh → bỏ qua")
            continue

        final_scene_file = f"scene_{scene_index:03d}_{RUN_DATE}.mp4"
        existing_scenes = list(Path(".").glob(f"scene_{scene_index:03d}_*.mp4"))
        if existing_scenes and os.path.getsize(str(existing_scenes[0])) > 10000:
            rendered_files.append(str(existing_scenes[0]))
            print(f"⏩ Cảnh {scene_index}: video đã có")
            continue

        vid_prompt = str(row.get("video_prompt", "")).strip()
        neg_prompt = str(row.get("negative_prompt", "")).strip()
        dialogue_text = str(row.get("dialogue", "")).strip()
        char_name = str(row.get("character_name", "")).strip()
        
        # Cột từ khóa âm thanh SFX (vd: "thunderstorm, sword clash")
        sfx_prompt = str(row.get("sound_effect", row.get("sfx_prompt", ""))).strip()

        if dialogue_text.lower() in ["nan", "[empty]", "none", "null"]:
            dialogue_text = ""
        if sfx_prompt.lower() in ["nan", "[empty]", "none", "null"]:
            sfx_prompt = ""

        raw_video_file = f"raw_scene_{scene_index:03d}_{RUN_DATE}.mp4"
        audio_tts_file = f"tts_scene_{scene_index:03d}_{RUN_DATE}.mp3"
        audio_sfx_file = f"sfx_scene_{scene_index:03d}_{RUN_DATE}.wav"

        print(f"\n🎬 [{scene_index}/{total_scenes}] LTX...")
        try:
            clear_memory()
            image_input = load_image(image_file).resize((512, 288))
            motion_prompt = vid_prompt if vid_prompt and vid_prompt.lower() not in ["nan", "none"] else "smooth cinematic movement, gentle wind"

            video_frames = ltx_pipe(
                image=image_input,
                prompt=motion_prompt,
                negative_prompt=neg_prompt if neg_prompt and neg_prompt.lower() not in ["nan", "none"] else None,
                width=512,
                height=288,
                num_frames=20,
                num_inference_steps=15,
                generator=torch.Generator("cpu").manual_seed(42 + scene_index),
            ).frames[0]

            export_to_video(video_frames, raw_video_file, fps=12)
            print(f"   ✅ Video thô: {raw_video_file}")
            del video_frames
            clear_memory()
        except Exception as e:
            print(f"❌ Lỗi LTX cảnh {scene_index}: {e}")
            clear_memory()
            continue

        # ----- 1. Lồng tiếng Việt (TTS) -----
        a_dur = 2.0
        has_tts = False
        if dialogue_text:
            voice = pick_voice(char_name)
            print(f"🎙️ TTS [{char_name}] ({voice}): {dialogue_text[:50]}...")
            communicate = edge_tts.Communicate(text=dialogue_text, voice=voice)
            await communicate.save(audio_tts_file)
            a_dur = get_media_duration(audio_tts_file)
            has_tts = True

        # ----- 2. Sinh Hiệu Ứng Âm Thanh (SFX) -----
        has_sfx = False
        if sfx_prompt:
            has_sfx = generate_ai_sfx(sfx_prompt, duration_sec=a_dur, output_path=audio_sfx_file)

        # ----- 3. Ghép & Căn chỉnh Thời lượng Video với TTS + SFX -----
        v_dur = get_media_duration(raw_video_file)
        pad_dur = max(0.0, a_dur - v_dur) if has_tts else 0.0

        # Xử lý Filter ffmpeg trộn Voice + SFX
        inputs_str = f'-i "{raw_video_file}" '
        filter_complex = []
        
        if pad_dur > 0:
            filter_complex.append(f"[0:v]tpad=stop_mode=clone:stop_duration={pad_dur:.3f}[v]")
            map_v = "[v]"
        else:
            map_v = "0:v:0"

        if has_tts and has_sfx:
            inputs_str += f'-i "{audio_tts_file}" -i "{audio_sfx_file}" '
            filter_complex.append("[1:a]volume=1.0[tts];[2:a]volume=0.5[sfx];[tts][sfx]amix=inputs=2:duration=first[a]")
            map_a = "[a]"
        elif has_tts:
            inputs_str += f'-i "{audio_tts_file}" '
            map_a = "1:a:0"
        elif has_sfx:
            inputs_str += f'-i "{audio_sfx_file}" '
            map_a = "1:a:0"
        else:
            inputs_str += '-f lavfi -i anullsrc=r=44100:cl=stereo '
            map_a = "1:a:0"

        filter_str = f'-filter_complex "{";".join(filter_complex)}"' if filter_complex else ""
        
        mix_cmd = (
            f'ffmpeg -y {inputs_str} {filter_str} '
            f'-map {map_v} -map {map_a} -c:v libx264 -pix_fmt yuv420p -r 12 '
            f'-c:a aac -ar 44100 -ac 2 -b:a 192k -shortest "{final_scene_file}"'
        )
        subprocess.run(mix_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Dọn dẹp file tạm cảnh
        for f in [raw_video_file, audio_tts_file, audio_sfx_file]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

        if os.path.exists(final_scene_file) and os.path.getsize(final_scene_file) > 10000:
            upload_file_to_drive(final_scene_file, DRIVE_FOLDER_ID)
            rendered_files.append(final_scene_file)
            print(f"✅ Cảnh {scene_index} xong ({time.time() - scene_start:.1f}s)")
        else:
            print(f"⚠️ Cảnh {scene_index}: file video lỗi")

    del ltx_pipe
    clear_memory()

    # ========== GIAI ĐOẠN 3: GỘP + TRỘN BGM ==========
    print("\n🎞️ [3] Gộp tất cả cảnh thành 1 video...")
    if not rendered_files:
        send_n8n_webhook("failed", error_message="Không render được cảnh nào", extra=project_info)
        sys.exit(1)

    rendered_files = sorted(rendered_files, key=lambda x: x)
    with open("file_list.txt", "w", encoding="utf-8") as f:
        for file in rendered_files:
            f.write(f"file '{file}'\n")

    concat_output = f"final_concat_{RUN_DATE}.mp4"
    final_output = f"final_movie_{RUN_DATE}.mp4"

    subprocess.run(
        f'ffmpeg -y -f concat -safe 0 -i file_list.txt -c copy "{concat_output}"',
        shell=True, check=True,
    )

    if os.path.exists(BGM_FILE):
        print("🎵 Ghép nhạc nền AI vào toàn bộ phim...")
        bgm_cmd = (
            f'ffmpeg -y -i "{concat_output}" -stream_loop -1 -i "{BGM_FILE}" '
            f'-filter_complex "[0:a]volume=1.0[a_tts];[1:a]volume=0.15[a_bgm];'
            f'[a_tts][a_bgm]amix=inputs=2:duration=first[a]" '
            f'-map 0:v:0 -map "[a]" -c:v copy -c:a aac -ar 44100 -ac 2 -b:a 192k "{final_output}"'
        )
        try:
            subprocess.run(bgm_cmd, shell=True, check=True)
        except Exception as e:
            print(f"⚠️ Ghép BGM lỗi, dùng bản không BGM: {e}")
            final_output = concat_output
    else:
        print("⚠️ Không có file BGM → bỏ qua nhạc nền")
        final_output = concat_output

    print(f"\n☁️ Upload phim hoàn chỉnh: {final_output}")
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
# 6. RUN
# -------------------------------------------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(process_video_pipeline())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(process_video_pipeline())
