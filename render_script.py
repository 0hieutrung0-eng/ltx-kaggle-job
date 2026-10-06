# ============================================================
# PIPELINE KAGGLE / GITHUB
# - Nhận drive_folder_id từ n8n (sau khi tạo xong 100 ảnh cảnh)
# - Chỉ đọc nội dung chương + lời thoại từ Google Sheet
# - Tải ảnh scene_001.png ... từ đúng folder Drive
# - LTX Image-to-Video từng cảnh
# - TTS tiếng Việt (dialogue) + ghép + BGM + narration
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
        "safetensors": "safetensors",
        "pandas": "pandas",
        "requests": "requests",
        "PIL": "Pillow",
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
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--no-warn-script-location"]
        + missing
        + ["protobuf<6.0.0,>=3.20.2", "imageio-ffmpeg"]
    )
    print("✅ Cài xong")

install_requirements()

import nest_asyncio
nest_asyncio.apply()

import torch
import pandas as pd
import requests
import scipy.io.wavfile as wavfile
import edge_tts
from diffusers import LTXImageToVideoPipeline
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

# ==================== CONFIG ====================
N8N_WEBHOOK_URL = os.environ.get(
    "N8N_WEBHOOK_URL",
    "https://n8n-latest-namx.onrender.com/webhook/kaggle-video-done",
)

SHEET_ID = os.environ.get(
    "SHEET_ID", "1DmA-yuPwDl1riceSMGzWPXhuxrL4y987lOZ6Af351l8"
)
GID_SCENES = os.environ.get("GID_SCENES", "0")

# Folder ngày do n8n tạo và truyền vào
DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID", "").strip()

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN.startswith("hf_"):
    try:
        login(token=HF_TOKEN)
        print("🔑 HF OK")
    except Exception as e:
        print(f"⚠️ HF: {e}")
else:
    print("⚠️ Thiếu HF_TOKEN")

OAUTH_CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID", "")
OAUTH_CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET", "")
OAUTH_REFRESH_TOKEN = os.environ.get("OAUTH_REFRESH_TOKEN", "")
TOKEN_URI = "https://oauth2.googleapis.com/token"
USE_DRIVE = bool(OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET and OAUTH_REFRESH_TOKEN)
print("✅ Drive ON" if USE_DRIVE else "⚠️ LOCAL ONLY")

VOICE_MAP = {
    "nam": "vi-VN-NamMinhNeural",
    "nu": "vi-VN-HoaiMyNeural"
}

BGM_FILE = f"bgm_{RUN_DATE}.mp3"
VIDEO_W, VIDEO_H = 640, 360
NUM_FRAMES = 25
FPS = 16
LTX_STEPS = 20

_ltx_pipe = None

# ==================== HELPERS ====================
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
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

def prepare_offload_pipe(pipe):
    if DEVICE != "cuda":
        return pipe.to(DEVICE)
    try:
        pipe.enable_sequential_cpu_offload()
    except Exception:
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pipe = pipe.to(DEVICE)
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
            f'ffprobe -v error -show_entries format=duration '
            f'-of default=noprint_wrappers=1:nokey=1 "{path}"',
            shell=True, capture_output=True, text=True, check=True,
        )
        return float(r.stdout.strip())
    except Exception:
        return 2.0

def list_files_in_folder(folder_id):
    if not USE_DRIVE or not folder_id:
        return []
    try:
        service = build("drive", "v3", credentials=get_oauth_credentials(), cache_discovery=False)
        files, token = [], None
        while True:
            resp = (
                service.files()
                .list(
                    q=f"'{folder_id}' in parents and trashed=false",
                    fields="nextPageToken, files(id, name, mimeType)",
                    pageSize=200,
                    pageToken=token,
                )
                .execute()
            )
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
    if not USE_DRIVE or not folder_id or not os.path.exists(file_path):
        return None
    name = os.path.basename(file_path)
    mime = "video/mp4"
    if name.endswith((".png", ".jpg", ".jpeg", ".webp")):
        mime = "image/png"
    if name.endswith(".mp3"):
        mime = "audio/mpeg"
    for i in range(1, retries + 1):
        try:
            service = build("drive", "v3", credentials=get_oauth_credentials(), cache_discovery=False)
            media = MediaFileUpload(file_path, mimetype=mime, resumable=True)
            up = (
                service.files()
                .create(
                    body={"name": name, "parents": [folder_id]},
                    media_body=media,
                    fields="id",
                )
                .execute()
            )
            print(f"☁️ {name}")
            return up.get("id")
        except Exception as e:
            print(f"⚠️ upload {i}: {e}")
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
        "drive_folder_id": DRIVE_FOLDER_ID,
    }
    if extra:
        payload.update(extra)
    try:
        r = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=30)
        print(f"📡 [{status}] → {r.status_code}")
    except Exception as e:
        print(f"❌ webhook: {e}")

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

def resolve_drive_folder_id(df_scenes) -> str:
    """Ưu tiên: ENV từ n8n → cột sheet drive_folder_id"""
    global DRIVE_FOLDER_ID
    if DRIVE_FOLDER_ID:
        print(f"📁 Folder từ n8n (ENV): {DRIVE_FOLDER_ID}")
        return DRIVE_FOLDER_ID

    if df_scenes is not None and not df_scenes.empty:
        fid = safe_get(df_scenes.iloc[0], "drive_folder_id", "folder_id")
        if fid:
            DRIVE_FOLDER_ID = fid
            print(f"📁 Folder từ Sheet: {DRIVE_FOLDER_ID}")
            return fid

    raise SystemExit(
        "❌ Thiếu drive_folder_id.\n"
        "→ n8n phải truyền ENV DRIVE_FOLDER_ID\n"
        "→ hoặc thêm cột drive_folder_id trên Sheet (dòng 1)"
    )

def pick_voice(char_name: str) -> str:
    name = (char_name or "").lower()
    if any(k in name for k in ["nữ", "cô", "chị", "muội", "my", "linh", "nhi", "lan", "hoa", "nương"]):
        return VOICE_MAP["nu"]
    return VOICE_MAP["nam"]

def find_scene_image_on_drive(drive_files, scene_index: int):
    patterns = [
        f"scene_{scene_index:03d}",
        f"scene_{scene_index}",
        f"image_{scene_index:03d}",
        f"image_{scene_index}",
    ]
    for f in drive_files:
        name = f["name"].lower()
        if not name.endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue
        for p in patterns:
            if p in name:
                return f
    return None

def get_ltx_pipe():
    global _ltx_pipe
    if _ltx_pipe is None:
        print("📦 Load LTX Image-to-Video...")
        clear_memory()
        _ltx_pipe = LTXImageToVideoPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            torch_dtype=torch.bfloat16,
            token=HF_TOKEN if HF_TOKEN.startswith("hf_") else None,
            low_cpu_mem_usage=True,
        )
        _ltx_pipe = prepare_offload_pipe(_ltx_pipe)
        print("✅ LTX ready")
    return _ltx_pipe

def generate_ai_bgm(prompt_text, duration_sec, output_path):
    print("🎵 BGM...")
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

        subprocess.run(
            ["ffmpeg", "-y", "-i", wav, "-acodec", "libmp3lame", output_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
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

# ==================== MAIN ====================
async def process_video_pipeline():
    global _ltx_pipe, DRIVE_FOLDER_ID
    rendered_files = []
    scene_image_paths = {}

    print("\n📊 [1] Đọc Google Sheet (chỉ lấy chương + lời thoại)...")
    df_scenes = get_sheet_csv(GID_SCENES)
    if df_scenes.empty:
        send_n8n_webhook("failed", error_message="Không đọc được Scenes từ Sheet")
        return

    total_scenes = len(df_scenes)
    print(f"✅ {total_scenes} cảnh | cột: {df_scenes.columns.tolist()}")

    # Nhận folder ID từ n8n
    folder_id = resolve_drive_folder_id(df_scenes)

    # Lấy thông tin dự án (nếu có)
    project_info = {}
    first = df_scenes.iloc[0]
    for key in ["title", "genre", "visual_style", "world_setting", "bgm_prompt", "narration", "folder_name"]:
        val = safe_get(first, key)
        if val:
            project_info[key] = val

    print(f"\n📥 [2] Liệt kê file trong folder Drive: {folder_id}")
    drive_files = list_files_in_folder(folder_id)
    print(f"✅ Drive: {len(drive_files)} file")
    for f in drive_files[:10]:
        print(f"   - {f['name']}")

    # ----- Tải ảnh cảnh từ Drive -----
    print("\n🖼️  [3] Tải ảnh cảnh từ Drive...")
    for index, row in df_scenes.iterrows():
        scene_index = int(row.get("scene_index")) if pd.notna(row.get("scene_index")) else index + 1
        local_img = f"scene_{scene_index:03d}.png"

        if os.path.exists(local_img) and os.path.getsize(local_img) > 1000:
            scene_image_paths[scene_index] = local_img
            print(f"⏩ Cảnh {scene_index}: đã có local")
            continue

        # Ưu tiên tìm trên Drive
        found = find_scene_image_on_drive(drive_files, scene_index)
        if found:
            download_drive_file(found["id"], local_img)
            if os.path.exists(local_img) and os.path.getsize(local_img) > 1000:
                scene_image_paths[scene_index] = local_img
                print(f"✅ Cảnh {scene_index}: Drive ({found['name']})")
                continue

        print(f"❌ Cảnh {scene_index}: không tìm thấy ảnh")

    print(f"✅ Có ảnh: {len(scene_image_paths)}/{total_scenes}")

    if not scene_image_paths:
        send_n8n_webhook("failed", error_message="Không có ảnh cảnh trong folder Drive", extra=project_info)
        return

    # ----- BGM -----
    bgm_desc = project_info.get("bgm_prompt") or f"{project_info.get('genre', 'dramatic')} cinematic background music"
    generate_ai_bgm(bgm_desc, duration_sec=max(len(scene_image_paths) * 4, 30), output_path=BGM_FILE)

    # ----- Load LTX -----
    print("\n🎬 [4] Load LTX Image-to-Video...")
    try:
        ltx_pipe = get_ltx_pipe()
    except Exception as e:
        send_n8n_webhook("failed", error_message=f"LTX load lỗi: {e}")
        return

    # ----- Tạo video từng cảnh -----
    print("\n🎬 [5] Tạo video từng cảnh...")
    for index, row in df_scenes.iterrows():
        t0 = time.time()
        scene_index = int(row.get("scene_index")) if pd.notna(row.get("scene_index")) else index + 1
        image_file = scene_image_paths.get(scene_index)

        if not image_file or not os.path.exists(image_file):
            continue

        final_scene_file = f"scene_{scene_index:03d}_{RUN_DATE}.mp4"

        # Bỏ qua nếu đã có
        existing = list(Path(".").glob(f"scene_{scene_index:03d}_*.mp4"))
        if existing and os.path.getsize(str(existing[0])) > 15000:
            rendered_files.append(str(existing[0]))
            print(f"⏩ Cảnh {scene_index}: đã có sẵn")
            continue

        # Chỉ lấy dữ liệu cần thiết từ Sheet
        vid_prompt = safe_get(row, "video_prompt", "camera_motion", default="smooth cinematic camera movement")
        neg_prompt = safe_get(row, "negative_prompt")
        dialogue_text = safe_get(row, "dialogue")          # lời thoại
        char_name = safe_get(row, "character_name")
        chapter = safe_get(row, "chapter")                 # chương

        raw_video = f"raw_{scene_index:03d}.mp4"
        tts_file = f"tts_{scene_index:03d}.mp3"

        print(f"\n🎬 [{scene_index}/{total_scenes}] {chapter} | {char_name[:40] if char_name else ''}")

        # Image → Video
        try:
            clear_memory()
            img = load_image(image_file).resize((VIDEO_W, VIDEO_H))
            frames = ltx_pipe(
                image=img,
                prompt=vid_prompt,
                negative_prompt=neg_prompt or None,
                width=VIDEO_W,
                height=VIDEO_H,
                num_frames=NUM_FRAMES,
                num_inference_steps=LTX_STEPS,
                generator=torch.Generator("cpu").manual_seed(42 + scene_index),
            ).frames[0]
            export_to_video(frames, raw_video, fps=FPS)
            del frames, img
            clear_memory()
            print("   ✅ video thô")
        except Exception as e:
            print(f"❌ LTX {scene_index}: {e}")
            clear_memory()
            continue

        # TTS lời thoại
        has_tts = False
        a_dur = 2.0
        if dialogue_text:
            try:
                await edge_tts.Communicate(
                    text=dialogue_text,
                    voice=pick_voice(char_name)
                ).save(tts_file)
                a_dur = get_media_duration(tts_file)
                has_tts = True
                print(f"   🗣️ TTS {a_dur:.1f}s")
            except Exception as e:
                print(f"   ⚠️ TTS: {e}")

        # Ghép video + audio
        v_dur = get_media_duration(raw_video)
        pad = max(0.0, a_dur - v_dur) if has_tts else 0.0

        inputs = [f'-i "{raw_video}"']
        filters, map_v, map_a = [], "0:v:0", "1:a:0"

        if pad > 0.05:
            filters.append(f"[0:v]tpad=stop_mode=clone:stop_duration={pad:.3f}[v]")
            map_v = "[v]"

        if has_tts:
            inputs.append(f'-i "{tts_file}"')
        else:
            inputs.append("-f lavfi -i anullsrc=r=44100:cl=stereo")

        fc = f'-filter_complex "{";".join(filters)}"' if filters else ""

        subprocess.run(
            f'ffmpeg -y {" ".join(inputs)} {fc} -map {map_v} -map {map_a} '
            f'-c:v libx264 -pix_fmt yuv420p -r {FPS} -c:a aac -ar 44100 -ac 2 '
            f'-b:a 192k -shortest "{final_scene_file}"',
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        for f in [raw_video, tts_file]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

        if os.path.exists(final_scene_file) and os.path.getsize(final_scene_file) > 15000:
            upload_file_to_drive(final_scene_file, folder_id)
            rendered_files.append(final_scene_file)
            print(f"✅ Cảnh {scene_index} xong ({time.time() - t0:.1f}s)")
        else:
            print(f"⚠️ Cảnh {scene_index}: file lỗi")

    _ltx_pipe = None
    clear_memory()

    # ----- Ghép toàn bộ -----
    print("\n🎞️ [6] Ghép toàn bộ video...")
    if not rendered_files:
        send_n8n_webhook("failed", error_message="Không có video nào được tạo", extra=project_info)
        return

    rendered_files = sorted(rendered_files)
    with open("file_list.txt", "w", encoding="utf-8") as f:
        for p in rendered_files:
            f.write(f"file '{p}'\n")

    concat_out = f"final_concat_{RUN_DATE}.mp4"
    final_out = f"final_movie_{RUN_DATE}.mp4"

    subprocess.run(
        f'ffmpeg -y -f concat -safe 0 -i file_list.txt -c copy "{concat_out}"',
        shell=True, check=True,
    )

    # Thêm BGM
    if os.path.exists(BGM_FILE):
        try:
            subprocess.run(
                f'ffmpeg -y -i "{concat_out}" -stream_loop -1 -i "{BGM_FILE}" '
                f'-filter_complex "[0:a]volume=1.0[a1];[1:a]volume=0.12[a2];'
                f'[a1][a2]amix=inputs=2:duration=first[a]" '
                f'-map 0:v:0 -map "[a]" -c:v copy -c:a aac -ar 44100 -ac 2 '
                f'-b:a 192k "{final_out}"',
                shell=True, check=True,
            )
        except Exception:
            final_out = concat_out
    else:
        final_out = concat_out

    # Lời kể (narration)
    narration = project_info.get("narration", "")
    if narration and len(narration) > 20:
        print("🗣️ Thêm lời kể tiếng Việt...")
        narr_mp3 = f"narration_{RUN_DATE}.mp3"
        try:
            await edge_tts.Communicate(text=narration, voice=VOICE_MAP["nam"]).save(narr_mp3)
            mixed = f"final_with_narration_{RUN_DATE}.mp4"
            subprocess.run(
                f'ffmpeg -y -i "{final_out}" -i "{narr_mp3}" '
                f'-filter_complex "[0:a]volume=0.35[a0];[1:a]volume=1.2[a1];'
                f'[a0][a1]amix=inputs=2:duration=first[a]" '
                f'-map 0:v -map "[a]" -c:v copy -c:a aac -b:a 192k "{mixed}"',
                shell=True, check=True,
            )
            final_out = mixed
        except Exception as e:
            print(f"⚠️ Narration: {e}")

    # Upload kết quả cuối
    drive_id = upload_file_to_drive(final_out, folder_id)

    send_n8n_webhook(
        "completed_all",
        total_scenes=len(rendered_files),
        final_file=final_out,
        drive_file_id=drive_id,
        extra=project_info,
    )

    print(f"\n🎉 HOÀN TẤT: {final_out}")
    print(f"📁 Folder Drive: {folder_id}")

if __name__ == "__main__":
    try:
        asyncio.run(process_video_pipeline())
    except RuntimeError:
        asyncio.get_event_loop().run_until_complete(process_video_pipeline())
