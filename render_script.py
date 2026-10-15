import os
import sys
import json
import time
import gc
import requests
import subprocess
import pandas as pd
import glob

# ==========================================
# CÀI ĐẶT DEPENDENCY NGAY TỪ ĐẦU
# ==========================================
def install_all_deps():
    print("📦 Đang kiểm tra và cài đặt dependency cho LTX-Video...")

    try:
        import ftfy
        print("✅ ftfy đã có")
    except ImportError:
        print("📦 Đang cài ftfy...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "ftfy"], check=False)
        import ftfy
        print("✅ ftfy đã cài xong")

    try:
        from diffusers import LTXPipeline
        print("✅ diffusers + LTXPipeline đã có")
    except Exception:
        print("📦 Đang cài diffusers và các gói hỗ trợ LTX...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q",
            "diffusers>=0.32.0", "transformers>=4.46.0", "accelerate>=1.1.0",
            "sentencepiece", "imageio", "imageio-ffmpeg", "edge-tts",
            "safetensors", "huggingface_hub", "protobuf<6", "opencv-python-headless", "ftfy"
        ], check=False)

    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print("✅ ffmpeg đã có")
    except Exception:
        print("📦 Đang cài ffmpeg...")
        subprocess.run("apt-get update -qq && apt-get install -y -qq ffmpeg", shell=True, check=False)

    print("✅ Tất cả dependency đã sẵn sàng\n")

install_all_deps()

import torch
import ftfy

# ==========================================
# CẤU HÌNH LTX-VIDEO TỐI ƯU
# ==========================================
TOTAL_SCENES = 100
OUTPUT_DIR = "./output_scenes"
VIDEO_DIR = "./output_videos"
FINAL_VIDEO = "final_full_movie.mp4"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID", "").strip()
CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET", "")
REFRESH_TOKEN = os.environ.get("OAUTH_REFRESH_TOKEN", "")
AGNES_API_KEY = os.environ.get("AGNES_API_KEY", "")
SHEET_ID = os.environ.get("SHEET_ID", "1DmA-yuPwDl1riceSMGzWPXhuxrL4y987lOZ6Af351l8")
GID_SCENES = os.environ.get("GID_SCENES", "0")

# --- LTX-Video Configuration ---
DEFAULT_MODEL_ID = "Lightricks/LTX-Video"
LTX_WIDTH = 512
LTX_HEIGHT = 512
LTX_NUM_FRAMES = 49
LTX_STEPS = 25
LTX_FPS = 24
FINAL_FPS = 24
FINAL_WIDTH = 768
FINAL_HEIGHT = 512

# --- Agnes ---
AGNES_URL = "https://apihub.agnes-ai.com/v1/images/generations"
AGNES_MODEL = "agnes-image-2.1-flash"
AGNES_SIZE = "1024x768"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:32"

# ==========================================
# HÀM TỰ ĐỘNG TÌM DATASET CỤC BỘ TRÊN KAGGLE
# ==========================================
def find_local_model_path():
    print("🔍 Đang quét các dataset có sẵn trong hệ thống (/kaggle/input/)...")
    
    # Tìm kiếm tất cả các thư mục con trong kaggle input có chứa file trọng số hoặc cấu hình model
    search_patterns = [
        "/kaggle/input/**/ltx-video*",
        "/kaggle/input/**/Lightricks*",
        "/kaggle/input/**/*.safetensors"
    ]
    
    matched_dirs = set()
    for pattern in search_patterns:
        for path in glob.glob(pattern, recursive=True):
            if os.path.isfile(path):
                matched_dirs.add(os.path.dirname(path))
            elif os.path.isdir(path):
                matched_dirs.add(path)
                
    if matched_dirs:
        # Lấy thư mục hợp lệ đầu tiên tìm thấy
        local_path = list(matched_dirs)[0]
        # Nếu path trỏ tới thư mục như 'model' hoặc 'weights', kiểm tra sâu hơn chút nếu cần
        print(f"✅ Đã tìm thấy dataset cục bộ tại: {local_path}")
        return local_path
    
    print("⚠️ Không tìm thấy dataset cục bộ, sẽ sử dụng nguồn online từ Hugging Face.")
    return DEFAULT_MODEL_ID

# ==========================================
# GOOGLE OAUTH + DRIVE
# ==========================================
def get_google_access_token():
    if not all([CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN]):
        return None
    try:
        res = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "refresh_token": REFRESH_TOKEN,
                "grant_type": "refresh_token",
            },
            timeout=20,
        ).json()
        return res.get("access_token")
    except Exception as e:
        print(f"❌ OAuth: {e}")
        return None

def get_drive_files_map():
    token = get_google_access_token()
    if not token or not DRIVE_FOLDER_ID:
        return {}
    headers = {"Authorization": f"Bearer {token}"}
    query = f"'{DRIVE_FOLDER_ID}' in parents and trashed = false"
    url = (
        "https://www.googleapis.com/drive/v3/files"
        f"?q={requests.utils.quote(query)}&fields=files(id,name)&pageSize=1000"
    )
    try:
        res = requests.get(url, headers=headers, timeout=30).json()
        files = res.get("files", [])
        print(f"📁 Drive: {len(files)} file")
        return {f["name"]: f["id"] for f in files}
    except Exception as e:
        print(f"❌ Quét Drive: {e}")
        return {}

def upload_file_to_drive(file_path: str):
    token = get_google_access_token()
    if not token or not DRIVE_FOLDER_ID:
        return None
    if not os.path.exists(file_path) or os.path.getsize(file_path) < 1000:
        return None
    file_name = os.path.basename(file_path)
    metadata = {"name": file_name, "parents": [DRIVE_FOLDER_ID]}
    headers = {"Authorization": f"Bearer {token}"}
    print(f"🚀 Upload {file_name}...")
    try:
        with open(file_path, "rb") as f:
            files = {
                "data": ("metadata", json.dumps(metadata), "application/json; charset=UTF-8"),
                "file": (file_name, f, "application/octet-stream"),
            }
            res = requests.post(
                "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
                headers=headers,
                files=files,
                timeout=600,
            )
        if res.status_code in (200, 201):
            print(f"✅ Uploaded: {file_name}")
            return res.json()
    except Exception as e:
        print(f"❌ Upload: {e}")
    return None

def download_from_drive(file_id: str, save_path: str) -> bool:
    token = get_google_access_token()
    if not token:
        return False
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = requests.get(
            f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media",
            headers=headers,
            timeout=180,
        )
        if r.status_code == 200 and len(r.content) > 1000:
            with open(save_path, "wb") as f:
                f.write(r.content)
            return True
    except Exception as e:
        print(f"❌ DL Drive: {e}")
    return False

# ==========================================
# SHEET & AGNES
# ==========================================
def safe_get(row, *keys, default=""):
    for key in keys:
        val = row.get(key)
        if pd.notna(val) and str(val).strip().lower() not in ("nan", "none", "null", "", "[empty]"):
            return str(val).strip()
    return default

def get_sheet_csv(gid: str) -> pd.DataFrame:
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={gid}"
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip().str.lower()
        return df.dropna(how="all")
    except Exception as e:
        print(f"❌ Sheet gid={gid}: {e}")
        return pd.DataFrame()

def call_agnes(prompt: str, retries: int = 3):
    if not AGNES_API_KEY:
        print("❌ Thiếu AGNES_API_KEY")
        return None
    body = {
        "model": AGNES_MODEL,
        "prompt": prompt,
        "size": AGNES_SIZE,
        "extra_body": {"response_format": "url"},
    }
    headers = {
        "Authorization": f"Bearer {AGNES_API_KEY}",
        "Content-Type": "application/json",
    }
    for attempt in range(retries + 1):
        try:
            r = requests.post(AGNES_URL, headers=headers, json=body, timeout=120)
            if r.status_code == 200:
                data = r.json().get("data", [])
                if data and data[0].get("url"):
                    return data[0]["url"]
        except Exception as e:
            print(f"❌ Agnes attempt {attempt+1}: {e}")
        time.sleep(2 + attempt)
    return None

def download_url(url: str, path: str) -> bool:
    try:
        r = requests.get(url, timeout=90)
        if r.status_code == 200 and len(r.content) > 2000:
            with open(path, "wb") as f:
                f.write(r.content)
            return True
    except Exception as e:
        print(f"❌ DL url: {e}")
    return False

# ==========================================
# GIAI ĐOẠN 1: ĐẢM BẢO ĐỦ ẢNH
# ==========================================
def step_1_ensure_images():
    print("\n" + "=" * 55)
    print("🔍 GIAI ĐOẠN 1: Tạo / tải đủ ảnh scene")
    print("=" * 55)
    drive_map = get_drive_files_map()
    df = get_sheet_csv(GID_SCENES)
    if df.empty:
        print("❌ Sheet rỗng")
        return False

    ok = 0
    for idx, row in df.iterrows():
        scene_num = int(row["scene_index"]) if pd.notna(row.get("scene_index")) else idx + 1
        if scene_num > TOTAL_SCENES:
            break

        name = f"scene_{scene_num:03d}.png"
        local = f"{OUTPUT_DIR}/{name}"

        if name in drive_map:
            if not (os.path.exists(local) and os.path.getsize(local) > 2000):
                download_from_drive(drive_map[name], local)
            if os.path.exists(local) and os.path.getsize(local) > 2000:
                ok += 1
                print(f"✅ [{scene_num}] Drive OK")
                continue

        if os.path.exists(local) and os.path.getsize(local) > 2000:
            upload_file_to_drive(local)
            ok += 1
            print(f"✅ [{scene_num}] Local OK")
            continue

        prompt = safe_get(row, "image_prompt", "scene_description") or f"Cinematic scene {scene_num}"
        print(f"🎨 [{scene_num}] Đang tạo ảnh bằng Agnes...")
        url = call_agnes(prompt)
        if url and download_url(url, local):
            upload_file_to_drive(local)
            ok += 1
            print(f"✅ [{scene_num}] Agnes OK")
        else:
            print(f"❌ [{scene_num}] Không tạo được ảnh")

    print(f"\n✅ Giai đoạn 1 xong: {ok} ảnh")
    return ok >= max(1, int(min(TOTAL_SCENES, len(df)) * 0.7))

# ==========================================
# CÁC HÀM HỖ TRỢ AUDIO & VIDEO
# ==========================================
def get_video_prompt(row) -> str:
    vp = safe_get(row, "video_prompt")
    if vp:
        return vp[:220]
    ip = safe_get(row, "image_prompt", "scene_description")
    if ip:
        return f"Cinematic motion, high quality. {ip[:160]}"
    return "Cinematic motion, high quality"

def get_dialogue(row) -> str:
    return safe_get(row, "dialogue")[:180]

def make_tts(text: str, out_mp3: str) -> bool:
    if not text or len(text.strip()) < 2:
        return False
    if os.path.exists(out_mp3) and os.path.getsize(out_mp3) > 400:
        return True
    safe = text.replace('"', "'").replace("\n", " ").replace("`", "'")
    for voice in ["vi-VN-NamMinhNeural", "vi-VN-HoaiMyNeural"]:
        try:
            cmd = f'edge-tts --text "{safe}" --voice {voice} --rate=+5% --write-media "{out_mp3}"'
            subprocess.run(cmd, shell=True, timeout=90, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(out_mp3) and os.path.getsize(out_mp3) > 400:
                return True
        except Exception:
            continue
    return False

def mix_video_audio(video_in: str, audio_mp3: str, video_out: str) -> str:
    if not (os.path.exists(audio_mp3) and os.path.getsize(audio_mp3) > 400):
        if video_in != video_out and os.path.exists(video_in):
            subprocess.run(f'cp -f "{video_in}" "{video_out}"', shell=True)
        return video_out if os.path.exists(video_out) else video_in
    cmd = (
        f'ffmpeg -y -hide_banner -loglevel error '
        f'-i "{video_in}" -i "{audio_mp3}" '
        f'-c:v copy -c:a aac -b:a 96k -shortest -movflags +faststart "{video_out}"'
    )
    subprocess.run(cmd, shell=True)
    if os.path.exists(video_out) and os.path.getsize(video_out) > 5000:
        return video_out
    return video_in

def post_process_video(raw_path: str, out_path: str) -> str:
    if not os.path.exists(raw_path):
        return raw_path
    cmd = (
        f'ffmpeg -y -hide_banner -loglevel error -i "{raw_path}" '
        f'-vf "scale={FINAL_WIDTH}:{FINAL_HEIGHT}:flags=lanczos,'
        f'fps={FINAL_FPS}" '
        f'-c:v libx264 -preset fast -crf 17 -pix_fmt yuv420p '
        f'-movflags +faststart "{out_path}"'
    )
    try:
        subprocess.run(cmd, shell=True, timeout=120)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
            return out_path
    except Exception as e:
        print(f"⚠️ Post-process: {e}")
    return raw_path

def load_ltx_pipe():
    from diffusers import LTXPipeline
    from diffusers.utils import logging
    logging.set_verbosity_error()

    # Tự động tìm nguồn model cục bộ hoặc online
    model_source = find_local_model_path()
    print(f"🧠 Đang load LTX-Video từ: {model_source}...")
    dtype = torch.float16

    pipe = LTXPipeline.from_pretrained(
        model_source,
        torch_dtype=dtype,
    )

    if torch.cuda.is_available():
        pipe.enable_sequential_cpu_offload()
        print("✅ LTX sequential_cpu_offload đã bật (Tiết kiệm VRAM)")

        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
            print("✅ VAE slicing đã bật")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("✅ LTX-Video ready")
    return pipe

# ==========================================
# GIAI ĐOẠN 2: RENDER VỚI LTX
# ==========================================
def step_2_ltx_render():
    from diffusers.utils import export_to_video
    from PIL import Image

    print("\n" + "=" * 55)
    print(f"🎬 GIAI ĐOẠN 2: LTX-Video | {LTX_WIDTH}x{LTX_HEIGHT} | frames={LTX_NUM_FRAMES} | steps={LTX_STEPS}")
    print("=" * 55)

    pipe = load_ltx_pipe()

    df = get_sheet_csv(GID_SCENES)
    drive_map = get_drive_files_map()

    rendered = []
    t0 = time.time()

    for idx, row in df.iterrows():
        scene_num = int(row["scene_index"]) if pd.notna(row.get("scene_index")) else idx + 1
        if scene_num > TOTAL_SCENES:
            break

        img_path = f"{OUTPUT_DIR}/scene_{scene_num:03d}.png"
        raw_video = f"{VIDEO_DIR}/ltx_raw_{scene_num:03d}.mp4"
        upscaled = f"{VIDEO_DIR}/ltx_up_{scene_num:03d}.mp4"
        voice_mp3 = f"{VIDEO_DIR}/voice_{scene_num:03d}.mp3"
        final_scene = f"{VIDEO_DIR}/scene_{scene_num:03d}.mp4"
        scene_name = f"scene_{scene_num:03d}.mp4"

        if os.path.exists(final_scene) and os.path.getsize(final_scene) > 200_000:
            print(f"⏩ [{scene_num}] Đã có local")
            rendered.append(final_scene)
            continue

        if not (os.path.exists(img_path) and os.path.getsize(img_path) > 2000):
            print(f"⚠️ [{scene_num}] Thiếu ảnh → bỏ qua")
            continue

        prompt = get_video_prompt(row)
        dialogue = get_dialogue(row)
        print(f"🎥 [{scene_num}/{TOTAL_SCENES}] {prompt[:75]}...")

        try:
            t1 = time.time()

            image = Image.open(img_path).convert("RGB")
            image = image.resize((LTX_WIDTH, LTX_HEIGHT), Image.Resampling.LANCZOS)

            generator = torch.Generator(device="cpu").manual_seed(42 + scene_num)

            with torch.inference_mode():
                res = pipe(
                    image=image,
                    prompt=prompt,
                    negative_prompt="worst quality, inconsistent motion, blurry, jittery",
                    height=LTX_HEIGHT,
                    width=LTX_WIDTH,
                    num_frames=LTX_NUM_FRAMES,
                    num_inference_steps=LTX_STEPS,
                    generator=generator,
                )

            export_to_video(res.frames[0], raw_video, fps=LTX_FPS)
            post_process_video(raw_video, upscaled)
            make_tts(dialogue, voice_mp3)
            out_file = mix_video_audio(upscaled, voice_mp3, final_scene)

            sz = os.path.getsize(out_file) if os.path.exists(out_file) else 0
            print(f"✅ [{scene_num}] {time.time()-t1:.0f}s | {sz/1024:.0f} KB")

            if sz > 100_000:
                upload_file_to_drive(out_file)
                rendered.append(out_file)

            del res
            del image
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ [{scene_num}] Lỗi: {e}")
            gc.collect()

    print(f"\nRender xong {len(rendered)} cảnh / {(time.time()-t0)/60:.1f} phút")

    valid = [v for v in sorted(rendered) if os.path.exists(v) and os.path.getsize(v) > 100_000]
    if not valid:
        print("❌ Không đủ video để ghép")
        return False

    concat_file = "concat_list.txt"
    with open(concat_file, "w", encoding="utf-8") as f:
        for v in valid:
            safe_path = os.path.abspath(v).replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")

    print("🎞️ Đang ghép final_full_movie.mp4...")
    subprocess.run(
        f'ffmpeg -y -hide_banner -loglevel error '
        f'-f concat -safe 0 -i "{concat_file}" -c copy -movflags +faststart "{FINAL_VIDEO}"',
        shell=True,
    )

    if os.path.exists(FINAL_VIDEO):
        mb = os.path.getsize(FINAL_VIDEO) / 1024 / 1024
        print(f"🎉 Phim hoàn chỉnh: {FINAL_VIDEO} ({mb:.1f} MB)")
        upload_file_to_drive(FINAL_VIDEO)
        return True

    print("❌ Ghép phim thất bại")
    return False

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print("DRIVE_FOLDER_ID =", DRIVE_FOLDER_ID or "(EMPTY)")
    print("AGNES_API_KEY    =", "có" if AGNES_API_KEY else "(EMPTY)")
    try:
        print("CUDA:", torch.cuda.is_available(),
              torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    except Exception as e:
        print("Torch:", e)

    if step_1_ensure_images():
        step_2_ltx_render()
        print("\n🎉 HOÀN TẤT TOÀN BỘ QUY TRÌNH!")
    else:
        print("\n❌ Dừng: chưa đủ ảnh ở Giai đoạn 1")
