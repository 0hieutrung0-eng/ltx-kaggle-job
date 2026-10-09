import os
import sys
import json
import time
import gc
import requests
import subprocess
import pandas as pd
import torch

# ==========================================
# CẤU HÌNH
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

# LTX — an toàn T4 16GB
LTX_MODEL = "Lightricks/LTX-Video"
LTX_WIDTH = 768
LTX_HEIGHT = 512
LTX_NUM_FRAMES = 49
LTX_STEPS = 20
LTX_FPS = 24
LTX_NEG = "worst quality, inconsistent motion, blurry, jittery, distorted, morphing, text, watermark"

AGNES_URL = "https://apihub.agnes-ai.com/v1/images/generations"
AGNES_MODEL = "agnes-image-2.0-flash"
AGNES_SIZE = "1024x768"

# ==========================================
# GOOGLE OAUTH + DRIVE
# ==========================================
def get_google_access_token():
    if not all([CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN]):
        print("❌ Thiếu OAUTH credentials")
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
        print(f"❌ Upload {res.status_code}: {res.text[:250]}")
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
        print(f"❌ Sheet gid={gid}: {e}")
        return pd.DataFrame()

def call_agnes(prompt: str):
    if not AGNES_API_KEY:
        return None
    body = {
        "model": AGNES_MODEL,
        "prompt": prompt,
        "size": AGNES_SIZE,
        "extra_body": {"response_format": "url"},
    }
    headers = {"Authorization": f"Bearer {AGNES_API_KEY}", "Content-Type": "application/json"}
    try:
        r = requests.post(AGNES_URL, headers=headers, json=body, timeout=120)
        if r.status_code == 200:
            return r.json().get("data", [{}])[0].get("url")
    except Exception as e:
        print(f"❌ Agnes: {e}")
    return None

def download_url(url: str, path: str) -> bool:
    try:
        r = requests.get(url, timeout=90)
        if r.status_code == 200 and len(r.content) > 2000:
            open(path, "wb").write(r.content)
            return True
    except Exception as e:
        print(f"❌ DL url: {e}")
    return False

# ==========================================
# GIAI ĐOẠN 1
# ==========================================
def step_1_ensure_images():
    print("\n" + "=" * 50)
    print("🔍 GIAI ĐOẠN 1: Đủ ảnh scene trên Drive")
    print("=" * 50)

    if not DRIVE_FOLDER_ID:
        print("❌ THIẾU DRIVE_FOLDER_ID")
        return False

    drive_map = get_drive_files_map()
    df_scenes = get_sheet_csv(GID_SCENES)
    if df_scenes.empty:
        print("❌ Không đọc sheet cảnh")
        return False

    ok = 0
    for idx, row in df_scenes.iterrows():
        scene_num = int(row.get("scene_index")) if pd.notna(row.get("scene_index")) else idx + 1
        name = f"scene_{scene_num:03d}.png"
        local = f"{OUTPUT_DIR}/{name}"

        if name in drive_map:
            if not (os.path.exists(local) and os.path.getsize(local) > 2000):
                download_from_drive(drive_map[name], local)
            ok += 1
            print(f"✅ [{scene_num}] Drive OK")
            continue

        if os.path.exists(local) and os.path.getsize(local) > 2000:
            upload_file_to_drive(local)
            ok += 1
            continue

        prompt = safe_get(row, "image_prompt", "scene_description") or f"Scene {scene_num}"
        print(f"🎨 [{scene_num}] Agnes...")
        url = call_agnes(prompt)
        if url and download_url(url, local):
            upload_file_to_drive(local)
            ok += 1
        else:
            print(f"❌ [{scene_num}] thiếu ảnh")

    print(f"✅ Giai đoạn 1: {ok}/{len(df_scenes)} ảnh")
    return ok >= max(1, int(len(df_scenes) * 0.8))

# ==========================================
# GIAI ĐOẠN 2: LTX 1 GPU
# ==========================================
def install_ltx_deps():
    """Cài nhẹ — không đụng PyTorch / CUDA."""
    print("📦 Cài nhẹ LTX...")

    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torchao", "diffusers"],
        check=False,
    )
    subprocess.run(
        "rm -rf /usr/local/lib/python3.13/dist-packages/diffusers "
        "/usr/local/lib/python3.13/dist-packages/diffusers-*.dist-info",
        shell=True,
        check=False,
    )

    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q",
            "--no-cache-dir", "--no-deps",
            "diffusers==0.32.2",
        ],
        check=False,
    )
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
            "transformers==4.46.3",
            "safetensors", "sentencepiece", "imageio", "imageio-ffmpeg", "edge-tts",
        ],
        check=False,
    )

    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception:
        subprocess.run(
            "apt-get update -qq && apt-get install -y -qq ffmpeg",
            shell=True,
            check=False,
        )
    print("✅ Cài package xong")

def _patch_imports():
    """Vá thiếu symbol trước khi import LTX."""
    for k in list(sys.modules.keys()):
        if (
            k == "diffusers"
            or k.startswith("diffusers.")
            or k == "transformers"
            or k.startswith("transformers.")
        ):
            del sys.modules[k]

    import transformers.utils as tu
    if not hasattr(tu, "FLAX_WEIGHTS_NAME"):
        tu.FLAX_WEIGHTS_NAME = "flax_model.msgpack"
    if not hasattr(tu, "SAFE_WEIGHTS_NAME"):
        tu.SAFE_WEIGHTS_NAME = "model.safetensors"
    if not hasattr(tu, "SAFE_WEIGHTS_INDEX_NAME"):
        tu.SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"

    import diffusers.utils as du
    for name, fn in [
        ("is_flax_available", lambda: False),
        ("is_bs4_available", lambda: False),
        ("is_ftfy_available", lambda: False),
    ]:
        if not hasattr(du, name):
            setattr(du, name, fn)

def load_ltx_pipe():
    _patch_imports()

    from diffusers import LTXImageToVideoPipeline
    from diffusers.utils import logging

    logging.set_verbosity_error()
    print(f"🧠 Load LTX: {LTX_MODEL}")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipe = LTXImageToVideoPipeline.from_pretrained(LTX_MODEL, torch_dtype=dtype)

    if torch.cuda.is_available():
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pipe.to("cuda")
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass

    print("✅ LTX pipeline sẵn sàng")
    return pipe

def get_video_prompt(row) -> str:
    vp = safe_get(row, "video_prompt")
    if vp:
        return vp
    ip = safe_get(row, "image_prompt", "scene_description")
    if ip:
        return f"Cinematic camera motion, subtle character movement, natural atmosphere. {ip}"
    return "Cinematic slow camera push-in, subtle natural motion, high quality"

def get_dialogue(row) -> str:
    return safe_get(row, "dialogue")[:180]

def make_tts(text: str, out_mp3: str) -> bool:
    if not text or len(text.strip()) < 2:
        return False
    if os.path.exists(out_mp3) and os.path.getsize(out_mp3) > 400:
        return True
    safe = text.replace('"', "'").replace("\n", " ")
    try:
        subprocess.run(
            f'edge-tts --text "{safe}" --voice vi-VN-NamMinhNeural --rate=+5% --write-media "{out_mp3}"',
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
        return os.path.exists(out_mp3) and os.path.getsize(out_mp3) > 400
    except Exception:
        return False

def mix_video_audio(video_in: str, audio_mp3: str, video_out: str) -> str:
    if not (os.path.exists(audio_mp3) and os.path.getsize(audio_mp3) > 400):
        return video_in
    cmd = (
        f'ffmpeg -y -hide_banner -loglevel error '
        f'-i "{video_in}" -i "{audio_mp3}" '
        f'-c:v copy -c:a aac -b:a 96k -shortest -movflags +faststart "{video_out}"'
    )
    subprocess.run(cmd, shell=True)
    if os.path.exists(video_out) and os.path.getsize(video_out) > 5000:
        return video_out
    return video_in

def step_2_ltx_render_merge():
    print("\n" + "=" * 50)
    print(f"🎬 GIAI ĐOẠN 2: LTX 1 GPU | {LTX_WIDTH}x{LTX_HEIGHT} | frames={LTX_NUM_FRAMES}")
    print("=" * 50)

    install_ltx_deps()
    pipe = load_ltx_pipe()

    from diffusers.utils import export_to_video, load_image

    df_scenes = get_sheet_csv(GID_SCENES)
    drive_map = get_drive_files_map()

    for i in range(1, TOTAL_SCENES + 1):
        local = f"{OUTPUT_DIR}/scene_{i:03d}.png"
        name = f"scene_{i:03d}.png"
        if not (os.path.exists(local) and os.path.getsize(local) > 2000):
            if name in drive_map:
                download_from_drive(drive_map[name], local)

    rendered = []
    t0 = time.time()

    for idx, row in df_scenes.iterrows():
        scene_num = int(row.get("scene_index")) if pd.notna(row.get("scene_index")) else idx + 1
        scene_name = f"scene_{scene_num:03d}.mp4"
        img_path = f"{OUTPUT_DIR}/scene_{scene_num:03d}.png"
        raw_video = f"{VIDEO_DIR}/ltx_{scene_num:03d}.mp4"
        voice_mp3 = f"{VIDEO_DIR}/voice_{scene_num:03d}.mp3"
        final_scene = f"{VIDEO_DIR}/{scene_name}"

        if os.path.exists(final_scene) and os.path.getsize(final_scene) > 5000:
            print(f"⏩ [{scene_num}] đã có local")
            rendered.append(final_scene)
            continue

        if scene_name in drive_map:
            ok_dl = download_from_drive(drive_map[scene_name], final_scene)
            if ok_dl and os.path.exists(final_scene) and os.path.getsize(final_scene) > 5000:
                print(f"⏩ [{scene_num}] đã có trên Drive")
                rendered.append(final_scene)
                continue

        if not (os.path.exists(img_path) and os.path.getsize(img_path) > 2000):
            print(f"⚠️ [{scene_num}] thiếu ảnh gốc")
            continue

        prompt = get_video_prompt(row)
        dialogue = get_dialogue(row)
        print(f"\n🎥 [{scene_num}/{TOTAL_SCENES}] LTX...")
        print(f"   prompt: {prompt[:90]}...")

        try:
            image = load_image(img_path)
            generator = torch.Generator(device="cpu").manual_seed(42 + scene_num)
            t_start = time.time()

            res = pipe(
                image=image,
                prompt=prompt,
                negative_prompt=LTX_NEG,
                width=LTX_WIDTH,
                height=LTX_HEIGHT,
                num_frames=LTX_NUM_FRAMES,
                num_inference_steps=LTX_STEPS,
                generator=generator,
            )
            frames = res.frames[0]
            export_to_video(frames, raw_video, fps=LTX_FPS)

            make_tts(dialogue, voice_mp3)
            out_file = mix_video_audio(raw_video, voice_mp3, final_scene)

            print(f"✅ [{scene_num}] {time.time() - t_start:.1f}s")
            upload_file_to_drive(out_file)
            rendered.append(out_file)

            del res, frames
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ [{scene_num}] LTX lỗi: {e}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n✅ LTX xong {len(rendered)}/{TOTAL_SCENES} trong {(time.time() - t0) / 60:.1f} phút")

    valid = [v for v in sorted(rendered) if os.path.exists(v) and os.path.getsize(v) > 5000]
    if not valid:
        print("❌ Không có video để ghép")
        return False

    concat_file = "concat_list.txt"
    with open(concat_file, "w", encoding="utf-8") as f:
        for v in valid:
            f.write(f"file '{os.path.abspath(v)}'\n")

    print("🎞️ Ghép final_full_movie.mp4...")
    subprocess.run(
        f'ffmpeg -y -hide_banner -loglevel error '
        f'-f concat -safe 0 -i "{concat_file}" -c copy -movflags +faststart "{FINAL_VIDEO}"',
        shell=True,
    )

    if not (os.path.exists(FINAL_VIDEO) and os.path.getsize(FINAL_VIDEO) > 10000):
        subprocess.run(
            f'ffmpeg -y -hide_banner -loglevel error '
            f'-f concat -safe 0 -i "{concat_file}" '
            f'-c:v libx264 -preset veryfast -crf 23 -c:a aac -b:a 96k '
            f'-movflags +faststart "{FINAL_VIDEO}"',
            shell=True,
        )

    if os.path.exists(FINAL_VIDEO) and os.path.getsize(FINAL_VIDEO) > 10000:
        mb = os.path.getsize(FINAL_VIDEO) / 1024 / 1024
        print(f"🎉 Phim: {FINAL_VIDEO} ({mb:.1f} MB)")
        upload_file_to_drive(FINAL_VIDEO)
        return True

    print("❌ Ghép phim thất bại")
    return False

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print("DRIVE_FOLDER_ID =", DRIVE_FOLDER_ID or "(EMPTY)")
    try:
        print(
            "CUDA:",
            torch.cuda.is_available(),
            "|",
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        )
    except Exception as e:
        print("Torch:", e)

    if step_1_ensure_images():
        step_2_ltx_render_merge()
        print("\n🎉 HOÀN TẤT")
    else:
        print("\n❌ Dừng: chưa đủ ảnh")
