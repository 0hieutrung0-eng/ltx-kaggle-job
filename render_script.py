# ========== FULL PIPELINE: Agnes Image → Agnes Video 2.5 → TTS → Merge ==========
import os
import sys
import json
import time
import gc
import requests
import subprocess
import pandas as pd

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

# --- Agnes ---
AGNES_BASE = "https://apihub.agnes-ai.com"
AGNES_IMAGE_URL = f"{AGNES_BASE}/v1/images/generations"
AGNES_VIDEO_URL = f"{AGNES_BASE}/v1/videos"
AGNES_IMAGE_MODEL = "agnes-image-2.1-flash"
AGNES_VIDEO_MODEL = "agnes-video-2.5"
AGNES_SIZE = "1024x768"

# Video settings
VIDEO_SECONDS = "5"          # 4 ~ 12
VIDEO_SIZE = "720P"
VIDEO_ASPECT = "16:9"

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
        print(f"❌ Upload lỗi {res.status_code}: {res.text[:200]}")
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

def get_public_drive_url(file_id: str) -> str | None:
    """Tạo link công khai tạm thời để Agnes lấy được ảnh"""
    token = get_google_access_token()
    if not token:
        return None
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        # Bật quyền anyone with link
        requests.post(
            f"https://www.googleapis.com/drive/v3/files/{file_id}/permissions",
            headers=headers,
            json={"role": "reader", "type": "anyone"},
            timeout=15,
        )
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    except Exception as e:
        print(f"⚠️ Public link: {e}")
        return None

# ==========================================
# SHEET & AGNES IMAGE
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

def call_agnes_image(prompt: str, retries: int = 3):
    if not AGNES_API_KEY:
        print("❌ Thiếu AGNES_API_KEY")
        return None
    headers = {
        "Authorization": f"Bearer {AGNES_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": AGNES_IMAGE_MODEL,
        "prompt": prompt,
        "size": AGNES_SIZE,
        "extra_body": {"response_format": "url"},
    }
    for attempt in range(retries + 1):
        try:
            r = requests.post(AGNES_IMAGE_URL, headers=headers, json=body, timeout=120)
            if r.status_code == 200:
                data = r.json().get("data", [])
                if data and data[0].get("url"):
                    return data[0]["url"]
            print(f"⚠️ Agnes Image {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"❌ Agnes Image attempt {attempt+1}: {e}")
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
# AGNES VIDEO 2.5
# ==========================================
def create_agnes_video(prompt: str, image_url: str = None):
    """Tạo task video bằng Agnes Video 2.5"""
    headers = {
        "Authorization": f"Bearer {AGNES_API_KEY}",
        "Content-Type": "application/json",
    }

    body = {
        "model": AGNES_VIDEO_MODEL,
        "prompt": prompt,
        "seconds": VIDEO_SECONDS,
        "size": VIDEO_SIZE,
        "aspect_ratio": VIDEO_ASPECT,
        "mode": "keyframe" if image_url else "text",
    }

    if image_url:
        body["first_frame"] = image_url

    try:
        r = requests.post(AGNES_VIDEO_URL, headers=headers, json=body, timeout=60)
        if r.status_code in (200, 201):
            data = r.json()
            video_id = data.get("video_id") or data.get("id")
            return video_id
        print(f"❌ Tạo video task lỗi {r.status_code}: {r.text[:300]}")
    except Exception as e:
        print(f"❌ Create video: {e}")
    return None

def poll_agnes_video(video_id: str, max_wait: int = 600):
    """Poll kết quả video"""
    headers = {"Authorization": f"Bearer {AGNES_API_KEY}"}
    url = f"{AGNES_BASE}/agnesapi"
    params = {"video_id": video_id, "model_name": AGNES_VIDEO_MODEL}

    start = time.time()
    while time.time() - start < max_wait:
        try:
            r = requests.get(url, headers=headers, params=params, timeout=30)
            if r.status_code == 200:
                data = r.json()
                status = str(data.get("status", "")).lower()
                progress = data.get("progress", 0)
                print(f"   ⏳ status={status} | progress={progress}%")

                if status in ("completed", "succeeded", "success", "done"):
                    return data.get("url") or data.get("video_url")
                if status in ("failed", "error", "cancelled"):
                    print(f"❌ Video failed: {data}")
                    return None
            time.sleep(5)
        except Exception as e:
            print(f"⚠️ Poll: {e}")
            time.sleep(5)
    print("❌ Timeout chờ video")
    return None

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
        url = call_agnes_image(prompt)
        if url and download_url(url, local):
            upload_file_to_drive(local)
            ok += 1
            print(f"✅ [{scene_num}] Agnes Image OK")
        else:
            print(f"❌ [{scene_num}] Không tạo được ảnh")

    print(f"\n✅ Giai đoạn 1 xong: {ok} ảnh")
    return ok >= max(1, int(min(TOTAL_SCENES, len(df)) * 0.7))

# ==========================================
# TTS + POST PROCESS
# ==========================================
def get_video_prompt(row) -> str:
    vp = safe_get(row, "video_prompt")
    if vp:
        return vp[:300]
    ip = safe_get(row, "image_prompt", "scene_description")
    if ip:
        return f"Slow cinematic camera move, subtle natural motion, consistent characters. {ip[:200]}"
    return "Slow cinematic camera push-in, subtle character movement, natural atmosphere"

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
            subprocess.run(cmd, shell=True, timeout=90,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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

# ==========================================
# GIAI ĐOẠN 2: RENDER BẰNG AGNES VIDEO 2.5
# ==========================================
def step_2_agnes_video_render():
    print("\n" + "=" * 55)
    print(f"🎬 GIAI ĐOẠN 2: Agnes Video 2.5 | {VIDEO_SECONDS}s | {VIDEO_SIZE}")
    print("=" * 55)

    df = get_sheet_csv(GID_SCENES)
    drive_map = get_drive_files_map()

    # Tải ảnh thiếu
    for i in range(1, TOTAL_SCENES + 1):
        local = f"{OUTPUT_DIR}/scene_{i:03d}.png"
        name = f"scene_{i:03d}.png"
        if not (os.path.exists(local) and os.path.getsize(local) > 2000):
            if name in drive_map:
                download_from_drive(drive_map[name], local)

    rendered = []
    t0 = time.time()

    for idx, row in df.iterrows():
        scene_num = int(row["scene_index"]) if pd.notna(row.get("scene_index")) else idx + 1
        if scene_num > TOTAL_SCENES:
            break

        img_path = f"{OUTPUT_DIR}/scene_{scene_num:03d}.png"
        raw_video = f"{VIDEO_DIR}/agnes_raw_{scene_num:03d}.mp4"
        voice_mp3 = f"{VIDEO_DIR}/voice_{scene_num:03d}.mp3"
        final_scene = f"{VIDEO_DIR}/scene_{scene_num:03d}.mp4"
        scene_name = f"scene_{scene_num:03d}.mp4"

        if os.path.exists(final_scene) and os.path.getsize(final_scene) > 400_000:
            print(f"⏩ [{scene_num}] Đã có local")
            rendered.append(final_scene)
            continue

        if scene_name in drive_map:
            if download_from_drive(drive_map[scene_name], final_scene):
                if os.path.getsize(final_scene) > 400_000:
                    print(f"⏩ [{scene_num}] Tải từ Drive")
                    rendered.append(final_scene)
                    continue

        if not (os.path.exists(img_path) and os.path.getsize(img_path) > 2000):
            print(f"⚠️ [{scene_num}] Thiếu ảnh → bỏ qua")
            continue

        prompt = get_video_prompt(row)
        dialogue = get_dialogue(row)
        print(f"🎥 [{scene_num}/{TOTAL_SCENES}] {prompt[:70]}...")

        # Lấy public URL của ảnh
        img_name = f"scene_{scene_num:03d}.png"
        image_url = None
        if img_name in drive_map:
            image_url = get_public_drive_url(drive_map[img_name])

        # Nếu không có public link thì dùng tạm URL từ Agnes Image (nếu còn)
        if not image_url:
            print(f"⚠️ [{scene_num}] Không có public image URL, thử text mode")

        try:
            t1 = time.time()
            video_id = create_agnes_video(prompt, image_url)
            if not video_id:
                print(f"❌ [{scene_num}] Không tạo được task")
                continue

            print(f"   📡 video_id = {video_id}")
            video_url = poll_agnes_video(video_id)

            if not video_url:
                print(f"❌ [{scene_num}] Không lấy được video")
                continue

            if not download_url(video_url, raw_video):
                print(f"❌ [{scene_num}] Tải video thất bại")
                continue

            make_tts(dialogue, voice_mp3)
            out_file = mix_video_audio(raw_video, voice_mp3, final_scene)

            sz = os.path.getsize(out_file) if os.path.exists(out_file) else 0
            print(f"✅ [{scene_num}] {time.time()-t1:.0f}s | {sz/1024:.0f} KB")

            if sz > 100_000:
                upload_file_to_drive(out_file)
                rendered.append(out_file)

            if os.path.exists(raw_video) and raw_video != out_file:
                try:
                    os.remove(raw_video)
                except Exception:
                    pass

        except Exception as e:
            print(f"❌ [{scene_num}] Lỗi: {e}")

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

    if step_1_ensure_images():
        step_2_agnes_video_render()
        print("\n🎉 HOÀN TẤT TOÀN BỘ QUY TRÌNH!")
    else:
        print("\n❌ Dừng: chưa đủ ảnh ở Giai đoạn 1")
