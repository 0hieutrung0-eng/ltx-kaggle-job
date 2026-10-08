import os
import sys
import json
import time
import gc
import requests
import subprocess
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

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
GID_CHARACTERS = os.environ.get("GID_CHARACTERS", "1382939846")

AGNES_URL = "https://apihub.agnes-ai.com/v1/images/generations"
MODEL = "agnes-image-2.0-flash"
SIZE = "1024x768"
WAIT_BETWEEN = 3
MAX_RETRIES = 3

# Video tối ưu T4
SCENE_DURATION = 3.0
VIDEO_SCALE = "960:540"
VIDEO_PRESET = "veryfast"
VIDEO_CRF = "23"
AUDIO_BITRATE = "96k"
MAX_WORKERS_VIDEO = 2

STYLE_LOCK = "3d chinese donghua style, unreal engine 5, cinematic lighting, highly detailed, realistic proportions, adult proportions, consistent art style"
ANTI_CHIBI = "Do NOT make chibi, cute anime, cartoon, 2d anime, flat color, big head, small body, moe style, child proportion, oversized eyes."

# ==========================================
# GOOGLE OAUTH + DRIVE
# ==========================================
def get_google_access_token():
    if not all([CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN]):
        print("❌ Thiếu OAUTH_CLIENT_ID / SECRET / REFRESH_TOKEN")
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
        token = res.get("access_token")
        if not token:
            print("❌ OAuth lỗi:", res)
        return token
    except Exception as e:
        print(f"❌ OAuth exception: {e}")
        return None

def get_drive_files_map():
    """{file_name: file_id}"""
    token = get_google_access_token()
    if not token or not DRIVE_FOLDER_ID:
        print("⚠️ Không quét Drive được")
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
        print(f"📁 Drive folder có {len(files)} file")
        return {f["name"]: f["id"] for f in files}
    except Exception as e:
        print(f"❌ Quét Drive lỗi: {e}")
        return {}

def upload_file_to_drive(file_path: str):
    token = get_google_access_token()
    if not token or not DRIVE_FOLDER_ID:
        print("❌ Upload bỏ qua: thiếu token hoặc DRIVE_FOLDER_ID")
        return None
    if not os.path.exists(file_path) or os.path.getsize(file_path) < 1000:
        print(f"❌ File không hợp lệ: {file_path}")
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
                timeout=300,
            )
        if res.status_code in (200, 201):
            print(f"✅ Uploaded: {file_name}")
            return res.json()
        print(f"❌ Upload fail {res.status_code}: {res.text[:300]}")
    except Exception as e:
        print(f"❌ Upload error: {e}")
    return None

def download_from_drive(file_id: str, save_path: str) -> bool:
    token = get_google_access_token()
    if not token:
        return False
    headers = {"Authorization": f"Bearer {token}"}
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
    try:
        r = requests.get(url, headers=headers, timeout=120)
        if r.status_code == 200 and len(r.content) > 1000:
            with open(save_path, "wb") as f:
                f.write(r.content)
            return True
    except Exception as e:
        print(f"❌ Download Drive: {e}")
    return False

# ==========================================
# GOOGLE SHEET
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

def build_character_map(df_chars: pd.DataFrame) -> dict:
    image_map = {}
    for _, row in df_chars.iterrows():
        name = safe_get(row, "character_name", "name")
        url = safe_get(row, "image_url", "character_image_url", "url")
        if name and url.startswith("https://"):
            image_map[name] = url
            image_map[name.lower()] = url
    print(f"✅ {len(image_map)//2} nhân vật có ảnh reference")
    return image_map

def parse_names(raw: str) -> list:
    if not raw:
        return []
    return [p.strip() for p in raw.replace("&", ",").replace("|", ",").split(",") if p.strip()]

def find_character_urls(names: list, image_map: dict) -> list:
    urls = []
    for name in names:
        n = name.strip().lower()
        if n in ["nhân vật nền", "background", "extra", "crowd", "quần chúng"]:
            continue
        if n in image_map:
            urls.append(image_map[n])
            continue
        for key, url in image_map.items():
            if n in key.lower() or key.lower() in n:
                urls.append(url)
                break
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

# ==========================================
# AGNES
# ==========================================
def call_agnes(prompt: str, image_urls: list = None) -> str | None:
    if not AGNES_API_KEY:
        print("❌ Thiếu AGNES_API_KEY")
        return None
    body = {
        "model": MODEL,
        "prompt": prompt,
        "size": SIZE,
        "extra_body": {"response_format": "url"},
    }
    if image_urls:
        body["extra_body"]["image"] = image_urls

    headers = {
        "Authorization": f"Bearer {AGNES_API_KEY}",
        "Content-Type": "application/json",
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(AGNES_URL, headers=headers, json=body, timeout=120)
            if r.status_code == 200:
                url = r.json().get("data", [{}])[0].get("url")
                if url:
                    return url
            elif r.status_code in (429, 503):
                wait = 6 * attempt
                print(f"⚠️ {r.status_code} đợi {wait}s ({attempt}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                print(f"❌ Agnes {r.status_code}: {r.text[:250]}")
                time.sleep(3)
        except Exception as e:
            print(f"❌ Agnes error: {e}")
            time.sleep(3)
    return None

def download_url(url: str, save_path: str) -> bool:
    try:
        r = requests.get(url, timeout=90)
        if r.status_code == 200 and len(r.content) > 2000:
            with open(save_path, "wb") as f:
                f.write(r.content)
            return True
    except Exception as e:
        print(f"❌ Download URL: {e}")
    return False

def build_prompt(base: str, names: list, has_ref: bool) -> str:
    if has_ref and names:
        return (
            f"Use the reference images to keep the EXACT appearance of these characters: {', '.join(names)}. "
            f"Do NOT change face, hair, body, clothing. "
            f"Only change pose and environment: {base}. {STYLE_LOCK}. {ANTI_CHIBI}"
        )
    return f"{base}. {STYLE_LOCK}. {ANTI_CHIBI}"

# ==========================================
# GIAI ĐOẠN 1: ĐỦ 100 ẢNH TRÊN DRIVE
# ==========================================
def step_1_ensure_all_scene_images():
    print("\n" + "=" * 50)
    print("🔍 GIAI ĐOẠN 1: Kiểm tra & tạo đủ ảnh cảnh trên Drive")
    print("=" * 50)

    if not DRIVE_FOLDER_ID:
        print("❌ THIẾU DRIVE_FOLDER_ID")
        return False
    if not AGNES_API_KEY:
        print("❌ THIẾU AGNES_API_KEY")
        return False

    drive_map = get_drive_files_map()

    df_scenes = get_sheet_csv(GID_SCENES)
    df_chars = get_sheet_csv(GID_CHARACTERS)
    if df_scenes.empty:
        print("❌ Không đọc được sheet cảnh")
        return False

    image_map = build_character_map(df_chars)
    total = len(df_scenes)
    print(f"📊 {total} cảnh | reference: {len(image_map)//2} nhân vật")

    results = []
    ok_count = 0

    for idx, row in df_scenes.iterrows():
        scene_num = int(row.get("scene_index")) if pd.notna(row.get("scene_index")) else idx + 1
        img_name = f"scene_{scene_num:03d}.png"
        local_path = f"{OUTPUT_DIR}/{img_name}"

        # Đã có trên Drive
        if img_name in drive_map:
            print(f"✅ [{scene_num}] đã có trên Drive")
            if not (os.path.exists(local_path) and os.path.getsize(local_path) > 2000):
                download_from_drive(drive_map[img_name], local_path)
            ok_count += 1
            results.append({"scene_index": scene_num, "status": "drive_exists"})
            continue

        # Có local chưa upload
        if os.path.exists(local_path) and os.path.getsize(local_path) > 2000:
            print(f"📤 [{scene_num}] có local → upload Drive")
            if upload_file_to_drive(local_path):
                ok_count += 1
                results.append({"scene_index": scene_num, "status": "uploaded_from_local"})
            else:
                results.append({"scene_index": scene_num, "status": "upload_fail"})
            continue

        # Chưa có → tạo bằng Agnes
        names = parse_names(safe_get(row, "character_name", "character_names"))
        char_urls = find_character_urls(names, image_map)
        base = safe_get(row, "image_prompt", "scene_description") or f"Scene {scene_num}"
        prompt = build_prompt(base, names, bool(char_urls))

        print(f"🎨 [{scene_num}/{total}] tạo ảnh | chars={names or '-'} | ref={len(char_urls)}")
        image_url = call_agnes(prompt, char_urls if char_urls else None)

        if not image_url:
            print(f"❌ [{scene_num}] Agnes fail")
            results.append({"scene_index": scene_num, "status": "generate_fail"})
            continue

        if not download_url(image_url, local_path):
            print(f"❌ [{scene_num}] download fail")
            results.append({"scene_index": scene_num, "status": "download_fail", "url": image_url})
            continue

        if upload_file_to_drive(local_path):
            ok_count += 1
            results.append({
                "scene_index": scene_num,
                "status": "created_and_uploaded",
                "image_url": image_url,
                "characters": names,
            })
        else:
            results.append({
                "scene_index": scene_num,
                "status": "created_but_upload_fail",
                "image_url": image_url,
            })

        gc.collect()
        time.sleep(WAIT_BETWEEN)

    with open("agnes_scene_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ GIAI ĐOẠN 1 xong: {ok_count}/{total} ảnh sẵn sàng")
    return ok_count >= max(1, int(total * 0.8))

# ==========================================
# GIAI ĐOẠN 2: VIDEO NHANH – NHẸ (T4)
# ==========================================
def auto_install_video_tools():
    print("📦 Cài ffmpeg + edge-tts...")
    try:
        subprocess.run(["edge-tts", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "edge-tts"], check=False)
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        subprocess.run("apt-get update -qq && apt-get install -y -qq ffmpeg", shell=True, check=False)

def get_dialogue_for_scene(scene_num: int, df_scenes: pd.DataFrame) -> str:
    try:
        if "scene_index" in df_scenes.columns:
            row = df_scenes[df_scenes["scene_index"] == scene_num]
            if row.empty:
                row = df_scenes.iloc[scene_num - 1:scene_num]
        else:
            row = df_scenes.iloc[scene_num - 1:scene_num]
        if not row.empty:
            text = safe_get(row.iloc[0], "dialogue")
            if text and len(text) > 2:
                return text[:180]
    except Exception:
        pass
    return ""

def make_tts(text: str, out_mp3: str) -> bool:
    if not text or len(text.strip()) < 2:
        return False
    if os.path.exists(out_mp3) and os.path.getsize(out_mp3) > 400:
        return True
    safe = text.replace('"', "'").replace("\n", " ").strip()
    cmd = (
        f'edge-tts --text "{safe}" --voice vi-VN-NamMinhNeural '
        f'--rate=+5% --write-media "{out_mp3}"'
    )
    try:
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        return os.path.exists(out_mp3) and os.path.getsize(out_mp3) > 400
    except Exception:
        return False

def process_single_video_scene(args):
    scene_num, dialogue = args
    img_path = f"{OUTPUT_DIR}/scene_{scene_num:03d}.png"
    voice_path = f"{VIDEO_DIR}/voice_{scene_num:03d}.mp3"
    video_path = f"{VIDEO_DIR}/scene_{scene_num:03d}.mp4"

    if os.path.exists(video_path) and os.path.getsize(video_path) > 2000:
        return video_path
    if not (os.path.exists(img_path) and os.path.getsize(img_path) > 2000):
        return None

    has_voice = make_tts(dialogue, voice_path)
    vf = f"scale={VIDEO_SCALE}:force_original_aspect_ratio=decrease,pad={VIDEO_SCALE}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=24"

    if has_voice:
        cmd = (
            f'ffmpeg -y -hide_banner -loglevel error '
            f'-loop 1 -i "{img_path}" -i "{voice_path}" '
            f'-vf "{vf}" '
            f'-c:v libx264 -preset {VIDEO_PRESET} -crf {VIDEO_CRF} -pix_fmt yuv420p '
            f'-c:a aac -b:a {AUDIO_BITRATE} -ac 2 -ar 44100 '
            f'-shortest -movflags +faststart '
            f'"{video_path}"'
        )
    else:
        cmd = (
            f'ffmpeg -y -hide_banner -loglevel error '
            f'-loop 1 -t {SCENE_DURATION} -i "{img_path}" '
            f'-vf "{vf}" '
            f'-c:v libx264 -preset {VIDEO_PRESET} -crf {VIDEO_CRF} -pix_fmt yuv420p '
            f'-an -movflags +faststart '
            f'"{video_path}"'
        )

    try:
        subprocess.run(cmd, shell=True, timeout=120)
    except Exception as e:
        print(f"❌ scene {scene_num}: {e}")
        return None

    if os.path.exists(video_path) and os.path.getsize(video_path) > 2000:
        return video_path
    return None

def step_2_render_merge_upload():
    print("\n" + "=" * 50)
    print("🎬 GIAI ĐOẠN 2: Video nhanh – nhẹ (T4)")
    print("=" * 50)

    auto_install_video_tools()
    df_scenes = get_sheet_csv(GID_SCENES)

    # Tải ảnh từ Drive về local nếu thiếu
    drive_map = get_drive_files_map()
    for i in range(1, TOTAL_SCENES + 1):
        local = f"{OUTPUT_DIR}/scene_{i:03d}.png"
        name = f"scene_{i:03d}.png"
        if not (os.path.exists(local) and os.path.getsize(local) > 2000):
            if name in drive_map:
                download_from_drive(drive_map[name], local)

    tasks = []
    for i in range(1, TOTAL_SCENES + 1):
        dialogue = get_dialogue_for_scene(i, df_scenes) if not df_scenes.empty else ""
        tasks.append((i, dialogue))

    print(f"⚡ Render {TOTAL_SCENES} cảnh | {VIDEO_SCALE} | preset={VIDEO_PRESET} | crf={VIDEO_CRF}")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_VIDEO) as ex:
        videos = list(ex.map(process_single_video_scene, tasks))

    valid = [v for v in videos if v]
    print(f"✅ Render xong {len(valid)}/{TOTAL_SCENES} trong {time.time()-t0:.1f}s")

    if not valid:
        print("❌ Không có video để ghép")
        return False

    # Ghép (copy stream – nhanh)
    concat_file = "concat_list.txt"
    with open(concat_file, "w", encoding="utf-8") as f:
        for v in sorted(valid):
            f.write(f"file '{os.path.abspath(v)}'\n")

    print("🎞️ Ghép phim...")
    subprocess.run(
        f'ffmpeg -y -hide_banner -loglevel error '
        f'-f concat -safe 0 -i "{concat_file}" -c copy -movflags +faststart "{FINAL_VIDEO}"',
        shell=True,
    )

    if not (os.path.exists(FINAL_VIDEO) and os.path.getsize(FINAL_VIDEO) > 5000):
        print("⚠️ Copy fail → re-encode nhẹ")
        subprocess.run(
            f'ffmpeg -y -hide_banner -loglevel error '
            f'-f concat -safe 0 -i "{concat_file}" '
            f'-c:v libx264 -preset veryfast -crf 23 -c:a aac -b:a 96k '
            f'-movflags +faststart "{FINAL_VIDEO}"',
            shell=True,
        )

    if not (os.path.exists(FINAL_VIDEO) and os.path.getsize(FINAL_VIDEO) > 5000):
        print("❌ Ghép phim thất bại")
        return False

    size_mb = os.path.getsize(FINAL_VIDEO) / 1024 / 1024
    print(f"✅ Phim: {FINAL_VIDEO} ({size_mb:.1f} MB)")
    upload_file_to_drive(FINAL_VIDEO)
    return True

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print("DRIVE_FOLDER_ID =", DRIVE_FOLDER_ID or "(EMPTY)")
    print("AGNES_API_KEY  =", "OK" if AGNES_API_KEY else "(EMPTY)")

    images_ready = step_1_ensure_all_scene_images()

    if images_ready:
        step_2_render_merge_upload()
        print("\n🎉 HOÀN TẤT PIPELINE")
    else:
        print("\n❌ Dừng: chưa đủ ảnh cảnh trên Drive")
