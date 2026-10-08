# ============================================================
# AGNES - TẠO 100 ẢNH CẢNH
# - Đọc scenes + character images từ Google Sheet
# - Gọi Agnes API (agnes-image-2.0-flash)
# - Có reference nhân vật thì dùng, không có thì text-to-image
# ============================================================

import os
import time
import gc
import json
import requests
import pandas as pd
from pathlib import Path

# ==================== CONFIG ====================
AGNES_API_KEY = os.environ.get("AGNES_API_KEY", "sk-ix43BxLdae2nhuPomn1j39qvUGQd2DDXrXv3DSrPdCITnPRX")         # bắt buộc
AGNES_URL = "https://apihub.agnes-ai.com/v1/images/generations"
MODEL = "agnes-image-2.0-flash"
SIZE = "1024x768"

SHEET_ID = os.environ.get("SHEET_ID", "1DmA-yuPwDl1riceSMGzWPXhuxrL4y987lOZ6Af351l8")
GID_SCENES = os.environ.get("GID_SCENES", "0")          # sheet chứa 100 cảnh
GID_CHARACTERS = os.environ.get("GID_CHARACTERS", "1") # sheet chứa nhân vật + image_url

# Folder Drive (nếu muốn upload)
DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID", "").strip()

# OAuth Google Drive (nếu upload)
OAUTH_CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID", "")
OAUTH_CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET", "")
OAUTH_REFRESH_TOKEN = os.environ.get("OAUTH_REFRESH_TOKEN", "")
USE_DRIVE = bool(OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET and OAUTH_REFRESH_TOKEN and DRIVE_FOLDER_ID)

# Webhook báo về n8n (tuỳ chọn)
N8N_WEBHOOK_URL = os.environ.get("N8N_WEBHOOK_URL", "")

# Giới hạn tốc độ (tránh rate limit + tràn RAM)
WAIT_BETWEEN_SCENES = 4          # giây
MAX_RETRIES = 3

STYLE_LOCK = "3d chinese donghua style, unreal engine 5, cinematic lighting, highly detailed, realistic proportions, adult proportions, consistent art style"
ANTI_CHIBI = "Do NOT make chibi, cute anime, cartoon, 2d anime, flat color, big head, small body, moe style, child proportion, oversized eyes."

# ==================== HELPERS ====================
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
        print(f"❌ Đọc Sheet gid={gid}: {e}")
        return pd.DataFrame()

def build_character_map(df_chars: pd.DataFrame) -> dict:
    """Tên nhân vật → image_url"""
    image_map = {}
    if df_chars.empty:
        return image_map

    for _, row in df_chars.iterrows():
        name = safe_get(row, "character_name", "name", "character")
        url = safe_get(row, "image_url", "character_image_url", "url", "image")
        if name and url.startswith("https://"):
            image_map[name] = url
            image_map[name.lower()] = url
    print(f"✅ Character map: {len(image_map)//2} nhân vật")
    return image_map

def find_character_urls(names: list, image_map: dict) -> list:
    """Tìm đúng số ảnh = số tên nhân vật trong cảnh"""
    urls = []
    for name in names:
        if not name:
            continue
        name_clean = name.strip().lower()
        if name_clean in ["nhân vật nền", "background", "extra", "crowd", "quần chúng"]:
            continue

        # khớp chính xác
        if name_clean in image_map:
            urls.append(image_map[name_clean])
            continue

        # khớp tương đối
        found = None
        for key, url in image_map.items():
            key_clean = key.lower()
            if key_clean.includes(name_clean) if False else (name_clean in key_clean or key_clean in name_clean):
                found = url
                break
        if found:
            urls.append(found)
            continue

        # khớp theo từ
        words = [w for w in name_clean.split() if len(w) >= 2]
        for word in words:
            for key, url in image_map.items():
                if word in key.lower():
                    found = url
                    break
            if found:
                urls.append(found)
                break

    # loại trùng, giữ thứ tự
    seen = set()
    result = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result

def parse_character_names(row) -> list:
    raw = safe_get(row, "character_name", "character_names")
    if not raw:
        return []
    # hỗ trợ "A, B, C" hoặc "A & B"
    parts = [p.strip() for p in raw.replace("&", ",").replace("|", ",").split(",") if p.strip()]
    return parts

def build_prompt(row, has_reference: bool, names: list) -> str:
    base = safe_get(row, "image_prompt", "scene_description")
    if not base:
        base = "cinematic scene"

    if has_reference and names:
        char_list = ", ".join(names)
        return (
            f"Use the reference images to keep the EXACT appearance of these characters: {char_list}. "
            f"Each character must keep the same face, hair, body type, clothing and accessories as in their reference image. "
            f"Do NOT change gender, face, hair, or outfit. Do NOT generate different people. "
            f"Only change pose, expression, camera and environment according to: {base}. "
            f"{STYLE_LOCK}. {ANTI_CHIBI}"
        )
    return f"{base}. {STYLE_LOCK}. {ANTI_CHIBI}"

def call_agnes(prompt: str, image_urls: list = None) -> str | None:
    """Gọi Agnes, trả về image URL hoặc None"""
    if not AGNES_API_KEY:
        print("❌ Thiếu AGNES_API_KEY")
        return None

    body = {
        "model": MODEL,
        "prompt": prompt,
        "size": SIZE,
        "extra_body": {
            "response_format": "url"
        }
    }
    if image_urls:
        body["extra_body"]["image"] = image_urls

    headers = {
        "Authorization": f"Bearer {AGNES_API_KEY}",
        "Content-Type": "application/json"
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(AGNES_URL, headers=headers, json=body, timeout=120)
            if r.status_code == 200:
                data = r.json()
                url = data.get("data", [{}])[0].get("url")
                if url:
                    return url
                print(f"⚠️ Response không có url: {data}")
            elif r.status_code in (429, 503):
                wait = 8 * attempt
                print(f"⚠️ {r.status_code} - đợi {wait}s rồi thử lại ({attempt}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                print(f"❌ Agnes {r.status_code}: {r.text[:300]}")
                if attempt < MAX_RETRIES:
                    time.sleep(5)
        except Exception as e:
            print(f"❌ Request error: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(5)
    return None

def upload_to_drive(file_path: str, folder_id: str):
    """Upload file local lên Drive (cần oauth)"""
    if not USE_DRIVE or not os.path.exists(file_path):
        return None
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload

        creds = Credentials(
            token=None,
            refresh_token=OAUTH_REFRESH_TOKEN,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=OAUTH_CLIENT_ID,
            client_secret=OAUTH_CLIENT_SECRET,
            scopes=["https://www.googleapis.com/auth/drive"],
        )
        service = build("drive", "v3", credentials=creds, cache_discovery=False)
        media = MediaFileUpload(file_path, mimetype="image/png", resumable=True)
        name = os.path.basename(file_path)
        f = service.files().create(
            body={"name": name, "parents": [folder_id]},
            media_body=media,
            fields="id"
        ).execute()
        print(f"☁️ Uploaded {name}")
        return f.get("id")
    except Exception as e:
        print(f"⚠️ Upload Drive: {e}")
        return None

def download_image(url: str, save_path: str) -> bool:
    try:
        r = requests.get(url, timeout=60)
        if r.status_code == 200 and len(r.content) > 1000:
            with open(save_path, "wb") as f:
                f.write(r.content)
            return True
    except Exception as e:
        print(f"⚠️ Download: {e}")
    return False

def notify_n8n(status: str, total: int = 0, extra: dict = None):
    if not N8N_WEBHOOK_URL:
        return
    payload = {
        "status": status,
        "total_scenes": total,
        "drive_folder_id": DRIVE_FOLDER_ID,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
    }
    if extra:
        payload.update(extra)
    try:
        requests.post(N8N_WEBHOOK_URL, json=payload, timeout=20)
    except Exception:
        pass

# ==================== MAIN ====================
def main():
    print("🚀 Agnes - Tạo 100 ảnh cảnh")
    if not AGNES_API_KEY:
        raise SystemExit("Thiếu AGNES_API_KEY")

    # 1. Đọc Sheet
    print("\n📊 Đọc Google Sheet...")
    df_scenes = get_sheet_csv(GID_SCENES)
    df_chars = get_sheet_csv(GID_CHARACTERS)

    if df_scenes.empty:
        notify_n8n("failed", error_message="Không đọc được scenes")
        raise SystemExit("Không có dữ liệu scenes")

    image_map = build_character_map(df_chars)
    total = len(df_scenes)
    print(f"✅ {total} cảnh | {len(image_map)//2} nhân vật có ảnh")

    results = []
    success = 0

    for idx, row in df_scenes.iterrows():
        scene_index = int(row.get("scene_index")) if pd.notna(row.get("scene_index")) else idx + 1
        names = parse_character_names(row)
        char_urls = find_character_urls(names, image_map)
        has_ref = len(char_urls) > 0

        prompt = build_prompt(row, has_ref, names)
        print(f"\n🎬 [{scene_index}/{total}] {names or 'no-char'} | ref={len(char_urls)}")

        # Bỏ qua nếu đã có file local
        local_name = f"scene_{scene_index:03d}.png"
        if os.path.exists(local_name) and os.path.getsize(local_name) > 5000:
            print(f"⏩ Đã có {local_name}")
            results.append({"scene_index": scene_index, "image_url": local_name, "status": "exists"})
            success += 1
            continue

        image_url = call_agnes(prompt, char_urls if has_ref else None)

        if image_url:
            # Tải về local
            if download_image(image_url, local_name):
                if USE_DRIVE:
                    upload_to_drive(local_name, DRIVE_FOLDER_ID)
                results.append({
                    "scene_index": scene_index,
                    "image_url": image_url,
                    "local": local_name,
                    "characters": names,
                    "has_reference": has_ref,
                    "status": "ok"
                })
                success += 1
                print(f"✅ {local_name}")
            else:
                results.append({"scene_index": scene_index, "status": "download_fail", "url": image_url})
                print("❌ Download fail")
        else:
            results.append({"scene_index": scene_index, "status": "generate_fail", "characters": names})
            print("❌ Generate fail")

        # Giải phóng + chờ
        gc.collect()
        time.sleep(WAIT_BETWEEN_SCENES)

    # Lưu kết quả
    out_path = "agnes_scene_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n🎉 Xong: {success}/{total} ảnh")
    print(f"📄 Kết quả: {out_path}")

    notify_n8n("completed", total=success, extra={"result_file": out_path})
    return results

if __name__ == "__main__":
    main()
