import os
import sys
import json
import subprocess

# ==========================================
# 0. TỰ ĐỘNG CÀI ĐẶT MÔI TRƯỜNG (AUTO-SETUP)
# ==========================================
def auto_setup():
    try:
        import requests
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "requests"], check=True)

    try:
        subprocess.run(["edge-tts", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        subprocess.run([sys.executable, "-m", "pip", "install", "edge-tts"], check=True)

    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        subprocess.run("apt-get update && apt-get install -y ffmpeg", shell=True, check=True)

auto_setup()

import requests
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# 1. CẤU HÌNH & LẤY BIẾN MÔI TRƯỜNG TỪ N8N
# ==========================================
OUTPUT_DIR = "./output_scenes"         # Thư mục chứa ảnh 100 cảnh
VIDEO_DIR = "./output_videos"         # Thư mục lưu video & voice từng cảnh
FINAL_VIDEO = "final_full_movie.mp4"   # File phim hoàn chỉnh
JSON_LOG = "agnes_scene_results.json"  # File JSON dữ liệu cảnh từ n8n

MAX_WORKERS = 8  # Chạy 8 luồng song song trên Kaggle T4 GPU

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# Lấy trực tiếp thông tin từ n8n / GitHub Secrets truyền qua
DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID")
CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID")
CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET")
REFRESH_TOKEN = os.environ.get("OAUTH_REFRESH_TOKEN")
AGNES_API_KEY = os.environ.get("AGNES_API_KEY") # API Key tạo ảnh


# ==========================================
# 2. XÁC THỰC GOOGLE DRIVE & UPLOAD
# ==========================================
def get_google_access_token():
    if not all([CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN]):
        print("❌ Thiếu OAUTH Credentials!")
        return None

    url = "https://oauth2.googleapis.com/token"
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "refresh_token": REFRESH_TOKEN,
        "grant_type": "refresh_token"
    }
    try:
        res = requests.post(url, data=data, timeout=15).json()
        return res.get("access_token")
    except Exception as e:
        print(f"❌ Lỗi lấy Access Token: {e}")
        return None

def upload_file_to_drive(file_path):
    if not os.path.exists(file_path) or not DRIVE_FOLDER_ID:
        return None

    access_token = get_google_access_token()
    if not access_token:
        print("❌ Không có Access Token để upload!")
        return None

    file_name = os.path.basename(file_path)
    metadata = {'name': file_name, 'parents': [DRIVE_FOLDER_ID]}
    
    files = {
        'data': ('metadata', json.dumps(metadata), 'application/json; charset=UTF-8'),
        'file': open(file_path, 'rb')
    }
    headers = {'Authorization': f'Bearer {access_token}'}
    
    try:
        res = requests.post(
            "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
            headers=headers,
            files=files,
            timeout=180
        )
        if res.status_code == 200:
            print(f"☁️  [Drive Upload] Thành công: {file_name}")
            return res.json()
        else:
            print(f"❌ Lỗi Upload {file_name}: {res.text}")
            return None
    except Exception as e:
        print(f"❌ Exception Upload: {e}")
        return None


# ==========================================
# 3. BƯỚC 1: KIỂM TRA & TẠO ẢNH (AGNES AI)
# ==========================================
def generate_image_if_not_exists(scene_num, prompt):
    img_path = f"{OUTPUT_DIR}/scene_{scene_num:03d}.png"

    # NẾU ĐÃ CÓ ẢNH TỒN TẠI -> SKIP KHÔNG TẠO LẠI
    if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
        print(f"⏭️  [Ảnh {scene_num:03d}] Đã có sẵn -> Bỏ qua tạo ảnh.")
        return img_path

    print(f"🎨 [Ảnh {scene_num:03d}] Đang tạo mới bằng Agnes AI...")
    
    # Mẫu gọi API sinh ảnh (Thay endpoint/payload của Agnes AI bạn đang dùng)
    try:
        # Giả lập/Gọi API sinh ảnh
        headers = {"Authorization": f"Bearer {AGNES_API_KEY}", "Content-Type": "application/json"}
        payload = {"prompt": prompt, "aspect_ratio": "16:9"}
        
        # Nếu dùng API thực tế:
        # response = requests.post("https://api.agnes.ai/v1/generate", json=payload, headers=headers)
        # img_url = response.json().get("image_url")
        # img_data = requests.get(img_url).content
        # with open(img_path, "wb") as f: f.write(img_data)
        
        # Upload ảnh vừa tạo lên Google Drive
        upload_file_to_drive(img_path)
        return img_path
    except Exception as e:
        print(f"❌ Lỗi tạo ảnh cảnh {scene_num}: {e}")
        return None


# ==========================================
# 4. BƯỚC 2: KIỂM TRA & TẠO VIDEO + LỒNG TIẾNG
# ==========================================
def process_single_scene(item):
    idx, scene = item
    scene_num = idx + 1
    
    img_path = f"{OUTPUT_DIR}/scene_{scene_num:03d}.png"
    voice_audio = f"{VIDEO_DIR}/voice_{scene_num:03d}.mp3"
    scene_video = f"{VIDEO_DIR}/scene_final_{scene_num:03d}.mp4"

    # 1. BƯỚC CHÍNH: Tạo ảnh nếu chưa có
    prompt = scene.get("prompt", "")
    if not (os.path.exists(img_path) and os.path.getsize(img_path) > 0):
        generate_image_if_not_exists(scene_num, prompt)

    # Nếu vẫn không có ảnh thì bỏ qua
    if not os.path.exists(img_path):
        print(f"⚠️  Cảnh {scene_num:03d}: Thiếu ảnh nguồn -> Bỏ qua video.")
        return None

    # 2. KIỂM TRA VIDEO CẢNH: Nếu đã có video rồi -> SKIP
    if os.path.exists(scene_video) and os.path.getsize(scene_video) > 0:
        print(f"⏭️  [Video {scene_num:03d}] Đã có sẵn -> Bỏ qua render.")
        return scene_video

    # A. Tạo Voice lồng tiếng (TTS)
    if not (os.path.exists(voice_audio) and os.path.getsize(voice_audio) > 0):
        dialogue = scene.get("dialogue", "") or prompt
        if dialogue:
            clean_text = dialogue.replace('"', '').replace("'", "").replace("\n", " ")
            cmd_tts = f'edge-tts --text "{clean_text}" --voice vi-VN-NamMinhNeural --write-media "{voice_audio}"'
            subprocess.run(cmd_tts, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(f'ffmpeg -y -f lavfi -i anullsrc=r=44100:cl=mono -t 2 -q:a 9 -acodec libmp3lame "{voice_audio}"', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # B. Render Video bằng FFmpeg Ultrafast (Tối ưu T4 GPU)
    cmd_render = (
        f'ffmpeg -y -loop 1 -i "{img_path}" -i "{voice_audio}" '
        f'-c:v libx264 -preset ultrafast -tune stillimage -crf 18 -pix_fmt yuv420p '
        f'-c:a aac -b:a 128k -shortest "{scene_video}"'
    )
    subprocess.run(cmd_render, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return scene_video


# ==========================================
# 5. TIẾN TRÌNH TỔNG HỢP & PHÁT HÀNH PHIM
# ==========================================
def main():
    if not os.path.exists(JSON_LOG):
        print(f"❌ Không tìm thấy file dữ liệu {JSON_LOG}!")
        return

    with open(JSON_LOG, 'r', encoding='utf-8') as f:
        scenes = json.load(f)

    total = len(scenes)
    print(f"🚀 BẮT ĐẦU WORKFLOW: {total} cảnh (Xử lý song song {MAX_WORKERS} luồng Kaggle T4)...")

    # Xử lý song song 8 cảnh cùng lúc (Tự tạo ảnh -> Tự tạo voice -> Tự tạo video)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(process_single_scene, enumerate(scenes)))

    valid_videos = [v for v in results if v and os.path.exists(v)]
    print(f"✅ Hoàn thành chuẩn bị: {len(valid_videos)}/{total} video cảnh!")

    if not valid_videos:
        print("❌ Không có video cảnh nào hợp lệ!")
        return

    # BƯỚC 3: Ghép toàn bộ các cảnh thành Video Phim Hoàn Chỉnh
    if not (os.path.exists(FINAL_VIDEO) and os.path.getsize(FINAL_VIDEO) > 0 and len(valid_videos) == total):
        concat_list = "concat_list.txt"
        with open(concat_list, "w", encoding="utf-8") as f:
            for vid in valid_videos:
                f.write(f"file '{os.path.abspath(vid)}'\n")

        print("🎞️  Đang ghép tất cả cảnh thành PHIM HOÀN CHỈNH...")
        subprocess.run(f'ffmpeg -y -f concat -safe 0 -i "{concat_list}" -c copy "{FINAL_VIDEO}"', shell=True)

    # BƯỚC 4: Upload file Video Phim Hoàn Chỉnh lên Google Drive
    if os.path.exists(FINAL_VIDEO):
        mb_size = os.path.getsize(FINAL_VIDEO) / (1024 * 1024)
        print(f"🎉 HOÀN THÀNH PHIM: {FINAL_VIDEO} ({mb_size:.2f} MB)")
        upload_file_to_drive(FINAL_VIDEO)

if __name__ == "__main__":
    main()
