import os
import sys
import json
import requests
import subprocess
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# CẤU HÌNH BIẾN MÔI TRƯỜNG TỪ N8N
# ==========================================
TOTAL_SCENES = 100
OUTPUT_DIR = "./output_scenes"
VIDEO_DIR = "./output_videos"
FINAL_VIDEO = "final_full_movie.mp4"
MAX_WORKERS = 8  # Chạy đa luồng T4 GPU

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID")
CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID")
CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET")
REFRESH_TOKEN = os.environ.get("OAUTH_REFRESH_TOKEN")
AGNES_API_KEY = os.environ.get("AGNES_API_KEY")

# ==========================================
# 1. HÀM TƯƠNG TÁC GOOGLE DRIVE OAUTH
# ==========================================
def get_google_access_token():
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
        print(f"❌ Lỗi OAuth Google: {e}")
        return None

def get_drive_files_list():
    """Lấy danh sách tất cả tên file trên Drive Folder"""
    token = get_google_access_token()
    if not token or not DRIVE_FOLDER_ID:
        print("⚠️ Không lấy được danh sách file trên Drive Folder!")
        return set()

    headers = {'Authorization': f'Bearer {token}'}
    query = f"'{DRIVE_FOLDER_ID}' in parents and trashed = false"
    url = f"https://www.googleapis.com/drive/v3/files?q={requests.utils.quote(query)}&fields=files(name)&pageSize=1000"
    
    try:
        res = requests.get(url, headers=headers, timeout=15).json()
        return {f['name'] for f in res.get('files', [])}
    except Exception as e:
        print(f"❌ Lỗi quét Drive: {e}")
        return set()

def upload_file_to_drive(file_path):
    token = get_google_access_token()
    if not token or not DRIVE_FOLDER_ID or not os.path.exists(file_path):
        return None

    file_name = os.path.basename(file_path)
    metadata = {'name': file_name, 'parents': [DRIVE_FOLDER_ID]}
    files = {
        'data': ('metadata', json.dumps(metadata), 'application/json; charset=UTF-8'),
        'file': open(file_path, 'rb')
    }
    headers = {'Authorization': f'Bearer {token}'}
    
    print(f"🚀 [Drive Upload] Đang tải {file_name}...")
    res = requests.post("https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart", headers=headers, files=files, timeout=180)
    if res.status_code == 200:
        print(f"✅ Upload thành công: {file_name}")
        return res.json()
    return None


# ==========================================
# BƯỚC 1: KIỂM TRA VÀ TẠO ĐỦ 100 ẢNH CẢNH
# ==========================================
def step_1_check_and_generate_all_images():
    print("🔍 [GIAI ĐOẠN 1] Bắt đầu kiểm tra kho ảnh trên Google Drive...")
    drive_files = get_drive_files_list()
    
    missing_scenes = []
    for i in range(1, TOTAL_SCENES + 1):
        img_name = f"scene_{i:03d}.png"
        local_path = f"{OUTPUT_DIR}/{img_name}"
        
        # Nếu chưa có trên Drive lẫn dưới local Kaggle
        if img_name not in drive_files and not (os.path.exists(local_path) and os.path.getsize(local_path) > 0):
            missing_scenes.append(i)

    # NẾU THIẾU ẢNH -> TIẾN HÀNH TẠO ẢNH
    if missing_scenes:
        print(f"⚠️ Phát hiện thiếu {len(missing_scenes)} ảnh cảnh! Đang tiến hành tạo ảnh bổ sung...")
        
        # Đọc dữ liệu prompt từ file JSON n8n truyền sang
        json_log = "agnes_scene_results.json"
        scenes_data = []
        if os.path.exists(json_log):
            with open(json_log, 'r', encoding='utf-8') as f:
                scenes_data = json.load(f)

        for scene_num in missing_scenes:
            img_name = f"scene_{scene_num:03d}.png"
            local_path = f"{OUTPUT_DIR}/{img_name}"
            
            # Lấy prompt tương ứng
            prompt = scenes_data[scene_num - 1].get("prompt", f"Scene {scene_num}") if len(scenes_data) >= scene_num else f"Scene {scene_num}"
            
            print(f"🎨 Đang sinh ảnh [{scene_num}/{TOTAL_SCENES}] bằng Agnes AI...")
            
            # --- CHÈN CODE HÀM CALL API AGNES AI TẠO ẢNH CỦA BẠN TẠI ĐÂY ---
            # Sau khi tải xong ảnh xuống local_path, tự động upload lên Drive:
            upload_file_to_drive(local_path)

        # Quét lại danh sách Drive lần cuối
        drive_files = get_drive_files_list()

    print(f"✅ [GIAI ĐOẠN 1 HOÀN THÀNH] Đã xác nhận đủ 100/100 ảnh cảnh!")
    return True


# ==========================================
# BƯỚC 2: CÀI ĐẶT MÔI TRƯỜNG & TẠO VIDEO
# ==========================================
def auto_install_video_tools():
    """Chỉ cài đặt tool video sau khi đã chắc chắn đủ 100 ảnh"""
    print("📦 [GIAI ĐOẠN 2] Đang tiến hành cài đặt công cụ Video (ffmpeg, edge-tts)...")
    try:
        subprocess.run(["edge-tts", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        subprocess.run([sys.executable, "-m", "pip", "install", "edge-tts"], check=True)

    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        subprocess.run("apt-get update && apt-get install -y ffmpeg", shell=True, check=True)


def process_single_video_scene(scene_num):
    img_path = f"{OUTPUT_DIR}/scene_{scene_num:03d}.png"
    voice_audio = f"{VIDEO_DIR}/voice_{scene_num:03d}.mp3"
    scene_video = f"{VIDEO_DIR}/scene_final_{scene_num:03d}.mp4"

    # Nếu video cảnh này đã render sẵn -> Bỏ qua
    if os.path.exists(scene_video) and os.path.getsize(scene_video) > 0:
        return scene_video

    if not os.path.exists(img_path):
        return None

    # Tạo Voice thoại
    if not (os.path.exists(voice_audio) and os.path.getsize(voice_audio) > 0):
        cmd_tts = f'edge-tts --text "Cảnh {scene_num}" --voice vi-VN-NamMinhNeural --write-media "{voice_audio}"'
        subprocess.run(cmd_tts, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Render Video siêu nhanh với FFmpeg Ultrafast
    cmd_render = (
        f'ffmpeg -y -loop 1 -i "{img_path}" -i "{voice_audio}" '
        f'-c:v libx264 -preset ultrafast -tune stillimage -crf 18 -pix_fmt yuv420p '
        f'-c:a aac -b:a 128k -shortest "{scene_video}"'
    )
    subprocess.run(cmd_render, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return scene_video


def step_2_render_and_merge_video():
    # 1. Cài đặt môi trường
    auto_install_video_tools()

    # 2. Render 100 cảnh đa luồng T4 GPU
    print(f"⚡ Bắt đầu render Video cho {TOTAL_SCENES} cảnh (Đa luồng {MAX_WORKERS})...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(process_single_video_scene, range(1, TOTAL_SCENES + 1)))

    valid_videos = [v for v in results if v and os.path.exists(v)]

    # 3. Ghép phim hoàn chỉnh
    if valid_videos and not os.path.exists(FINAL_VIDEO):
        concat_list = "concat_list.txt"
        with open(concat_list, "w", encoding="utf-8") as f:
            for vid in valid_videos:
                f.write(f"file '{os.path.abspath(vid)}'\n")

        print("🎞️ Đang ghép toàn bộ các cảnh thành PHIM HOÀN CHỈNH...")
        subprocess.run(f'ffmpeg -y -f concat -safe 0 -i "{concat_list}" -c copy "{FINAL_VIDEO}"', shell=True)

    # 4. Upload phim hoàn chỉnh lên Google Drive
    if os.path.exists(FINAL_VIDEO):
        upload_file_to_drive(FINAL_VIDEO)


# ==========================================
# ĐIỂM BẮT ĐẦU CHẠY PHÂN LUỒNG TUẦN TỰ
# ==========================================
if __name__ == "__main__":
    # BƯỚC 1: Phải đảm bảo ĐỦ 100 ẢNH trước
    is_images_ready = step_1_check_and_generate_all_images()

    # BƯỚC 2: Chỉ khi đủ 100 ảnh mới chuyển sang CÀI ĐẶT & RENDER VIDEO
    if is_images_ready:
        step_2_render_and_merge_video()
