import os
import gc
import re
import time
import torch
import pandas as pd
from PIL import Image
from diffusers import StableDiffusionXLPipeline, StableVideoDiffusionPipeline
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
import edge_tts
import asyncio

# ==========================================
# CẤU HÌNH BIẾN MÔI TRƯỜNG & TOKEN
# ==========================================
HF_TOKEN = os.getenv("HF_TOKEN", "")
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID", "")
REFRESH_TOKEN = os.getenv("REFRESH_TOKEN", "")
CLIENT_ID = os.getenv("CLIENT_ID", "")
CLIENT_SECRET = os.getenv("CLIENT_SECRET", "")

# Rút ngắn STYLE_PREFIX để đảm bảo tổng số Token < 77 (ngưỡng giới hạn của CLIP)
STYLE_PREFIX = "3d chinese donghua animation, unreal engine 5, detailed 3d face, anime, cinematic,"
NEGATIVE_PROMPT = "deformed, distorted, disfigured, low quality, bad anatomy, bad hands, blurry"

# ==========================================
# TỰ ĐỘNG UPLOAD LÊN GOOGLE DRIVE
# ==========================================
def get_drive_service():
    creds = Credentials(
        None,
        refresh_token=REFRESH_TOKEN,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
    )
    return build('drive', 'v3', credentials=creds)

def upload_file_to_drive_fresh(file_path, folder_id):
    if not os.path.exists(file_path) or not folder_id:
        print(f"⚠️ Bỏ qua upload: Không tìm thấy file {file_path} hoặc chưa cấu hình DRIVE_FOLDER_ID.")
        return None
    
    filename = os.path.basename(file_path)
    print(f"☁️ Đang upload {filename} lên Google Drive...")
    try:
        service = get_drive_service()
        file_metadata = {
            'name': filename,
            'parents': [folder_id]
        }
        media = MediaFileUpload(file_path, resumable=True)
        file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        print(f"   ✅ Upload thành công! File ID: {file.get('id')}")
        return file.get('id')
    except Exception as e:
        print(f"   ❌ Lỗi upload file {filename}: {e}")
        return None

# ==========================================
# CÁC HÀM BỔ TRỢ & TỐI ƯU VRAM
# ==========================================
def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

async def generate_tts(text, output_audio_path, voice="vi-VN-NamMinhNeural"):
    communicator = edge_tts.Communicate(text, voice)
    await communicator.save(output_audio_path)

# ==========================================
# KHỞI TẠO PIPELINE TỐI ƯU (SỬA CẢNH BÁO)
# ==========================================
def load_sdxl_pipeline():
    print("\n🧠 Load SDXL Pipeline (Text → Image)...")
    sdxl = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        token=HF_TOKEN if HF_TOKEN and HF_TOKEN.startswith("hf_") else None,
    )
    sdxl.enable_model_cpu_offload()
    
    # Cú pháp chuẩn mới thay thế cho enable_vae_tiling() bị deprecated
    sdxl.vae.enable_tiling()
    return sdxl

def load_svd_pipeline():
    print("\n🎬 Load Stable Video Diffusion Pipeline (Image → Video)...")
    svd = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        token=HF_TOKEN if HF_TOKEN and HF_TOKEN.startswith("hf_") else None,
    )
    svd.enable_model_cpu_offload()
    return svd

# ==========================================
# TIẾN TRÌNH CHẠY CHÍNH (MAIN PROCESS)
# ==========================================
def main():
    if not os.path.exists("prompt.csv"):
        print("❌ Không tìm thấy tệp prompt.csv trong thư mục!")
        return

    df = pd.read_csv("prompt.csv")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    generated_images = {}

    # -------------------------------------------------------------------
    # GIAI ĐOẠN 1: TẠO & UPLOAD TOÀN BỘ ẢNH TĨNH (SDXL)
    # -------------------------------------------------------------------
    print("\n🎨 --- GIAI ĐOẠN 1: TẠO ẢNH TĨNH TỪ IMAGE_PROMPT (SDXL) ---")
    sdxl = load_sdxl_pipeline()

    for index, row in df.iterrows():
        scene_index = index + 1
        raw_prompt = str(row.get("image_prompt", "")).strip()
        final_prompt = f"{STYLE_PREFIX} {raw_prompt}"
        image_file = f"image_{scene_index:03d}_{timestamp}.png"

        print(f"\n🖼️ [{scene_index}/{len(df)}] Đang tạo ảnh cho Cảnh {scene_index}...")
        
        try:
            clear_vram()
            generator = torch.Generator(device="cuda").manual_seed(42 + scene_index)
            
            image = sdxl(
                prompt=final_prompt,
                negative_prompt=NEGATIVE_PROMPT,
                width=1024,
                height=576,
                num_inference_steps=25,
                guidance_scale=7.0,
                generator=generator,
            ).images[0]

            # 1. Lưu cục bộ trên Kaggle
            image.save(image_file)
            generated_images[scene_index] = image_file
            print(f"   ✅ Đã lưu ảnh: {image_file}")

            # 2. Upload ngay ảnh vừa tạo lên Google Drive
            upload_file_to_drive_fresh(image_file, DRIVE_FOLDER_ID)

        except Exception as e:
            print(f"❌ Lỗi tạo ảnh ở Cảnh {scene_index}: {e}")

    # GIẢI PHÓNG VRAM CỦA SDXL TRƯỚC KHI NẠP SVD
    del sdxl
    clear_vram()

    # -------------------------------------------------------------------
    # GIAI ĐOẠN 2: CHUYỂN ẢNH THÀNH VIDEO & UPLOAD TỪNG CẢNH (SVD + TTS)
    # -------------------------------------------------------------------
    print("\n🎬 --- GIAI ĐOẠN 2: TẠO VIDEO TỪNG CẢNH & GHÉP LỜI THOẠI (SVD) ---")
    svd = load_svd_pipeline()
    rendered_scene_videos = []

    for index, row in df.iterrows():
        scene_index = index + 1
        voice_text = str(row.get("voice_over", "")).strip()
        image_file = generated_images.get(scene_index)

        if not image_file or not os.path.exists(image_file):
            print(f"⚠️ Bỏ qua Cảnh {scene_index} vì không tìm thấy ảnh gốc.")
            continue

        raw_video_file = f"raw_video_{scene_index:03d}_{timestamp}.mp4"
        audio_file = f"audio_{scene_index:03d}_{timestamp}.mp3"
        final_scene_file = f"scene_{scene_index:03d}_{timestamp}.mp4"

        print(f"\n🎥 [{scene_index}/{len(df)}] Đang chuyển ảnh thành video Cảnh {scene_index}...")

        try:
            clear_vram()
            init_image = Image.open(image_file).convert("RGB")
            generator = torch.Generator(device="cuda").manual_seed(100 + scene_index)

            # Render video chuyển động từ ảnh
            frames = svd(
                init_image,
                decode_chunk_size=8,
                motion_bucket_id=127,
                fps=7,
                num_inference_steps=25,
                generator=generator
            ).frames[0]

            # Lưu file video thô
            from diffusers.utils import export_to_video
            export_to_video(frames, raw_video_file, fps=7)

            # Tạo giọng đọc TTS
            asyncio.run(generate_tts(voice_text, audio_file))

            # Ghép Video + Audio bằng FFmpeg
            os.system(
                f"ffmpeg -y -i {raw_video_file} -i {audio_file} "
                f"-c:v copy -c:a aac -shortest {final_scene_file} -loglevel error"
            )

            print(f"   ✅ Đã tạo xong video cảnh: {final_scene_file}")

            # Upload ngay video cảnh vừa ghép lên Google Drive
            upload_file_to_drive_fresh(final_scene_file, DRIVE_FOLDER_ID)
            rendered_scene_videos.append(final_scene_file)

            # Dọn dẹp file tạm không cần thiết để tiết kiệm dung lượng
            if os.path.exists(raw_video_file): os.remove(raw_video_file)
            if os.path.exists(audio_file): os.remove(audio_file)

        except Exception as e:
            print(f"❌ Lỗi tạo video Cảnh {scene_index}: {e}")

    # GIẢI PHÓNG VRAM CỦA SVD
    del svd
    clear_vram()

    # -------------------------------------------------------------------
    # GIAI ĐOẠN 3: NỐI TẤT CẢ CÁC CẢNH THÀNH PHIM HOÀN CHỈNH
    # -------------------------------------------------------------------
    if rendered_scene_videos:
        print("\n🎞️ --- GIAI ĐOẠN 3: GHÉP TOÀN BỘ CÁC CẢNH THÀNH PHIM HOÀN CHỈNH ---")
        concat_list_file = f"concat_list_{timestamp}.txt"
        final_movie_file = f"final_movie_{timestamp}.mp4"

        with open(concat_list_file, "w", encoding="utf-8") as f:
            for video_path in rendered_scene_videos:
                f.write(f"file '{video_path}'\n")

        os.system(
            f"ffmpeg -y -f concat -safe 0 -i {concat_list_file} "
            f"-c copy {final_movie_file} -loglevel error"
        )

        print(f"🎉 ĐÃ HOÀN THÀNH PHIM: {final_movie_file}")
        
        # Upload video phim hoàn chỉnh lên Google Drive
        upload_file_to_drive_fresh(final_movie_file, DRIVE_FOLDER_ID)

        if os.path.exists(concat_list_file): os.remove(concat_list_file)

if __name__ == "__main__":
    main()
