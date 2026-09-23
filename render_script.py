import asyncio
import gc
import os
import subprocess
import sys
import time
from pathlib import Path

def install_requirements():
    packages = [
        "nest_asyncio",
        "diffusers>=0.32.0",
        "transformers",
        "accelerate",
        "bitsandbytes",          # cần cho NF4
        "imageio-ffmpeg",
        "google-api-python-client",
        "google-auth-oauthlib",
        "huggingface_hub",
        "soundfile",
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
    print("📦 Đang kiểm tra và cài đặt packages...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "--no-warn-script-location", "--disable-pip-version-check"
    ] + packages)
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "realesrgan", "basicsr", "facexlib", "gfpgan",
            "--no-warn-script-location", "--disable-pip-version-check"
        ])
    except Exception as e:
        print(f"⚠️ Cài Real-ESRGAN thất bại (sẽ dùng upscale dự phòng): {e}")

install_requirements()

import nest_asyncio
nest_asyncio.apply()

import cv2
import torch
import numpy as np
import pandas as pd
import requests
import edge_tts
from PIL import Image, ImageFilter, ImageEnhance
from diffusers import FluxPipeline, FluxTransformer2DModel, BitsAndBytesConfig
from transformers import T5EncoderModel
from diffusers.utils import export_to_video, load_image
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from huggingface_hub import login

# ---- PATCH tương thích torchvision mới ----
try:
    import torchvision.transforms.functional as TVF
    sys.modules["torchvision.transforms.functional_tensor"] = TVF
except Exception:
    pass

HAS_REALESRGAN = False
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    HAS_REALESRGAN = True
except Exception as e:
    print(f"⚠️ Real-ESRGAN không dùng được: {e}")
    print("→ Sẽ dùng upscale dự phòng (PIL high-quality).")

try:
    from diffusers import LTXImageToVideoPipeline
    HAS_LTX_I2V = True
except ImportError:
    HAS_LTX_I2V = False
    print("⚠️ Không tìm thấy LTXImageToVideoPipeline")

RUN_DATE = time.strftime("%Y%m%d_%H%M%S")
print(f"🚀 FLUX NF4 + UPSCALE CAO + LTX (T4) - [{RUN_DATE}]")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"

# ==================== CẤU HÌNH ====================
FLUX_WIDTH = 512
FLUX_HEIGHT = 288
LTX_WIDTH = 512
LTX_HEIGHT = 288
LTX_NUM_FRAMES = 9
LTX_STEPS = 10
FLUX_STEPS = 4
USE_REALESRGAN = True          # Bật để upscale chất lượng cao
UPSCALE_FACTOR = 4             # ×4 → ~2048×1152 (gần 2K)

def sync_system_time():
    try:
        subprocess.run(
            "apt-get update -qq && apt-get install -y -qq ntpdate && ntpdate time.google.com",
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception:
        pass

sync_system_time()

# -------------------------------------------------------------------
# CONFIG + GHÉP HF TOKEN
# -------------------------------------------------------------------
N8N_WEBHOOK_URL = "https://n8n-latest-namx.onrender.com/webhook/kaggle-video-done"
SHEET_ID = "1DmA-yuPwDl1riceSMGzWPXhuxrL4y987lOZ6Af351l8"
GOOGLE_SHEET_CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
DRIVE_FOLDER_ID = "1oXS7LweDNK2fYsWonQay3U-hUmEIsgCF"

TOKEN_PART1 = os.environ.get("HF_TOKEN_PART1", "hf_elrhByUKOcJWQTDSTcZN")
TOKEN_PART2 = os.environ.get("HF_TOKEN_PART2", "ebmXSLFuIrugmH")
COMBINED_HF_TOKEN = f"{TOKEN_PART1.strip()}{TOKEN_PART2.strip()}".strip()

if COMBINED_HF_TOKEN and COMBINED_HF_TOKEN.startswith("hf_"):
    try:
        login(token=COMBINED_HF_TOKEN)
        print("🔑 Đã ghép token thành công và xác thực với Hugging Face.")
        hf_token_to_pass = COMBINED_HF_TOKEN
    except Exception as e:
        print(f"⚠️ Không thể đăng nhập Hugging Face: {e}")
        hf_token_to_pass = None
else:
    print("⚠️ Token ghép không hợp lệ. Chạy chế độ Public.")
    hf_token_to_pass = None

VOICE_MAP = {
    "nam": "vi-VN-NamMinhNeural",
    "nu": "vi-VN-HoaiMyNeural"
}

OAUTH_CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID", "948179937421-o55enfl61lb8ou0ms2jmrr4dlf1fhgip.apps.googleusercontent.com")
OAUTH_CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET", "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")
OAUTH_REFRESH_TOKEN = os.environ.get("OAUTH_REFRESH_TOKEN", "1//06AsOeOfzvnpxCgYIARAAGAYSNwF-L9Ir-Yoj_gfy3CYDrDfXfUkE0z95bPruk8RMjNG3Y0F-SuDd-VFnFfIo0jRZ4T8J4oZzmek")
TOKEN_URI = "https://oauth2.googleapis.com/token"

def get_oauth_credentials():
    return Credentials(
        token=None,
        refresh_token=OAUTH_REFRESH_TOKEN,
        token_uri=TOKEN_URI,
        client_id=OAUTH_CLIENT_ID,
        client_secret=OAUTH_CLIENT_SECRET,
        scopes=["https://www.googleapis.com/auth/drive"]
    )

def get_media_duration(file_path):
    cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{file_path}"'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        return 2.0

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

def truncate_prompt(prompt, max_words=60):
    words = prompt.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return prompt

# -------------------------------------------------------------------
# UPSCALER (ưu tiên chất lượng cao)
# -------------------------------------------------------------------
def create_upsampler():
    if not HAS_REALESRGAN or not USE_REALESRGAN:
        return None
    try:
        # Dùng model x4 để upscale mạnh
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(
            scale=4,
            model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            model=model,
            tile=200,          # nhỏ để an toàn VRAM
            tile_pad=10,
            pre_pad=0,
            half=True
        )
        return upsampler
    except Exception as e:
        print(f"⚠️ Không tạo được RealESRGANer x4: {e}")
        return None

def upscale_image_realesrgan(upsampler, input_path, output_path):
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    output, _ = upsampler.enhance(img, outscale=UPSCALE_FACTOR)
    cv2.imwrite(output_path, output)
    return output_path

def upscale_image_pil(input_path, output_path, scale=4):
    """Upscale chất lượng cao bằng PIL (LANCZOS + sharpen)"""
    img = Image.open(input_path).convert("RGB")
    new_size = (img.width * scale, img.height * scale)
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.8, percent=140, threshold=2))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.08)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.15)
    img.save(output_path, quality=95)
    return output_path

def upscale_image(upsampler, input_path, output_path):
    if upsampler is not None:
        try:
            return upscale_image_realesrgan(upsampler, input_path, output_path)
        except Exception as e:
            print(f"⚠️ Real-ESRGAN lỗi, chuyển sang PIL: {e}")
    return upscale_image_pil(input_path, output_path, scale=UPSCALE_FACTOR)

# -------------------------------------------------------------------
# 2. ĐỌC SHEETS
# -------------------------------------------------------------------
print("\n📊 2. TẢI DỮ LIỆU TỪ GOOGLE SHEETS...")
try:
    df = pd.read_csv(GOOGLE_SHEET_CSV_URL)
    df.columns = df.columns.str.strip().str.lower()
    total_scenes = len(df)
    print(f"✅ Tìm thấy {total_scenes} cảnh.")
    print(f"📋 Các cột: {df.columns.tolist()}")
except Exception as e:
    print(f"❌ Lỗi đọc Google Sheet: {e}")
    sys.exit(1)

project_info = {
    "title": "Chưa đặt tiêu đề",
    "genre": "Mặc định",
    "character_design": "",
    "world_setting": "",
    "visual_style": ""
}
if not df.empty:
    first_row = df.iloc[0]
    for key in project_info.keys():
        if key in df.columns:
            val = str(first_row[key]).strip()
            if val.lower() not in ["nan", "[empty]", ""]:
                project_info[key] = val

def upload_file_to_drive_fresh(file_path, folder_id, retries=3):
    if not os.path.exists(file_path) or not folder_id:
        return None
    file_name = os.path.basename(file_path)
    mimetype = 'image/png' if file_name.endswith('.png') else 'video/mp4'
    for attempt in range(1, retries + 1):
        try:
            sync_system_time()
            creds = get_oauth_credentials()
            service = build('drive', 'v3', credentials=creds, cache_discovery=False)
            file_metadata = {'name': file_name, 'parents': [folder_id]}
            media = MediaFileUpload(file_path, mimetype=mimetype, resumable=True)
            uploaded_file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            print(f"☁️ Upload thành công: {file_name}")
            return uploaded_file.get('id')
        except Exception as e:
            print(f"⚠️ Upload lần {attempt} thất bại: {e}")
            if attempt < retries:
                time.sleep(2)
            else:
                return None

def send_n8n_final_webhook(status, total_scenes, final_file=None, drive_file_id=None, error_message=None):
    payload = {
        "status": status,
        "total_scenes": total_scenes,
        "final_file": final_file,
        "drive_file_id": drive_file_id,
        "error_message": error_message,
        **project_info
    }
    try:
        res = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=60)
        print(f"📡 Webhook {status.upper()} → {res.status_code}")
    except Exception as e:
        print(f"❌ Lỗi webhook: {e}")

# -------------------------------------------------------------------
# 3. PIPELINE
# -------------------------------------------------------------------
async def process_video_pipeline():
    rendered_files = []
    image_paths = {}

    # ==================== GIAI ĐOẠN 1: FLUX NF4 ====================
    print("\n🧠 [GIAI ĐOẠN 1] Load FLUX.1-schnell NF4 (nhẹ + nhanh)...")
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        transformer = FluxTransformer2DModel.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            subfolder="transformer",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            token=hf_token_to_pass,
        )

        text_encoder_2 = T5EncoderModel.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            subfolder="text_encoder_2",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            token=hf_token_to_pass,
        )

        flux_pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            transformer=transformer,
            text_encoder_2=text_encoder_2,
            torch_dtype=torch.bfloat16,
            token=hf_token_to_pass,
        )

        flux_pipe.enable_model_cpu_offload()
        
        if hasattr(flux_pipe, "enable_attention_slicing"):
            flux_pipe.enable_attention_slicing("auto")
        
        if hasattr(flux_pipe, "vae"):
            flux_pipe.vae.enable_slicing()
            flux_pipe.vae.enable_tiling()
            
        print(f"✅ Load FLUX NF4 thành công! ({FLUX_WIDTH}×{FLUX_HEIGHT})")
    except Exception as e:
        print(f"❌ Lỗi load FLUX NF4: {e}")
        send_n8n_final_webhook("failed", 0, error_message=str(e))
        sys.exit(1)

    print("🔧 Khởi tạo Upscaler (chất lượng cao ×4)...")
    upsampler = create_upsampler()
    if upsampler:
        print("✅ Real-ESRGAN ×4 sẵn sàng!")
    else:
        print("✅ Dùng PIL upscale ×4 (chất lượng cao).")

    for index, row in df.iterrows():
        scene_start = time.time()
        scene_idx_val = row.get("scene_index")
        scene_index = int(scene_idx_val) if pd.notna(scene_idx_val) else index + 1

        img_prompt = str(row.get("image_prompt", "")).strip()
        if not img_prompt or img_prompt.lower() in ["nan", "none", "null"]:
            print(f"⚠️ Bỏ qua cảnh {scene_index}: không có image_prompt")
            continue

        existing = list(Path('.').glob(f"image_{scene_index:03d}_*.png"))
        if existing:
            image_file = str(existing[0])
            image_paths[scene_index] = image_file
            print(f"⏩ Ảnh {scene_index} đã có → bỏ qua")
            continue

        raw_image_file = f"raw_image_{scene_index:03d}_{RUN_DATE}.png"
        image_file = f"image_{scene_index:03d}_{RUN_DATE}.png"
        image_paths[scene_index] = image_file

        print(f"🖼️ [{scene_index}/{total_scenes}] Sinh ảnh FLUX NF4...")
        try:
            clear_memory()
            generator = torch.Generator(device="cpu").manual_seed(42 + scene_index)
            safe_prompt = truncate_prompt(img_prompt, max_words=60)

            image = flux_pipe(
                prompt=safe_prompt,
                width=FLUX_WIDTH,
                height=FLUX_HEIGHT,
                num_inference_steps=FLUX_STEPS,
                guidance_scale=0.0,
                generator=generator,
                max_sequence_length=256,
            ).images[0]

            image.save(raw_image_file)
            print(f"   ✅ Ảnh thô: {raw_image_file}")

            print(f"   🔍 Upscaling ×{UPSCALE_FACTOR} (chất lượng cao)...")
            upscale_image(upsampler, raw_image_file, image_file)

            if os.path.exists(raw_image_file):
                os.remove(raw_image_file)

            print(f"   ✅ Ảnh nét cao: {image_file}")
            upload_file_to_drive_fresh(image_file, DRIVE_FOLDER_ID)
            
            del image
            clear_memory()

        except Exception as e:
            print(f"❌ Lỗi ảnh cảnh {scene_index}: {e}")
            clear_memory()

        print(f"⏱️ Cảnh {scene_index} (ảnh) xong trong {time.time() - scene_start:.1f}s")

    print("\n🧹 Xóa FLUX khỏi bộ nhớ...")
    del flux_pipe
    if upsampler is not None:
        del upsampler
    clear_memory()
    time.sleep(3)

    # ==================== GIAI ĐOẠN 2: LTX I2V ====================
    if not HAS_LTX_I2V:
        print("❌ Không có LTXImageToVideoPipeline. Dừng.")
        send_n8n_final_webhook("failed", 0, error_message="Missing LTXImageToVideoPipeline")
        sys.exit(1)

    print("\n🧠 [GIAI ĐOẠN 2] Load LTX Image-to-Video...")
    try:
        ltx_pipe = LTXImageToVideoPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            torch_dtype=torch.bfloat16,
            token=hf_token_to_pass,
        )
        ltx_pipe.enable_model_cpu_offload()
        print(f"✅ Load LTX I2V thành công! ({LTX_WIDTH}×{LTX_HEIGHT}, {LTX_NUM_FRAMES}f, {LTX_STEPS} steps)")
    except Exception as e:
        print(f"❌ Lỗi load LTX: {e}")
        send_n8n_final_webhook("failed", 0, error_message=str(e))
        sys.exit(1)

    for index, row in df.iterrows():
        scene_start = time.time()
        scene_idx_val = row.get("scene_index")
        scene_index = int(scene_idx_val) if pd.notna(scene_idx_val) else index + 1

        image_file = image_paths.get(scene_index)
        if not image_file or not os.path.exists(image_file):
            print(f"⚠️ Thiếu ảnh cảnh {scene_index} → bỏ qua")
            continue

        vid_prompt = str(row.get("video_prompt", "")).strip()
        neg_prompt = str(row.get("negative_prompt", "")).strip()
        dialogue_text = str(row.get("dialogue", "")).strip()
        char_name = str(row.get("character_name", "")).strip()

        if dialogue_text.lower() in ["nan", "[empty]", "none", "null"]:
            dialogue_text = ""

        voice_to_use = VOICE_MAP["nam"]
        if any(kw in char_name.lower() for kw in ["nữ", "chị", "muội", "cô"]):
            voice_to_use = VOICE_MAP["nu"]

        raw_video_file = f"raw_scene_{scene_index:03d}_{RUN_DATE}.mp4"
        audio_scene_file = f"audio_scene_{scene_index:03d}_{RUN_DATE}.mp3"
        final_scene_file = f"scene_{scene_index:03d}_{RUN_DATE}.mp4"

        existing_scenes = list(Path('.').glob(f"scene_{scene_index:03d}_*.mp4"))
        if existing_scenes and os.path.getsize(str(existing_scenes[0])) > 10000:
            found = str(existing_scenes[0])
            print(f"⏩ Video {scene_index} đã có → bỏ qua")
            rendered_files.append(found)
            continue

        print(f"\n🎬 [{scene_index}/{total_scenes}] LTX Image → Video...")
        try:
            clear_memory()
            # Resize ảnh đã upscale về size LTX
            image_input = load_image(image_file).resize((LTX_WIDTH, LTX_HEIGHT))

            motion_prompt = vid_prompt if vid_prompt and vid_prompt.lower() not in ["nan", "none"] else "smooth cinematic movement, gentle wind"

            video_frames = ltx_pipe(
                image=image_input,
                prompt=motion_prompt,
                negative_prompt=neg_prompt if neg_prompt else None,
                width=LTX_WIDTH,
                height=LTX_HEIGHT,
                num_frames=LTX_NUM_FRAMES,
                num_inference_steps=LTX_STEPS,
                generator=torch.Generator("cpu").manual_seed(42 + scene_index),
            ).frames[0]

            export_to_video(video_frames, raw_video_file, fps=12)
            print(f"   ✅ Video thô: {raw_video_file}")

            del video_frames
            clear_memory()

        except Exception as e:
            print(f"❌ Lỗi video cảnh {scene_index}: {e}")
            clear_memory()
            continue

        if dialogue_text:
            print(f"🎙️ Voice [{char_name}]: {dialogue_text[:40]}...")
            communicate = edge_tts.Communicate(text=dialogue_text, voice=voice_to_use)
            await communicate.save(audio_scene_file)

            v_dur = get_media_duration(raw_video_file)
            a_dur = get_media_duration(audio_scene_file)
            pad_dur = max(0.0, a_dur - v_dur)

            if pad_dur > 0:
                mix_cmd = (
                    f'ffmpeg -y -i "{raw_video_file}" -i "{audio_scene_file}" '
                    f'-filter_complex "[0:v]tpad=stop_mode=clone:stop_duration={pad_dur:.3f}[v]" '
                    f'-map "[v]" -map 1:a:0 -c:v libx264 -pix_fmt yuv420p -r 12 '
                    f'-c:a aac -ar 44100 -ac 2 -b:a 192k "{final_scene_file}"'
                )
            else:
                mix_cmd = (
                    f'ffmpeg -y -i "{raw_video_file}" -i "{audio_scene_file}" '
                    f'-map 0:v:0 -map 1:a:0 -c:v libx264 -pix_fmt yuv420p -r 12 '
                    f'-c:a aac -ar 44100 -ac 2 -b:a 192k -shortest "{final_scene_file}"'
                )
            subprocess.run(mix_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            silent_cmd = (
                f'ffmpeg -y -i "{raw_video_file}" -f lavfi -i anullsrc=r=44100:cl=stereo '
                f'-c:v libx264 -pix_fmt yuv420p -r 12 -c:a aac -ar 44100 -ac 2 -b:a 192k '
                f'-shortest "{final_scene_file}"'
            )
            subprocess.run(silent_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        for f in [raw_video_file, audio_scene_file]:
            if os.path.exists(f):
                os.remove(f)

        upload_file_to_drive_fresh(final_scene_file, DRIVE_FOLDER_ID)
        rendered_files.append(final_scene_file)

        print(f"⏱️ Cảnh {scene_index} hoàn thành trong {time.time() - scene_start:.1f} giây")

    del ltx_pipe
    clear_memory()

    # ==================== GIAI ĐOẠN 3: GỘP + BGM ====================
    print("\n🎞️ Gộp các cảnh...")
    if not rendered_files:
        send_n8n_final_webhook("failed", 0, error_message="Không render được cảnh nào.")
        sys.exit(1)

    with open("file_list.txt", "w") as f:
        for file in rendered_files:
            f.write(f"file '{file}'\n")

    concat_output = f"final_concat_{RUN_DATE}.mp4"
    final_output = f"final_movie_{RUN_DATE}.mp4"

    subprocess.run(f"ffmpeg -y -f concat -safe 0 -i file_list.txt -c copy {concat_output}", shell=True, check=True)

    bgm_file = "bgm_xianxia.mp3"
    if os.path.exists(bgm_file):
        bgm_cmd = (
            f'ffmpeg -y -i {concat_output} -stream_loop -1 -i {bgm_file} '
            f'-filter_complex "[0:a]volume=1.2[v_tts];[1:a]volume=0.15[v_bgm];'
            f'[v_tts][v_bgm]amix=inputs=2:duration=first[a]" '
            f'-map 0:v:0 -map "[a]" -c:v copy -c:a aac -ar 44100 -ac 2 -b:a 192k {final_output}'
        )
        try:
            subprocess.run(bgm_cmd, shell=True, check=True)
        except Exception:
            final_output = concat_output
    else:
        final_output = concat_output

    print(f"\n☁️ Upload phim hoàn chỉnh...")
    drive_file_id = upload_file_to_drive_fresh(final_output, DRIVE_FOLDER_ID)

    send_n8n_final_webhook(
        status="completed_all",
        total_scenes=len(rendered_files),
        final_file=final_output,
        drive_file_id=drive_file_id
    )
    print("\n🎉 HOÀN TẤT PIPELINE!")

if __name__ == "__main__":
    try:
        asyncio.run(process_video_pipeline())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(process_video_pipeline())
