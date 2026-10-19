# ========== FULL PIPELINE 1 GPU T4: Text-to-Video ~5s (LTX) → TTS → Drive → Merge ==========
# Đã sửa: Photorealistic + Medium shot + Ít méo mặt + Chuyển động tự nhiên hơn
import os
import sys
import json
import time
import gc
import requests
import subprocess
import pandas as pd

def install_all_deps():
    print("📦 Đang kiểm tra dependency...")
    try:
        import ftfy
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "ftfy"], check=False)
    try:
        from diffusers import LTXPipeline
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q",
            "diffusers>=0.32.0", "transformers>=4.46.0", "accelerate>=1.1.0",
            "sentencepiece", "imageio", "imageio-ffmpeg", "edge-tts",
            "safetensors", "huggingface_hub", "protobuf<6", "opencv-python-headless", "ftfy"
        ], check=False)
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        subprocess.run("apt-get update -qq && apt-get install -y -qq ffmpeg", shell=True, check=False)

install_all_deps()

import torch
import ftfy
from diffusers.utils import export_to_video

# ==========================================
# CẤU HÌNH TỐI ƯU CHO T4 + VIDEO ~5 GIÂY
# ==========================================
TOTAL_SCENES = 100
VIDEO_DIR = "./output_videos"
FINAL_VIDEO = "final_full_movie.mp4"
os.makedirs(VIDEO_DIR, exist_ok=True)

DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID", "").strip()
CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET", "")
REFRESH_TOKEN = os.environ.get("OAUTH_REFRESH_TOKEN", "")

SHEET_ID = os.environ.get("SHEET_ID", "1DmA-yuPwDl1riceSMGzWPXhuxrL4y987lOZ6Af351l8")
GID_CHARACTERS = os.environ.get("GID_CHARACTERS", "1382939846")
GID_SCENES = os.environ.get("GID_SCENES", "0")

DEFAULT_MODEL_ID = "Lightricks/LTX-Video"

# --- Cấu hình tối ưu hơn cho chất lượng + ít méo ---
LTX_WIDTH = 576
LTX_HEIGHT = 576
LTX_NUM_FRAMES = 65          # ~4 giây
LTX_FPS = 16
LTX_STEPS = 28               # Tăng steps → mượt hơn, ít méo mặt

FINAL_FPS = 24
FINAL_WIDTH = 768
FINAL_HEIGHT = 512

NEGATIVE_PROMPT_LOCK = (
    "worst quality, low quality, blurry, distorted face, deformed eyes, bad anatomy, "
    "cartoon, anime, donghua, stylized, 3d animation, unreal engine, cgi, plastic skin, "
    "text, watermark, black screen, extra limbs, morphing, face warp, melting face, "
    "inconsistent face, changed identity, slow motion, stiff pose, frozen, close-up only"
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:32"

# ==========================================
# GOOGLE DRIVE
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
    except:
        return None

def upload_file_to_drive(file_path: str):
    token = get_google_access_token()
    if not token or not DRIVE_FOLDER_ID:
        return None
    if not os.path.exists(file_path) or os.path.getsize(file_path) < 50000:
        return None
    file_name = os.path.basename(file_path)
    metadata = {"name": file_name, "parents": [DRIVE_FOLDER_ID]}
    headers = {"Authorization": f"Bearer {token}"}
    try:
        with open(file_path, "rb") as f:
            files = {
                "data": ("metadata", json.dumps(metadata), "application/json; charset=UTF-8"),
                "file": (file_name, f, "application/octet-stream"),
            }
            res = requests.post(
                "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
                headers=headers, files=files, timeout=600,
            )
        if res.status_code in (200, 201):
            print(f"✅ Backup Drive: {file_name}")
            return res.json()
    except Exception as e:
        print(f"❌ Drive: {e}")
    return None

# ==========================================
# SHEET
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
    except:
        return pd.DataFrame()

def load_character_profiles():
    df_char = get_sheet_csv(GID_CHARACTERS)
    if df_char.empty:
        return {}
    char_prompts = {}
    for idx, row in df_char.iterrows():
        char_id = safe_get(row, "character_id").lower()
        char_name = safe_get(row, "character_name")
        appearance = safe_get(row, "appearance", "design_detail", "costume", "design")
        if char_id:
            char_prompts[char_id] = appearance
        if char_name:
            char_prompts[char_name.lower()] = appearance
    return char_prompts

def build_pure_text_prompt(row, char_prompts) -> str:
    """
    Prompt tối ưu: Medium shot + photorealistic + ít méo + chuyển động tự nhiên
    """
    scene_desc = safe_get(row, "video_prompt", "scene_description", "action", "image_prompt")
    char_key = safe_get(row, "character_id", "character", "character_name", "nhan_vat").lower()
    location = safe_get(row, "location")

    char_details = char_prompts.get(char_key, "")
    if not char_details:
        char_details = "handsome young East Asian man, mid 20s, sharp jawline, detailed eyes, natural skin texture"

    prompt = (
        f"Masterpiece, best quality, highly detailed, photorealistic, realistic human, "
        f"cinematic lighting, natural skin texture, detailed eyes. "
        f"Medium shot, full upper body visible, not extreme close-up. "
        f"Character: {char_details[:140]}. "
        f"Keep the exact same face, hairstyle, clothing and identity. "
        f"Scene: {scene_desc[:180]}. "
        f"Location: {location}. "
        f"Natural body movement, natural blinking, subtle breathing, realistic gestures. "
        f"Dynamic camera: medium shot, slow push-in, gentle orbit, shallow depth of field, "
        f"smooth natural motion, ultra realistic, film grain."
    )
    return prompt[:500]

def get_dialogue(row) -> str:
    return safe_get(row, "dialogue")[:180]

def make_tts(text: str, out_mp3: str) -> bool:
    if not text or len(text.strip()) < 2:
        text = "Cảnh tiếp theo."
    if os.path.exists(out_mp3) and os.path.getsize(out_mp3) > 400:
        return True
    safe = text.replace('"', "'").replace("\n", " ")
    for voice in ["vi-VN-NamMinhNeural", "vi-VN-HoaiMyNeural"]:
        try:
            cmd = f'edge-tts --text "{safe}" --voice {voice} --rate=+5% --write-media "{out_mp3}"'
            subprocess.run(cmd, shell=True, timeout=90, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(out_mp3) and os.path.getsize(out_mp3) > 400:
                return True
        except:
            continue
    return False

def mix_video_audio(video_in, audio_mp3, video_out):
    if not (os.path.exists(audio_mp3) and os.path.getsize(audio_mp3) > 400):
        if video_in != video_out and os.path.exists(video_in):
            subprocess.run(f'cp -f "{video_in}" "{video_out}"', shell=True)
        return video_out if os.path.exists(video_out) else video_in
    cmd = f'ffmpeg -y -hide_banner -loglevel error -i "{video_in}" -i "{audio_mp3}" -c:v copy -c:a aac -b:a 96k -shortest -movflags +faststart "{video_out}"'
    subprocess.run(cmd, shell=True)
    return video_out if os.path.exists(video_out) and os.path.getsize(video_out) > 5000 else video_in

def post_process_video(raw_path, out_path):
    if not os.path.exists(raw_path):
        return raw_path
    cmd = (
        f'ffmpeg -y -hide_banner -loglevel error -i "{raw_path}" '
        f'-vf "scale={FINAL_WIDTH}:{FINAL_HEIGHT}:flags=lanczos,fps={FINAL_FPS}" '
        f'-c:v libx264 -preset fast -crf 17 -pix_fmt yuv420p -movflags +faststart "{out_path}"'
    )
    try:
        subprocess.run(cmd, shell=True, timeout=120)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
            return out_path
    except:
        pass
    return raw_path

def load_ltx_text_pipe():
    from diffusers import LTXPipeline
    from diffusers.utils import logging
    logging.set_verbosity_error()
    print(f"🧠 Load LTX Text-to-Video: {DEFAULT_MODEL_ID}")
    pipe = LTXPipeline.from_pretrained(DEFAULT_MODEL_ID, torch_dtype=torch.float16)
    if torch.cuda.is_available():
        pipe.enable_sequential_cpu_offload()
        print("✅ sequential_cpu_offload")
        try:
            pipe.vae.enable_slicing()
            print("✅ VAE slicing")
        except:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✅ LTX ready (tối ưu T4)")
    return pipe

# ==========================================
# MAIN RENDER
# ==========================================
def run_pipeline():
    char_prompts = load_character_profiles()
    df_scenes = get_sheet_csv(GID_SCENES)
    if df_scenes.empty:
        print("❌ Sheet rỗng")
        return False

    print(f"📄 Đã load {len(df_scenes)} cảnh từ Google Sheet")

    pipe = load_ltx_text_pipe()
    rendered = []
    t0 = time.time()

    for idx, row in df_scenes.iterrows():
        scene_num = int(row["scene_index"]) if pd.notna(row.get("scene_index")) else idx + 1
        if scene_num > TOTAL_SCENES:
            break

        raw_video = f"{VIDEO_DIR}/ltx_raw_{scene_num:03d}.mp4"
        upscaled = f"{VIDEO_DIR}/ltx_up_{scene_num:03d}.mp4"
        voice_mp3 = f"{VIDEO_DIR}/voice_{scene_num:03d}.mp3"
        final_scene = f"{VIDEO_DIR}/scene_{scene_num:03d}.mp4"

        if os.path.exists(final_scene) and os.path.getsize(final_scene) > 50000:
            print(f"⏩ [{scene_num}] Đã có")
            rendered.append(final_scene)
            continue

        text_prompt = build_pure_text_prompt(row, char_prompts)
        dialogue = get_dialogue(row)

        print(f"🎥 [{scene_num}/{TOTAL_SCENES}] ~4-5s Text-to-Video...")
        print(f"   Prompt: {text_prompt[:80]}...")

        try:
            t1 = time.time()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            generator = torch.Generator(device="cpu").manual_seed(42 + scene_num)

            with torch.inference_mode():
                res = pipe(
                    prompt=text_prompt,
                    negative_prompt=NEGATIVE_PROMPT_LOCK,
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

            if sz > 50000:
                upload_file_to_drive(out_file)
                rendered.append(out_file)

            for tmp in [raw_video, upscaled]:
                if os.path.exists(tmp) and tmp != out_file:
                    try:
                        os.remove(tmp)
                    except:
                        pass

            del res
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ [{scene_num}] {e}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nRender xong {len(rendered)} cảnh / {(time.time()-t0)/60:.1f} phút")
    valid = [v for v in sorted(rendered) if os.path.exists(v) and os.path.getsize(v) > 50000]
    if not valid:
        print("❌ Không đủ video")
        return False

    with open("concat_list.txt", "w") as f:
        for v in valid:
            f.write(f"file '{os.path.abspath(v)}'\n")

    subprocess.run(
        f'ffmpeg -y -hide_banner -loglevel error -f concat -safe 0 -i concat_list.txt '
        f'-c copy -movflags +faststart "{FINAL_VIDEO}"',
        shell=True,
    )

    if os.path.exists(FINAL_VIDEO):
        print(f"🎉 Phim hoàn chỉnh: {FINAL_VIDEO} ({os.path.getsize(FINAL_VIDEO)/1024/1024:.1f} MB)")
        upload_file_to_drive(FINAL_VIDEO)
        return True
    return False

if __name__ == "__main__":
    try:
        print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    except:
        pass
    run_pipeline()
