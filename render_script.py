# ========== 1 CELL — LTX nhanh Kaggle ==========
import os, sys, json, time, gc, requests, subprocess
import pandas as pd
import torch

TOTAL_SCENES = 100          # test nhanh: đổi = 5
OUTPUT_DIR, VIDEO_DIR = "./output_scenes", "./output_videos"
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

# --- LTX NHANH (T4): ~5s/cảnh, steps thấp ---
LTX_MODEL = "Lightricks/LTX-Video"
LTX_WIDTH, LTX_HEIGHT = 768, 512
LTX_NUM_FRAMES = 121      # ~5s @24fps
LTX_STEPS = 15            # nhanh (18–25 đẹp hơn nhưng chậm)
LTX_FPS = 24
LTX_NEG = "worst quality, blurry, jittery, distorted, still image, frozen, slideshow, no movement"

# ========== DRIVE ==========
def get_token():
    if not all([CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN]):
        return None
    try:
        return requests.post("https://oauth2.googleapis.com/token", data={
            "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET,
            "refresh_token": REFRESH_TOKEN, "grant_type": "refresh_token"
        }, timeout=20).json().get("access_token")
    except Exception as e:
        print("OAuth:", e); return None

def drive_map():
    tok = get_token()
    if not tok or not DRIVE_FOLDER_ID: return {}
    q = f"'{DRIVE_FOLDER_ID}' in parents and trashed=false"
    url = f"https://www.googleapis.com/drive/v3/files?q={requests.utils.quote(q)}&fields=files(id,name)&pageSize=1000"
    try:
        files = requests.get(url, headers={"Authorization": f"Bearer {tok}"}, timeout=30).json().get("files", [])
        print(f"📁 Drive: {len(files)} file")
        return {f["name"]: f["id"] for f in files}
    except Exception as e:
        print("Drive list:", e); return {}

def drive_upload(path):
    tok = get_token()
    if not tok or not DRIVE_FOLDER_ID or not os.path.exists(path) or os.path.getsize(path) < 1000:
        return None
    name = os.path.basename(path)
    meta = {"name": name, "parents": [DRIVE_FOLDER_ID]}
    print(f"🚀 Upload {name}...")
    try:
        with open(path, "rb") as f:
            r = requests.post(
                "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
                headers={"Authorization": f"Bearer {tok}"},
                files={"data": ("metadata", json.dumps(meta), "application/json; charset=UTF-8"),
                       "file": (name, f, "application/octet-stream")},
                timeout=600,
            )
        if r.status_code in (200, 201):
            print(f"✅ {name}"); return r.json()
        print("Upload fail", r.status_code, r.text[:200])
    except Exception as e:
        print("Upload:", e)
    return None

def drive_dl(fid, path):
    tok = get_token()
    if not tok: return False
    try:
        r = requests.get(f"https://www.googleapis.com/drive/v3/files/{fid}?alt=media",
                         headers={"Authorization": f"Bearer {tok}"}, timeout=180)
        if r.status_code == 200 and len(r.content) > 1000:
            open(path, "wb").write(r.content); return True
    except Exception as e:
        print("DL:", e)
    return False

# ========== SHEET ==========
def safe_get(row, *keys, default=""):
    for k in keys:
        v = row.get(k)
        if pd.notna(v) and str(v).strip().lower() not in ("nan", "none", "null", "", "[empty]"):
            return str(v).strip()
    return default

def sheet_csv(gid):
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={gid}"
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip().str.lower()
        return df.dropna(how="all")
    except Exception as e:
        print("Sheet:", e); return pd.DataFrame()

def video_prompt(row):
    motion = ("Smooth camera move, character body motion, cloth/hair movement, "
              "cinematic, not still, not frozen, not slideshow.")
    vp = safe_get(row, "video_prompt", "video prompt", "prompt")
    ip = safe_get(row, "image_prompt", "scene_description", "scene description")
    if vp: return f"{vp}. {motion}"
    if ip: return f"{ip}. {motion}"
    return f"Cinematic push-in, subtle motion. {motion}"

def dialogue(row):
    return safe_get(row, "dialogue")[:160]

# ========== CÀI NHẸ (chỉ khi thiếu) ==========
def ensure_pkgs():
    need_install = False
    try:
        import diffusers  # noqa
        from diffusers import LTXImageToVideoPipeline  # noqa
    except Exception:
        need_install = True

    if need_install:
        print("📦 Cài nhẹ diffusers...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao", "diffusers"], check=False)
        subprocess.run("rm -rf /usr/local/lib/python3.13/dist-packages/diffusers*", shell=True, check=False)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "--no-deps", "diffusers==0.32.2"], check=False)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
                        "transformers==4.46.3", "safetensors", "sentencepiece",
                        "imageio", "imageio-ffmpeg", "edge-tts"], check=False)
    else:
        print("✅ Package đã có — bỏ qua pip")

    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        subprocess.run("apt-get update -qq && apt-get install -y -qq ffmpeg", shell=True, check=False)

def patch_imports():
    for k in list(sys.modules):
        if k == "diffusers" or k.startswith("diffusers.") or k == "transformers" or k.startswith("transformers."):
            del sys.modules[k]
    import transformers.utils as tu
    if not hasattr(tu, "FLAX_WEIGHTS_NAME"):
        tu.FLAX_WEIGHTS_NAME = "flax_model.msgpack"
    if not hasattr(tu, "SAFE_WEIGHTS_NAME"):
        tu.SAFE_WEIGHTS_NAME = "model.safetensors"
    if not hasattr(tu, "SAFE_WEIGHTS_INDEX_NAME"):
        tu.SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
    import diffusers.utils as du
    for n, f in [("is_flax_available", lambda: False), ("is_bs4_available", lambda: False), ("is_ftfy_available", lambda: False)]:
        if not hasattr(du, n):
            setattr(du, n, f)

def load_pipe():
    patch_imports()
    from diffusers import LTXImageToVideoPipeline
    from diffusers.utils import logging
    logging.set_verbosity_error()
    print("🧠 Load LTX...")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipe = LTXImageToVideoPipeline.from_pretrained(LTX_MODEL, torch_dtype=dtype)
    if torch.cuda.is_available():
        try: pipe.enable_model_cpu_offload()
        except Exception: pipe.to("cuda")
        try: pipe.vae.enable_tiling()
        except Exception: pass
    print("✅ LTX ready")
    return pipe

def tts(text, path):
    if not text or len(text) < 2: return False
    if os.path.exists(path) and os.path.getsize(path) > 400: return True
    safe = text.replace('"', "'").replace("\n", " ")
    try:
        subprocess.run(
            f'edge-tts --text "{safe}" --voice vi-VN-NamMinhNeural --rate=+10% --write-media "{path}"',
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=45)
        return os.path.exists(path) and os.path.getsize(path) > 400
    except Exception:
        return False

def mix_av(video, audio, out):
    if not (os.path.exists(audio) and os.path.getsize(audio) > 400):
        if video != out and os.path.exists(video):
            subprocess.run(f'cp -f "{video}" "{out}"', shell=True)
        return out if os.path.exists(out) else video
    subprocess.run(
        f'ffmpeg -y -hide_banner -loglevel error -i "{video}" -i "{audio}" '
        f'-c:v copy -c:a aac -b:a 96k -shortest -movflags +faststart "{out}"',
        shell=True)
    return out if os.path.exists(out) and os.path.getsize(out) > 5000 else video

# ========== MAIN ==========
print("DRIVE_FOLDER_ID =", DRIVE_FOLDER_ID or "(EMPTY)")
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

# --- Giai đoạn 1: ảnh ---
print("\n=== GIAI ĐOẠN 1: Ảnh ===")
dmap = drive_map()
df = sheet_csv(GID_SCENES)
if df.empty:
    raise SystemExit("❌ Sheet rỗng")
print("Cột:", list(df.columns))
ok = 0
for idx, row in df.iterrows():
    n = int(row["scene_index"]) if pd.notna(row.get("scene_index")) else idx + 1
    if n > TOTAL_SCENES: break
    name, local = f"scene_{n:03d}.png", f"{OUTPUT_DIR}/scene_{n:03d}.png"
    if name in dmap:
        if not (os.path.exists(local) and os.path.getsize(local) > 2000):
            drive_dl(dmap[name], local)
        ok += 1
    elif os.path.exists(local) and os.path.getsize(local) > 2000:
        ok += 1
    else:
        print(f"⚠️ thiếu ảnh {name}")
print(f"Ảnh OK: {ok}")

# --- Giai đoạn 2: LTX ---
print(f"\n=== GIAI ĐOẠN 2: LTX {LTX_WIDTH}x{LTX_HEIGHT} f={LTX_NUM_FRAMES} steps={LTX_STEPS} ===")
ensure_pkgs()
pipe = load_pipe()
from diffusers.utils import export_to_video, load_image

dmap = drive_map()
rendered = []
t0 = time.time()

for idx, row in df.iterrows():
    n = int(row["scene_index"]) if pd.notna(row.get("scene_index")) else idx + 1
    if n > TOTAL_SCENES: break

    img = f"{OUTPUT_DIR}/scene_{n:03d}.png"
    raw = f"{VIDEO_DIR}/ltx_{n:03d}.mp4"
    mp3 = f"{VIDEO_DIR}/voice_{n:03d}.mp3"
    out = f"{VIDEO_DIR}/scene_{n:03d}.mp4"
    vname = f"scene_{n:03d}.mp4"

    # Skip video đã có (>800KB = LTX thật)
    if os.path.exists(out) and os.path.getsize(out) > 800_000:
        print(f"⏩ [{n}] local"); rendered.append(out); continue
    if vname in dmap:
        if drive_dl(dmap[vname], out) and os.path.getsize(out) > 800_000:
            print(f"⏩ [{n}] Drive"); rendered.append(out); continue

    if not (os.path.exists(img) and os.path.getsize(img) > 2000):
        print(f"⚠️ [{n}] no image"); continue

    prompt = video_prompt(row)
    print(f"🎥 [{n}/{TOTAL_SCENES}] {prompt[:80]}...")
    try:
        t1 = time.time()
        image = load_image(img)
        gen = torch.Generator(device="cpu").manual_seed(42 + n)
        res = pipe(
            image=image, prompt=prompt, negative_prompt=LTX_NEG,
            width=LTX_WIDTH, height=LTX_HEIGHT,
            num_frames=LTX_NUM_FRAMES, num_inference_steps=LTX_STEPS,
            generator=gen,
        )
        export_to_video(res.frames[0], raw, fps=LTX_FPS)
        tts(dialogue(row), mp3)
        final = mix_av(raw, mp3, out)
        sz = os.path.getsize(final) if os.path.exists(final) else 0
        print(f"✅ [{n}] {time.time()-t1:.0f}s | {sz/1024:.0f}KB")
        if sz > 500_000:
            drive_upload(final)
            rendered.append(final)
        del res; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ [{n}] {e}")
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

print(f"\nRender xong {len(rendered)} cảnh / {(time.time()-t0)/60:.1f} phút")

# --- Ghép phim ---
valid = [v for v in sorted(rendered) if os.path.exists(v) and os.path.getsize(v) > 500_000]
if valid:
    with open("concat_list.txt", "w") as f:
        for v in valid:
            f.write(f"file '{os.path.abspath(v)}'\n")
    subprocess.run(
        f'ffmpeg -y -hide_banner -loglevel error -f concat -safe 0 -i concat_list.txt '
        f'-c copy -movflags +faststart "{FINAL_VIDEO}"', shell=True)
    if os.path.exists(FINAL_VIDEO):
        print(f"🎉 {FINAL_VIDEO} {os.path.getsize(FINAL_VIDEO)/1024/1024:.1f}MB")
        drive_upload(FINAL_VIDEO)
else:
    print("❌ Không đủ video để ghép")

print("DONE")
