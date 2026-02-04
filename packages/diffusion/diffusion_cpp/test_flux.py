#!/usr/bin/env python3
import os
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from stable_diffusion_cpp import StableDiffusion

# ---------------- Config ----------------
MODELS = Path("../models")
MODELS.mkdir(parents=True, exist_ok=True)

# Diffusion (preconverted GGUF). These are public in leejet.
DIFFUSION_FILE = "flux1-schnell-q3_k.gguf"
DIFFUSION_URL  = "https://huggingface.co/leejet/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-q3_k.gguf"

# Public text encoders
CLIP_FILE = ("clip_l.safetensors",
             "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors")
T5_FILE   = ("t5xxl_fp16.safetensors",
             "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors")

# Gated VAE (requires HF token + license accept)
VAE_REPO_ID = "black-forest-labs/FLUX.1-dev"
VAE_FILENAME = "ae.safetensors"
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
# ---------------------------------------


def _download_url(url: str, dest: Path, headers: dict | None = None):
    """Minimal downloader with optional headers (Authorization for gated files)."""
    if dest.exists():
        print(f"Found: {dest}")
        return
    print(f"Downloading {dest.name} ...")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0", **(headers or {})})
    try:
        with urlopen(req) as r, open(dest, "wb") as f:
            f.write(r.read())
        print(f"Saved to {dest}")
    except (HTTPError, URLError) as e:
        raise RuntimeError(f"Failed to download {dest.name} from {url}: {e}")


def _download_hf_file(repo_id: str, filename: str, dest: Path, token: str | None):
    """
    Preferred: use huggingface_hub (handles auth, redirects, etags, resuming).
    Fallback: use raw URL + Authorization header if token present.
    """
    if dest.exists():
        print(f"Found: {dest}")
        return

    # Try huggingface_hub first
    try:
        from huggingface_hub import hf_hub_download
        print(f"Downloading {filename} from {repo_id} via huggingface_hub ...")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,              # None is fine for public; required for gated
            local_dir=str(dest.parent),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        # Ensure final name matches our dest (hf_hub_download may keep original name)
        got = Path(path)
        if got != dest:
            got.rename(dest)
        print(f"Saved to {dest}")
        return
    except ImportError:
        print("huggingface_hub not installed; falling back to direct URL.")
    except Exception as e:
        print(f"huggingface_hub download failed: {e}. Falling back to direct URL.")

    # Fallback: direct URL with token
    # Note: gated repos will return 401 if token missing or license not accepted.
    base = f"https://huggingface.co/{repo_id}/resolve/main/{filename}?download=true"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["Accept"] = "application/octet-stream"
    _download_url(base, dest, headers=headers)


def main():
    diffusion_path = MODELS / DIFFUSION_FILE
    _download_url(DIFFUSION_URL, diffusion_path)

    # Public encoders
    clip_path = MODELS / CLIP_FILE[0]
    t5_path   = MODELS / T5_FILE[0]
    _download_url(CLIP_FILE[1], clip_path)
    _download_url(T5_FILE[1], t5_path)

    # Gated VAE (requires token + license accept)
    vae_path = MODELS / VAE_FILENAME
    _download_hf_file(VAE_REPO_ID, VAE_FILENAME, vae_path, HF_TOKEN)

    # Init FLUX (stable-diffusion.cpp)
    sd = StableDiffusion(
        diffusion_model_path=str(diffusion_path),  # FLUX uses diffusion_model_path
        clip_l_path=str(clip_path),
        t5xxl_path=str(t5_path),
        vae_path=str(vae_path),
        vae_decode_only=True,      # fine for txt2img
        keep_clip_on_cpu=True,     # avoids black images with some T5 builds
    )

    out = sd.generate_image(
        prompt="a lovely cat holding a sign says 'flux.cpp'",
        cfg_scale=1.0,             # recommended for FLUX
        progress_callback=lambda i, n, t: print(f"Completed step: {i} of {n}"),
        # sample_method="euler",    # optional; default picks a good sampler for FLUX
        # seed=42,
    )
    out[0].save("output_flux.png")
    print("Image saved as output_flux.png")
    print(out[0].info)


if __name__ == "__main__":
    main()
