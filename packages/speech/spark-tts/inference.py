#!/usr/bin/env python3
import os
import torch
import logging
import argparse
import soundfile as sf

from datetime import datetime
from cli.SparkTTS import SparkTTS
from huggingface_hub import snapshot_download

# Use standard TRANSFORMERS_CACHE env var that's set in Dockerfile
cache_dir = os.environ.get("TRANSFORMERS_CACHE", "/data/models/huggingface")
os.makedirs(cache_dir, exist_ok=True)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run TTS inference.")
    # Update default model dir to use cache_dir
    parser.add_argument("--model_dir", type=str, default=os.path.join(cache_dir, "SparkAudio--Spark-TTS-0.5B"), help="Path to the model directory")
    parser.add_argument("--save_dir", type=str, default="/data/audio/tts/spark-tts", help="Directory to save generated audio files")
    parser.add_argument("--device", type=int, default=0, help="CUDA device number")
    parser.add_argument("--text", type=str, default="Hi, I'm Spark-TTS", help="Text for TTS generation")
    parser.add_argument("--prompt_text", type=str, help="Transcript of prompt audio for Voice Cloning")
    parser.add_argument("--prompt_speech_path", type=str, help="Path to the prompt audio file for Voice Cloning")
    parser.add_argument("--gender", choices=["male", "female"], default="male")
    parser.add_argument("--pitch", choices=["very_low", "low", "moderate", "high", "very_high"], default="moderate")
    parser.add_argument("--speed", choices=["very_low", "low", "moderate", "high", "very_high"], default="moderate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=float, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    return parser.parse_args()

def run_tts(args) -> None:
    logging.info(f"Using model from: {args.model_dir}")
    logging.info(f"Saving audio to: {args.save_dir} (under jetson-containers/data/)")
    logging.info(f"Args: {args}")

    if (not args.prompt_speech_path or not args.prompt_text) and (not args.gender or not args.pitch or not args.speed):
        raise ValueError("Please provide --gender, --pitch and --speed if not using Voice Cloning!")

    # Check if model already exists
    if os.path.exists(args.model_dir) and len(os.listdir(args.model_dir)) > 0:
        print("Model files already exist. Skipping download.")
    else:
        print(f"Downloading model files to {args.model_dir}...")
        snapshot_download(
            repo_id="SparkAudio/Spark-TTS-0.5B",
            local_dir=args.model_dir,
            resume_download=True,
        )
        print("Download complete!")

    # Ensure the save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    # Convert device argument to torch.device
    device = torch.device(f"cuda:{args.device}")

    # Initialize the model
    model = SparkTTS(args.model_dir, device)

    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(args.save_dir, f"{timestamp}.wav")

    logging.info("Starting inference...")

    # Perform inference and save the output audio
    with torch.no_grad():
        wav = model.inference(
            args.text,
            args.prompt_speech_path,
            prompt_text=args.prompt_text,
            gender=args.gender,
            pitch=args.pitch,
            speed=args.speed,
        )
        sf.write(save_path, wav, samplerate=16000)

    logging.info(f"Audio file saved to jetson-containers/data/audio/tts/spark-tts/{os.path.basename(save_path)}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    run_tts(args)
