from __future__ import annotations
import subprocess


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stderr}")


def extract_audio(video_path: str, out_wav: str) -> None:
    # Requires ffmpeg installed
    run(["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", out_wav])


def probe(video_path: str) -> dict:
    # Minimal ffprobe metadata
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,duration",
        "-of", "default=noprint_wrappers=1:nokey=0",
        video_path
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        return {}
    meta = {}
    for line in p.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            meta[k.strip()] = v.strip()
    return meta
