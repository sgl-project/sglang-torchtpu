"""Check the actual delivered standard H3 MP4, including complete decode."""

import argparse
import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path


def validate_media(path: Path, *, width: int = 1344, height: int = 768) -> dict:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    metadata = json.loads(probe.stdout)
    videos = [s for s in metadata["streams"] if s.get("codec_type") == "video"]
    audios = [s for s in metadata["streams"] if s.get("codec_type") == "audio"]
    if len(videos) != 1 or len(audios) != 1:
        raise ValueError("Expected exactly one video stream and one audio stream")
    video, audio = videos[0], audios[0]
    expected = {
        "codec_name": "h264",
        "width": width,
        "height": height,
        "nb_read_frames": "107",
    }
    for key, value in expected.items():
        if video.get(key) != value:
            raise ValueError(f"Video {key}: expected {value}, got {video.get(key)}")
    if Fraction(video["avg_frame_rate"]) != 24:
        raise ValueError("Expected 24 fps")
    if audio.get("codec_name") != "aac" or audio.get("sample_rate") != "32000":
        raise ValueError("Expected AAC at 32 kHz")
    if audio.get("channels") != 2:
        raise ValueError("Expected stereo audio")
    video_duration, audio_duration = float(video["duration"]), float(audio["duration"])
    if abs(video_duration - 107 / 24) > 1 / 24:
        raise ValueError("Video duration does not match delivered frame count")
    if abs(video_duration - audio_duration) > 0.25:
        raise ValueError("Audio/video drift exceeds 0.25 seconds")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        check=True,
    )
    return {
        "status": "PASS_MEDIA_ONLY",
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "video_duration": video_duration,
        "audio_duration": audio_duration,
        "av_drift_seconds": abs(video_duration - audio_duration),
        "ffprobe": metadata,
        "complete_decode": "PASS",
        "quality_or_numerical_equivalence": "NOT_TESTED",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = validate_media(args.video)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "ffprobe"}, indent=2))


if __name__ == "__main__":
    main()
