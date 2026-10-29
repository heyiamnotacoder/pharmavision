"""Ask Muse Spark to label ~20 labelled train-split clips and score it against the raters
(laptop only). Answers open decision 3: can it tell supported from unsupported rears?

Usage: uv run python -m spike.muse_clips

Clips are 2-4 s, 15 fps H.264, cut around consensus rears (≥ 2 raters agree) plus no-rear
windows as a control. Responses are cached by (model, prompt version, clip hash), so re-runs
are free. Needs MODEL_API_KEY in .env and `python -m spike.data` first.
"""

import base64
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from spike.data import LABELS, SPLIT, VIDEOS
from spike.rears import Agreement, Clip, Truth, agreement, load_rears, pick_clips

MODEL = "muse-spark-1.3-contributor"  # Sturman clips only; ARCHITECTURE "Cost control"
USD_PER_M = {"input": 0.10, "cached": 0.002, "output": 0.20}  # contributor tier, Oct 30, 2026
OUT = Path("eval/data/spike/muse")
N_PER_KIND: dict[Truth, int] = {"supported": 8, "unsupported": 8, "none": 4}
KEEP_SPLIT_KAPPA = 0.4  # below this, fall back to binary rearing / none

PROMPT_VERSION = "rear-v1"
PROMPT = """\
This clip is 2-4 seconds of a top-view camera over a mouse in an open-field arena (a square box
with walls). Decide whether the mouse rears during the clip.

- rearing_supported: the mouse stands up on its hind legs with its forepaws against an arena wall.
- rearing_unsupported: the mouse stands up on its hind legs without touching a wall.
- none: no rear (walking, sitting still, grooming, sniffing).

From above, a rear shows as the body suddenly looking shorter and rounder while the mouse stays
in place. Answer with the single best label."""

SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["rearing_supported", "rearing_unsupported", "none"]},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["label", "confidence", "rationale"],
    "additionalProperties": False,
}
FORMAT = {"type": "json_schema", "name": "rear_label", "schema": SCHEMA, "strict": True}
TO_TRUTH: dict[str, Truth] = {
    "rearing_supported": "supported",
    "rearing_unsupported": "unsupported",
    "none": "none",
}


def duration_s(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
         str(video)],
        check=True, capture_output=True, text=True,
    )  # fmt: skip
    return float(out.stdout)


def cut(clip: Clip) -> Path:
    dest = OUT / "clips" / f"{clip.video_id}_{clip.start:.3f}.mp4"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-ss", f"{clip.start:.3f}", "-to", f"{clip.end:.3f}",
             "-i", str(VIDEOS / f"{clip.video_id}.mp4"), "-r", "15", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-an", str(dest)],
            check=True,
        )  # fmt: skip
    return dest


def ask(client: OpenAI, clip_path: Path) -> dict[str, Any]:
    data = clip_path.read_bytes()
    key = hashlib.sha256(f"{MODEL}|{PROMPT_VERSION}|".encode() + data).hexdigest()
    cached = OUT / "cache" / f"{key}.json"
    if cached.exists():
        return json.loads(cached.read_text())
    started = time.monotonic()
    video_url = f"data:video/mp4;base64,{base64.b64encode(data).decode()}"
    # input_video is a Meta extension the OpenAI SDK's types lack, so the request is untyped.
    request: dict[str, Any] = {
        "model": MODEL,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT},
                    {"type": "input_video", "video_url": video_url},
                ],
            }
        ],
        "text": {"format": FORMAT},
    }
    response = client.responses.create(**request)
    answer = {
        "output": json.loads(response.output_text),
        "usage": response.usage.model_dump(mode="json") if response.usage else None,
        "seconds": time.monotonic() - started,
    }
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(answer, indent=2) + "\n")
    return answer


def cost_usd(usage: dict[str, Any] | None) -> float:
    if usage is None:
        return 0.0
    cached = usage["input_tokens_details"]["cached_tokens"]
    return (
        (usage["input_tokens"] - cached) * USD_PER_M["input"]
        + cached * USD_PER_M["cached"]
        + usage["output_tokens"] * USD_PER_M["output"]
    ) / 1e6


def main() -> None:
    load_dotenv()
    client = OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"])
    train: list[str] = json.loads(SPLIT.read_text())["train"]
    video_s = min(duration_s(VIDEOS / f"{v}.mp4") for v in train)
    clips = pick_clips(load_rears(LABELS), train, video_s, N_PER_KIND)

    rows: list[dict[str, Any]] = []
    for clip in clips:
        answer = ask(client, cut(clip))
        rows.append(asdict(clip) | {"predicted": TO_TRUTH[answer["output"]["label"]]} | answer)
        print(f"{clip.video_id} {clip.start:8.3f} {clip.truth:12} → {rows[-1]['predicted']}")

    truth = [r["truth"] for r in rows]
    predicted = [r["predicted"] for r in rows]
    rears = [r for r in rows if r["truth"] != "none"]
    split = agreement([r["truth"] for r in rears], [r["predicted"] for r in rears])
    scores: dict[str, Agreement] = {
        "three_way": agreement(truth, predicted),
        "supported_vs_unsupported_on_true_rears": split,
        "rear_vs_none": agreement(
            ["rear" if t != "none" else "none" for t in truth],
            ["rear" if p != "none" else "none" for p in predicted],
        ),
    }
    summary = {
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "clips": len(rows),
        "scores": {name: asdict(score) for name, score in scores.items()},
        "label_set": "three_way"
        if split.kappa is not None and split.kappa >= KEEP_SPLIT_KAPPA
        else "binary_rear_none",
        "cost_usd": sum(cost_usd(r["usage"]) for r in rows),
        "mean_seconds_per_clip": sum(r["seconds"] for r in rows) / len(rows),
        "mean_input_tokens": sum((r["usage"] or {}).get("input_tokens", 0) for r in rows)
        / len(rows),
        "rows": rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
