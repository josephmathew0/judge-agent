from __future__ import annotations
import json
from pathlib import Path
import typer

from judge_agent.pipeline import judge

app = typer.Typer(help="Judge agent: AI vs human, virality score, and audience distribution.")

@app.command()
def text(
    path: str = typer.Option(..., help="Path to a text file."),
    out: str = typer.Option(None, help="Optional output JSON file path."),
    debug: bool = typer.Option(False, help="Include debug features in output."),
):
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    result = judge(text=txt, include_debug=debug)
    payload = json.dumps(result.model_dump(), indent=2)

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(payload, encoding="utf-8")

    print(payload)

@app.command()
def video(
    path: str = typer.Option(..., help="Path to a video file."),
    transcript: str = typer.Option(None, help="Optional transcript file (txt)."),
    fps_sample: float = typer.Option(1.0, help="Frames per second to sample."),
    max_frames: int = typer.Option(60, help="Max frames to analyze."),
    out: str = typer.Option(None, help="Optional output JSON file path."),
    debug: bool = typer.Option(False, help="Include debug features in output."),
):
    result = judge(
        video_path=path,
        transcript_path=transcript,
        fps_sample=fps_sample,
        max_frames=max_frames,
        include_debug=debug,
    )
    payload = json.dumps(result.model_dump(), indent=2)

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(payload, encoding="utf-8")

    print(payload)
