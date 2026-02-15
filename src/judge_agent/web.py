from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from judge_agent.pipeline import judge

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="Judge Agent Web")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/judge")
async def judge_endpoint(
    file: UploadFile = File(...),
    content_type: str = Form(...),  # "text" or "video"
    transcript: Optional[UploadFile] = File(None),
    fps_sample: float = Form(1.0),
    max_frames: int = Form(60),
    debug: bool = Form(False),
):
    # Save uploads to temp files
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)

        in_name = Path(file.filename or "input.bin").name
        in_path = td_path / in_name
        in_path.write_bytes(await file.read())

        transcript_path = None
        if transcript is not None:
            tr_name = Path(transcript.filename or "transcript.txt").name
            transcript_path = td_path / tr_name
            transcript_path.write_bytes(await transcript.read())

        if content_type == "text":
            text = in_path.read_text(encoding="utf-8", errors="ignore")
            out = judge(text=text, include_debug=debug)
        elif content_type == "video":
            out = judge(
                video_path=str(in_path),
                transcript_path=str(transcript_path) if transcript_path else None,
                fps_sample=float(fps_sample),
                max_frames=int(max_frames),
                include_debug=debug,
            )
        else:
            return JSONResponse({"error": "content_type must be 'text' or 'video'."}, status_code=400)

        return JSONResponse(out.model_dump())

def main():
    import uvicorn
    uvicorn.run("judge_agent.web:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()
