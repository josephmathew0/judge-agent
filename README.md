# Judge Agent

This project implements a lightweight judge agent that evaluates text or video and returns:
1. AI-generated vs human-generated prediction
2. Virality score (0-100)
3. Distribution analysis (audience segments + reasons)
4. Concise explanations for each output

The system is intentionally interpretable: it extracts transparent features and applies heuristic scorers.

## What I Built
A CLI application (`judge-agent`) backed by a modular pipeline (`judge(...)`) for text and video inputs.  
Text mode uses readability, lexical diversity, repetition, and structural cues.  
Video mode uses motion, brightness, sharpness, overlay likelihood, and optional transcript-derived text signals.  
I also added a small FastAPI web UI for file upload and result visualization.  
Results are emitted as structured JSON with explanation strings.

## How To Run
### 1) Setup
```bash
git clone <your-repo-url>
cd judge-agent
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2) Video requirements
Install these on your system `PATH`:
- `ffmpeg`
- `ffprobe`

### 3) Run text evaluation
```bash
judge-agent text --path examples/text/sample.txt
```

Debug mode:
```bash
judge-agent text --path examples/text/sample.txt --debug
```

Save output to file:
```bash
judge-agent text --path examples/text/sample.txt --out outputs/sample_output.json
```

### 4) Run video evaluation
```bash
judge-agent video --path /path/to/video.mp4
```

With transcript + debug:
```bash
judge-agent video \
  --path /path/to/video.mp4 \
  --transcript /path/to/transcript.txt \
  --fps-sample 1.0 \
  --max-frames 60 \
  --debug
```

### 5) Run tests
```bash
pytest -q
```

`tests/test_video_smoke.py` is intentionally skipped until you point it to a local sample video.

### 6) Run web UI (optional)
```bash
judge-agent-web
```

Then open `http://127.0.0.1:8000`.

## Demo Inputs
- `examples/text/sample.txt` (typically more human-like)
- `examples/text/sample_aiish.txt` (typically more AI-like)

## Where Video Files Should Be
Video files can be anywhere on your machine.  
Pass either an absolute or relative path in `--path`.  
You can optionally keep samples in `examples/videos/`, but it is not required.

## Assumptions
- Heuristic, explainable scoring is preferred for this exercise.
- Origin detection is probabilistic, not forensic proof.
- Transcript can be user-provided; ASR is not run by default.
- Missing FFmpeg tools reduce video/audio signal coverage.

## What I Would Improve With More Time
- Add a calibrated learned classifier using labeled AI/human text+video data.
- Add ASR/OCR features for richer multimodal understanding.
- Add embedding-based audience clustering, not only format cues.
- Add regression fixtures with expected outputs to prevent scoring drift.

## Project Structure
```text
src/judge_agent/
  cli.py
  web.py
  pipeline.py
  schemas.py
  templates/
  feature_extractors/
  scorers/
  utils/
examples/
tests/
```
