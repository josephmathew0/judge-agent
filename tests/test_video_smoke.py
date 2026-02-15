import pytest
from judge_agent.pipeline import judge

@pytest.mark.skipif(True, reason="Provide a sample video locally to run this smoke test.")
def test_video_runs():
    out = judge(video_path="sample.mp4")
    assert out.virality_score >= 0
