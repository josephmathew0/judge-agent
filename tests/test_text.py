from judge_agent.pipeline import judge

def test_text_runs():
    out = judge(text="Here are 5 tips to improve focus: 1) Sleep 2) Plan 3) ... Like and subscribe!")
    assert out.virality_score >= 0
    assert out.origin_prediction.label in ["ai_generated", "human_generated"]
