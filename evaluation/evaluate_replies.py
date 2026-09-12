import os
import json
import pandas as pd
from typing import Dict
from src.pipeline import SupportPipeline
from evaluation.llm_judge import LLMJudge
from src.config import get_eval_set_path, RESULTS_PATH

def evaluate_generated_replies(num_samples: int = 50) -> Dict:
    """Evaluates generated RAG replies across verified golden set using LLM-as-a-Judge."""
    golden_path = get_eval_set_path()
    pipeline = SupportPipeline(golden_path=golden_path)
    pipeline.train_pipeline(golden_path=golden_path, exclude_golden_from_retrieval=True)
    judge = LLMJudge()

    df = pd.read_csv(golden_path).head(num_samples)

    scores_list = []
    eval_details = []

    for _, row in df.iterrows():
        msg = row['message']
        res = pipeline.process_message(msg, evaluation_mode=True)
        
        judgement = judge.evaluate_reply(
            customer_message=msg,
            retrieved_evidence=res['retrieved_examples'],
            generated_reply=res['draft_reply']
        )
        
        scores_list.append(judgement['scores'])
        eval_details.append({
            "id": row['id'],
            "message": msg,
            "draft_reply": res['draft_reply'],
            "scores": judgement['scores'],
            "overall_score": judgement['overall_score'],
            "explanations": judgement['explanations']
        })

    avg_scores = {
        "groundedness": round(float(pd.Series([s['groundedness'] for s in scores_list]).mean()), 2),
        "correctness": round(float(pd.Series([s['correctness'] for s in scores_list]).mean()), 2),
        "helpfulness": round(float(pd.Series([s['helpfulness'] for s in scores_list]).mean()), 2),
        "tone": round(float(pd.Series([s['tone'] for s in scores_list]).mean()), 2),
        "safety": round(float(pd.Series([s['safety'] for s in scores_list]).mean()), 2),
        "overall_mean": round(float(pd.Series([s['groundedness'] + s['correctness'] + s['helpfulness'] + s['tone'] + s['safety'] for s in scores_list]).mean() / 5.0), 2)
    }

    results_data = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            results_data = json.load(f)

    results_data["reply_quality_eval"] = {
        "num_evaluated": len(df),
        "rubric_scores_1_to_5": avg_scores,
        "sample_evaluations": eval_details[:3]
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    return results_data["reply_quality_eval"]

if __name__ == "__main__":
    res = evaluate_generated_replies()
    print("=" * 65)
    print("           LLM-AS-A-JUDGE REPLY QUALITY EVALUATION")
    print("=" * 65)
    print(f"Evaluated Samples: {res['num_evaluated']}")
    print("\nRubric Average Scores (1.0 to 5.0 Scale):")
    for criterion, score in res['rubric_scores_1_to_5'].items():
        print(f"  - {criterion:<15}: {score:.2f} / 5.0")
    print("-" * 65)
    print(f"Updated results saved to: {RESULTS_PATH}")
    print("=" * 65)
