"""
Evaluation runner: calls POST /chat for each test case and computes metrics.

Usage:
    python evaluation/run_eval.py [--url http://localhost:8000] [--reranking]
"""

import argparse
import json
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description="Run MedBot evaluation")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument(
        "--reranking", action="store_true", default=True, help="Use reranking"
    )
    parser.add_argument(
        "--no-reranking", action="store_false", dest="reranking", help="Disable reranking"
    )
    args = parser.parse_args()

    # Load test cases
    with open("evaluation/test_cases.json", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"Running evaluation with {len(test_cases)} test cases against {args.url}")
    print(f"Reranking: {args.reranking}")
    print("-" * 60)

    # Call /evaluate endpoint
    payload = {
        "test_cases": test_cases,
        "use_reranking": args.reranking,
    }

    start = time.time()
    req = urllib.request.Request(
        f"{args.url}/evaluate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        results = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - start

    # Save results
    with open("evaluation/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\nEvaluation completed in {elapsed:.1f}s")
    print("=" * 60)

    rm = results["retrieval_metrics"]
    print("\nRetrieval Metrics:")
    print(f"  Precision@5:  {rm['precision_at_5']:.4f}")
    print(f"  Recall@5:     {rm['recall_at_5']:.4f}")
    print(f"  MRR:          {rm['mrr']:.4f}")
    print(f"  NDCG@5:       {rm['ndcg_at_5']:.4f}")

    aq = results["answer_quality_metrics"]
    print("\nAnswer Quality Metrics:")
    print(f"  ROUGE-L:      {aq['rouge_l']:.4f}")
    print(f"  BERTScore F1: {aq['bert_score_f1']:.4f}")

    sm = results["system_metrics"]
    print("\nSystem Metrics:")
    print(f"  Avg Latency:        {sm['avg_latency_ms']:.0f} ms")
    print(f"  P90 Latency:        {sm['p90_latency_ms']:.0f} ms")
    print(f"  Context Hit Rate:   {sm['context_hit_rate']:.4f}")
    print(f"  Fallback Rate:      {sm['fallback_rate']:.4f}")
    print(f"  Cache Hit Rate:     {sm['cache_hit_rate']:.4f}")
    print(f"  Grounded Mode:      {sm['grounded_mode_count']}")
    print(f"  Knowledge Mode:     {sm['knowledge_mode_count']}")

    print("\nPer-Question Results:")
    print(f"{'#':<3} {'Lang':<5} {'Mode':<10} {'Score':<7} {'P@5':<6} {'ROUGE':<7} {'Latency':<10}")
    print("-" * 60)
    for i, q in enumerate(results["per_question"], 1):
        mode = "grounded" if q["context_found"] else "knowledge"
        print(
            f"{i:<3} {q['detected_lang']:<5} {mode:<10} "
            f"{q['top_score']:<7.4f} {q['precision_at_5']:<6.4f} "
            f"{q['rouge_l']:<7.4f} {q['latency_ms']:<10.0f}ms"
        )

    print(f"\nResults saved to evaluation/results.json")


if __name__ == "__main__":
    main()
