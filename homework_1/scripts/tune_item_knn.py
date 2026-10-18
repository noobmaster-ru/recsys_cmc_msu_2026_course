"""Validation-only experiment. This script never loads or evaluates test events."""

from itertools import product
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from data_utils import FitData
from evaluation import (EvaluationContext, evaluate_model, evaluate_popularity,
                        history_slices, popularity_slices, paired_bootstrap)
from models.item_knn import ItemKNN
from scripts.prepare_data import deterministic_gzip, sha256


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def run_validation_search(root=ROOT, *, batch_size=128):
    root = Path(root)
    protocol = json.loads((root / "reports/evaluation_protocol.json").read_text())
    config = json.loads((root / "configs/item_knn_search.json").read_text())
    if protocol["k"] != 10 or protocol["relevance"]["rating_at_least"] != 4:
        raise ValueError("This preregistered search requires K=10, relevance=rating>=4")
    frames, input_hashes = {}, {}
    # Explicitly read only train and validation, never protocol['splits']['test'].
    for name in ("train", "validation"):
        spec = protocol["splits"][name]
        path = root / spec["path"]
        digest = sha256(path)
        if digest != spec["sha256"]:
            raise ValueError(f"Split checksum mismatch: {name}")
        input_hashes[name] = digest
        frames[name] = pd.read_csv(path, dtype={"user_id": "string", "item_id": "string",
                                               "rating": "int8", "timestamp": "int64"})
    fit = FitData.from_frame(frames["train"], relevance_threshold=4)
    query_time = int(pd.Timestamp(protocol["validation_start_utc"]).timestamp())
    context = EvaluationContext.from_frames(frames["train"], frames["validation"],
        fit_data=fit, k=10, relevance_threshold=4, query_timestamp=query_time)
    expected = protocol["evaluation_diagnostics"]["validation"]
    for key in ("evaluation_users", "catalog_items", "relevant_interactions"):
        if context.diagnostics[key] != expected[key]:
            raise ValueError(f"Evaluation cohort changed: {key}")
    grid = list(product(config["grid"]["n_neighbors"], config["grid"]["shrinkage"]))
    if len(grid) != config["budget"]:
        raise ValueError("Search budget does not match the fixed grid")
    output = root / "reports/item_knn_validation"
    local = root / "data/validation/item_knn"
    output.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    source_files = ["data_utils.py", "evaluation.py", "models/item_knn.py",
                    "scripts/tune_item_knn.py", "configs/item_knn_search.json", "requirements.txt"]
    plan = {"configuration": config, "input_hashes": input_hashes,
            "source_hashes": {name: sha256(root / name) for name in source_files},
            "cohort_fingerprint": context.fingerprint, "cohort": context.diagnostics,
            "query_timestamp": query_time, "test_evaluated": False, "batch_size": batch_size}
    # Persist the space, tie rule and cohort BEFORE fitting the first model.
    write_json(output / "search_plan.json", plan)
    baseline = evaluate_popularity(context, batch_size=batch_size)
    records, best_key, best_result, best_params, best_fit_seconds = [], None, None, None, None
    for position, (neighbors, shrinkage) in enumerate(grid, 1):
        started = perf_counter()
        model = ItemKNN(neighbors, shrinkage).fit(fit)
        fit_seconds = perf_counter() - started
        result = evaluate_model(model, context, batch_size=batch_size)
        record = {"trial": position, "n_neighbors": neighbors, "shrinkage": shrinkage,
                  **result.metrics, "fit_seconds": fit_seconds,
                  "similarity_edges": int(model.neighbors_.nnz)}
        records.append(record)
        pd.DataFrame(records).to_csv(output / "search_results.csv", index=False)
        key = (-result.metrics["ndcg_at_10"], neighbors, shrinkage)
        if best_key is None or key < best_key:
            best_key, best_result = key, result
            best_params = {"n_neighbors": neighbors, "shrinkage": shrinkage}
            best_fit_seconds = fit_seconds
        print(f"[{position:02d}/{len(grid)}] k={neighbors}, shrink={shrinkage:g}: "
              f"NDCG@10={result.metrics['ndcg_at_10']:.6f}, "
              f"Recall@10={result.metrics['recall_at_10']:.6f}", flush=True)
        del model
    comparison = pd.DataFrame([
        {"model": "Popularity (control)", **baseline.metrics, "fit_seconds": 0.0},
        {"model": "ItemKNN (selected)", **best_result.metrics, "fit_seconds": best_fit_seconds}])
    comparison.to_csv(output / "comparison.csv", index=False)
    slices = pd.concat({"Popularity": history_slices(baseline), "ItemKNN": history_slices(best_result)},
                       names=["model"])
    slices.to_csv(output / "history_slices.csv")
    item_slices = pd.concat({"Popularity": popularity_slices(baseline, context),
                            "ItemKNN": popularity_slices(best_result, context)}, names=["model"])
    item_slices.to_csv(output / "popularity_slices.csv")
    boot = protocol["confidence_intervals"]
    intervals = paired_bootstrap(best_result, baseline, resamples=boot["resamples"],
                                seed=boot["seed"], confidence=boot["confidence"],
                                result_name="ItemKNN", baseline_name="Popularity")
    intervals.to_csv(output / "bootstrap.csv", index=False)
    for name, result in (("best", best_result), ("popularity", baseline)):
        with deterministic_gzip(local / f"{name}_per_user.csv.gz") as stream:
            result.per_user.to_csv(stream, lineterminator="\n")
        # Indices refer to the shared lexicographic fit catalog, not another model's order.
        np.save(local / f"{name}_recommendation_indices.npy", result.recommendations)
    summary = {**plan, "best_params": best_params, "best_validation_metrics": best_result.metrics,
               "popularity_validation_metrics": baseline.metrics, "completed_trials": len(records),
               "bootstrap_intervals": intervals.to_dict(orient="records"),
               "interval_interpretation": "descriptive validation intervals after selection; no unbiased test claim",
               "local_artifacts": {str(path.relative_to(root)): sha256(path) for path in sorted(local.iterdir())
                                   if path.is_file()}}
    write_json(output / "summary.json", summary)
    return {"search": pd.DataFrame(records), "comparison": comparison,
            "history_slices": slices, "popularity_slices": item_slices,
            "intervals": intervals, "summary": summary, "best": best_result,
            "baseline": baseline, "context": context}


if __name__ == "__main__":
    result = run_validation_search()
    print(result["comparison"].to_string(index=False))
