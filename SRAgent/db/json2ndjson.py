#!/usr/bin/env python3
import json

def json_to_ndjson(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    experiments = data.get("ai_data", {}) \
                      .get("hierarchical_data", {}) \
                      .get("GSE", {}) \
                      .get("experiments", {})

    with open(output_file, "w", encoding="utf-8") as out:
        for experiment_id, exp_data in experiments.items():
            # Experiment-level metadata (remove run-level metadata)
            experiment_metadata = {k: v for k, v in exp_data.items() if k != "individual_records"}

            # Run-level
            records = exp_data.get("individual_records", {})

            if not records:  
                # No run-level data, still output a row
                row = {
                    "experiment_id": experiment_id,
                    "experiment_metadata": experiment_metadata,
                    "run_id": None,
                    "run_metadata": None
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            # support dict and list, two data structures 
            if isinstance(records, dict):
                iterable = records.items()
            elif isinstance(records, list):
                iterable = enumerate(records)  
            else:
                raise TypeError(f"Unexpected type for individual_records: {type(records)}. Expected dict or list.")  

            for run_key, run_metadata in iterable:
                # if run_metadata already contains run_id, then use it  
                run_id = run_metadata.get("run_id", run_key) if isinstance(run_metadata, dict) else run_key
                row = {
                    "experiment_id": experiment_id,
                    "experiment_metadata": experiment_metadata,
                    "run_id": run_id,
                    "run_metadata": run_metadata
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert hierarchical experiment/run JSON to NDJSON for AI agent processing.")
    parser.add_argument("input", help="Input JSON file")
    parser.add_argument("output", help="Output NDJSON file")
    args = parser.parse_args()

    # script usage: uv run json2ndjson.py input.json output.ndjson
    json_to_ndjson(args.input, args.output)
