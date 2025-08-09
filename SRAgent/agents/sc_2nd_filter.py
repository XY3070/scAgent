import json
import os
import asyncio
from pathlib import Path
from SRAgent.agents.utils import set_model, load_settings
from dotenv import load_dotenv

load_dotenv()
os.environ["DYNACONF_ENV"] = "claude"

# ===== Config =====
INPUT_FILE = "/ssd2/xuyuan/output/ai_enhanced_data_20250808_121341.json"
OUTPUT_FILE = "/ssd2/xuyuan/output/output_single_cell.json"
DISCARDED_LIST_FILE = "/ssd2/xuyuan/output/discarded_experiments.txt"
BATCH_SIZE = 20
RETRY_LIMIT = 2

# ===== Init Model =====
settings = load_settings()
model = set_model(
    model_name="Qwen3-235B-A22B",
    agent_name="default",
    reasoning_effort="high",  # Thinking mode on
    temperature=0,
    settings=settings
)

# ===== Prompt Template =====
PROMPT_TEMPLATE = """
You are a bioinformatics expert, please based on the given metadata to determine whether the experiment is single-cell sequencing data (such as scRNA-seq, snRNA-seq, scATAC-seq, etc.).
If the experiment is single-cell sequencing data, please return the confidence level of the determination (0~3), where 0 means not single-cell, 1 means low confidence, 2 means medium confidence, and 3 means high confidence.
Requirements:
1. Strictly based on the provided data to determine, do not make up information.
2. The return format must be a valid JSON object, without any extra text or explanations.
3. JSON format:
{{
  "is_single_cell": true/false,
  "confidence": 0~3,
  "reason": "Brief reason"
}}

Metadata:
{metadata}
"""


async def classify_experiment(exp_id, exp_data):
    """
    Use Qwen to determine whether the experiment is single-cell sequencing data.
    Args:
        exp_id (str): experiment ID
        exp_data (dict): experiment data
    Returns:
        tuple: (exp_id, classification result)
    """
    meta_subset = {
        "shared_metadata": exp_data.get("shared_metadata", {}),
        "sequencing_method": exp_data.get("ai_targeted_metadata", {}).get("sequencing_method", {}).get("raw_data", {}),
        "records_sample": [r.get("characteristics_ch1") for r in exp_data.get("individual_records", [])[:3]]
    }
    meta_str = json.dumps(meta_subset, ensure_ascii=False)

    for attempt in range(RETRY_LIMIT):
        prompt = PROMPT_TEMPLATE.format(metadata=meta_str)
        try:
            resp = await model.ainvoke([{"role": "user", "content": prompt}])
            result = json.loads(resp.content)  # Parse JSON content compulsorily 
            if all(k in result for k in ("is_single_cell", "confidence", "reason")):
                return exp_id, result
        except Exception as e:
            print(f"[Retry {attempt+1}] {exp_id} parse error: {e}")
            await asyncio.sleep(1)  # Wait and retry
    return exp_id, {"is_single_cell": False, "confidence": 0, "reason": "Model did not return valid JSON"}


async def process_batch(batch):
    """
    Process a batch of experiments asynchronously.
    Args:
        batch (list): list of (exp_id, exp_data) tuples
    Returns:
        list: list of (exp_id, classification result) tuples
    """
    tasks = [classify_experiment(exp_id, exp_data) for exp_id, exp_data in batch]
    return await asyncio.gather(*tasks)


async def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    experiments = data["ai_data"]["hierarchical_data"]["GSE"]["experiments"]
    items = list(experiments.items())

    filtered_experiments = {}
    discarded = []

    # Concurrent processing in batches
    for i in range(0, len(items), BATCH_SIZE):
        batch = items[i:i+BATCH_SIZE]
        results = await process_batch(batch)
        for exp_id, res in results:
            if res["is_single_cell"]:
                exp_data = experiments[exp_id]
                exp_data["is_single_cell"] = res["is_single_cell"]
                exp_data["sc_ai_confidence"] = res["confidence"]
                exp_data["sc_ai_reason"] = res["reason"]
                filtered_experiments[exp_id] = exp_data
            else:
                title = experiments[exp_id].get("shared_metadata", {}).get("gse_title", "No Title")
                discarded.append(f"{exp_id}\t{title}")

    # Write output files
    # Write filtered experiments
    data["ai_data"]["hierarchical_data"]["GSE"]["experiments"] = filtered_experiments
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Write discarded list
    Path(DISCARDED_LIST_FILE).write_text("\n".join(discarded), encoding="utf-8")

    print(f"✅ 2nd filter completed: Reserve {len(filtered_experiments)} experiments, Discard {len(discarded)} experiments.")


if __name__ == "__main__":
    asyncio.run(main())
