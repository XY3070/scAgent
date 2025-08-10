import json
import re
import asyncio
from datetime import datetime
from pathlib import Path
import os

# Set the environment for Dynaconf to 'test' to load model configurations
# This must be done before importing load_settings
os.environ["DYNACONF_ENV"] = "test"

from dotenv import load_dotenv
from SRAgent.agents.utils import set_model, load_settings


# ====== Config ======
INPUT_FILE = "/ssd2/xuyuan/output/output_single_cell.json"  # the output of the second filter
OUTPUT_FILE = "/ssd2/xuyuan/output/metadata_try.json"
BATCH_SIZE = 1
RETRY_LIMIT = 2


# ====== Init Model ======
load_dotenv()
settings = load_settings()
model = set_model(
    model_name=settings["models"]["metadata"],
    agent_name="metadata",
    reasoning_effort=settings["reasoning_effort"]["metadata"],
    temperature=settings["temperature"]["metadata"],
    settings=settings
)


# ====== Helper Functions ======
def extract_json_from_response(text):
    """
    Extract JSON object from model response:
    1. Remove <think>...</think> reasoning chain.
    2. Prefer the ```json ...``` code block if present.
    3. Otherwise, match the first {...} block.
    4. If the outer {} is missing but content starts with "project_id",
       auto-wrap it with {}.
    """
    print("===== [DEBUG] Raw model output =====")
    print(repr(text))  # Use repr to avoid next line/space info loss
    print("====================================")

    # Remove reasoning chain
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # print("===== [DEBUG] After removing <think>...</think> =====")
    # print(repr(cleaned))
    # print("====================================")

    # Prefer: ```json ... ```
    match = re.search(r"```json\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if match:
        json_block = match.group(1).strip()
        # print("===== [DEBUG] Matched ```json``` block =====")
        # print(repr(json_block))
        # print("============================================")
        return json_block

    # Match standard JSON object {...}
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        json_block = match.group(0).strip()
        # print("===== [DEBUG] Matched {...} block =====")
        # print(repr(json_block))
        # print("============================================")
        return json_block

    # Last resort: missing outer {} but starting with "project_id"
    match = re.search(r'"project_id"\s*:\s*\[.*', cleaned, flags=re.DOTALL)
    if match:
        candidate = match.group(0).strip()
        wrapped = "{\n" + candidate.rstrip(",") + "\n}"
        # print("===== [DEBUG] Wrapped 'project_id' block =====")
        # print(repr(wrapped))
        # print("==============================================")
        return wrapped

    raise ValueError("No JSON object found in model output")


PROMPT_TEMPLATE = """
You are a bioinformatics expert. Extract the following metadata from the given experiment data.

Strict rules:
- Base your answer strictly on the provided data. Do NOT make up any values.
- If a field has no data in the content, output an empty list or empty string.
- If a field has multiple values, include all values.
- For fields "age" and "sex", also provide per-run mapping between run ID (SRA/GSM accession) and the value.
- Output strictly in JSON format, with the following keys:

{{
  "project_id": [list of IDs],
  "tissue": [list],
  "cell_type": [list],
  "technology": [list],
  "platform": [list],
  "country": [list],
  "sample_size": int or "",
  "age": {{
    "per_run": {{ "RUN_ID": "value", ... }},
    "unique_values": [list]
  }},
  "sex": {{
    "per_run": {{ "RUN_ID": "value", ... }},
    "unique_values": [list]
  }},
  "ancestry": [list],
  "health_status": [list],
  "tumor": true/false or "",
  "project_description": "string",
  "download_link": [list],
  "citation": [list]
}}

Here is the experiment data:
{metadata_json}

Return only one complete JSON object as the output, starting with '{{' and ending with '}}'.
"""


# ====== Core Extraction ======
async def extract_metadata(exp_id, exp_data):
    """
    Call model to extract metadata
    """
    meta_subset = {
        "shared_metadata": exp_data.get("shared_metadata", {}),
        "ai_targeted_metadata": exp_data.get("ai_targeted_metadata", {}),
        "sample_records": exp_data.get("individual_records", [])[:20]  # avoid token limit
    }
    meta_str = json.dumps(meta_subset, ensure_ascii=False)

    for attempt in range(RETRY_LIMIT):
        try:
            resp = await model.ainvoke([{"role": "user", "content": PROMPT_TEMPLATE.format(metadata_json=meta_str)}])

            # # DEBUG: print the full thinking chain (commentable)
            # print(f"[DEBUG] Raw model response for {exp_id}:\n{resp.content}\n{'-'*60}")

            # Extract JSON
            json_str = extract_json_from_response(resp.content)

            # Parse JSON
            result = json.loads(json_str)

            return exp_id, result

        except Exception as e:
            print(f"[Retry {attempt+1}] {exp_id} parse error: {e}")
            await asyncio.sleep(1)

    return exp_id, {}


async def process_batch(batch):
    tasks = [extract_metadata(exp_id, exp_data) for exp_id, exp_data in batch]
    return await asyncio.gather(*tasks)


# ====== Main Entry ======
async def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    experiments = data["ai_data"]["hierarchical_data"]["GSE"]["experiments"]
    items = list(experiments.items())

    output = {
        "metadata_extraction_timestamp": datetime.now().isoformat(),
        "projects": {}
    }

    for i in range(0, len(items), BATCH_SIZE):
        batch = items[i:i+BATCH_SIZE]
        results = await process_batch(batch)
        for exp_id, meta in results:
            output["projects"][exp_id] = meta

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ Metadata extraction completed for {len(output['projects'])} projects.")


if __name__ == "__main__":
    asyncio.run(main())
