import json
import re
import asyncio
from datetime import datetime
from pathlib import Path
import os
import time

# Set the environment for Dynaconf to 'test' to load model configurations
# This must be done before importing load_settings
os.environ["DYNACONF_ENV"] = "test"

from dotenv import load_dotenv
from SRAgent.agents.utils import set_model, load_settings


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
    1. Remove 《...》 reasoning chain.
    2. Prefer the ```json ...``` code block if present.
    3. Otherwise, match the first {...} block.
    4. If the outer {} is missing but content starts with "project_id",
       auto-wrap it with {}.
    """
    # Remove reasoning chain
    cleaned = re.sub(r"《.*?》", "", text, flags=re.DOTALL).strip()

    # Prefer: ```json ... ```
    match = re.search(r"```json\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    # Match standard JSON object {...}
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        return match.group(0).strip()

    # Last resort: missing outer {} but starting with "project_id"
    match = re.search(r'"project_id"\s*:\s*\[.*', cleaned, flags=re.DOTALL)
    if match:
        candidate = match.group(0).strip()
        wrapped = "{\n" + candidate.rstrip(",") + "\n}"
        return wrapped

    raise ValueError("No JSON object found in model output")


# ====== Test Function ======
async def test_model_and_parse():
    # 读取输入文件
    with open("/ssd2/xuyuan/output/output_single_cell.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    experiments = data["ai_data"]["hierarchical_data"]["GSE"]["experiments"]
    items = list(experiments.items())

    # 只处理第一个项目进行调试
    exp_id, exp_data = items[0]

    meta_subset = {
        "shared_metadata": exp_data.get("shared_metadata", {}),
        "ai_targeted_metadata": exp_data.get("ai_targeted_metadata", {}),
        "sample_records": exp_data.get("individual_records", [])[:20]  # avoid token limit
    }
    meta_str = json.dumps(meta_subset, ensure_ascii=False)

    # 修复模板中的大括号转义问题
    PROMPT_TEMPLATE = """
You are a bioinformatics expert. Extract the following metadata from the given experiment data.

Strict rules:
- Base your answer strictly on the provided data. Do NOT make up any values.
- If a field has no data in the content, output an empty list or empty string.
- If a field has multiple values, include all values.
- For fields "age" and "sex", also provide per-run mapping between run ID (SRA/GSM accession) and the value.
- Output strictly in JSON format, with the following keys:

{
  "project_id": [list of IDs],
  "tissue": [list],
  "cell_type": [list],
  "technology": [list],
  "platform": [list],
  "country": [list],
  "sample_size": int or "",
  "age": {
    "per_run": { "RUN_ID": "value", ... },
    "unique_values": [list]
  },
  "sex": {
    "per_run": { "RUN_ID": "value", ... },
    "unique_values": [list]
  },
  "ancestry": [list],
  "health_status": [list],
  "tumor": true/false or "",
  "project_description": "string",
  "download_link": [list],
  "citation": [list]
}

Here is the experiment data:
{metadata_json}

Return only one complete JSON object as the output, starting with '{{' and ending with '}}'.
"""

    try:
        print(f"Calling model for {exp_id}...")
        resp = await model.ainvoke([{"role": "user", "content": PROMPT_TEMPLATE.format(metadata_json=meta_str)}])
        
        print(f"Model response for {exp_id}:")
        print(resp.content)
        print("-" * 60)

        # Extract JSON
        json_str = extract_json_from_response(resp.content)
        
        print(f"Extracted JSON string for {exp_id}:")
        print(repr(json_str))
        print("-" * 60)

        # Parse JSON
        result = json.loads(json_str)
        print(f"Parsed JSON for {exp_id}:")
        print(result)
        
    except Exception as e:
        print(f"[ERROR] {exp_id} error: {e}")
        import traceback
        traceback.print_exc()


# ====== Main Entry ======
async def main():
    await test_model_and_parse()


if __name__ == "__main__":
    asyncio.run(main())