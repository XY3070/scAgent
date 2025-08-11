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


# ====== Config ======
INPUT_FILE = "/ssd2/xuyuan/output/output_single_cell.json"  # the output of the second filter
OUTPUT_FILE = "/ssd2/xuyuan/output/metadata_try.json"
INITIAL_CONCURRENCY = 5
MAX_CONCURRENCY = 10
MIN_CONCURRENCY = 1
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


# ====== Core Extraction ======
async def extract_metadata(semaphore, exp_id, exp_data):
    """
    Call model to extract metadata
    """
    async with semaphore:  # 控制并发数量
        meta_subset = {
            "shared_metadata": exp_data.get("shared_metadata", {}),
            "ai_targeted_metadata": exp_data.get("ai_targeted_metadata", {}),
            "sample_records": exp_data.get("individual_records", [])[:20]  # avoid token limit
        }
        meta_str = json.dumps(meta_subset, ensure_ascii=False)

        for attempt in range(RETRY_LIMIT):
            try:
                start_time = time.time()
                resp = await model.ainvoke([{"role": "user", "content": PROMPT_TEMPLATE.format(metadata_json=meta_str)}])
                end_time = time.time()
                
                # Extract JSON
                json_str = extract_json_from_response(resp.content)
                
                # 在尝试解析之前打印要解析的JSON字符串
                print(f"[DEBUG] JSON string to parse for {exp_id} (attempt {attempt+1}):")
                print(repr(json_str))
                print("-" * 60)

                # Parse JSON
                result = json.loads(json_str)
                
                # 计算响应时间
                response_time = end_time - start_time
                return exp_id, result, response_time

            except json.JSONDecodeError as e:
                print(f"[ERROR] {exp_id} JSON decode error: {e}")
                print(f"[DEBUG] Problematic JSON string for {exp_id}:")
                print(repr(json_str))
                print("-" * 60)
                if attempt == RETRY_LIMIT - 1:  # 最后一次尝试
                    # 返回一个空的元数据对象而不是抛出异常
                    return exp_id, {}, None
                await asyncio.sleep(1)
            except Exception as e:
                print(f"[ERROR] {exp_id} parse error: {e}")
                print(f"[DEBUG] Problematic response for {exp_id}:")
                print(repr(resp.content if 'resp' in locals() else 'No response'))
                print("-" * 60)
                if attempt == RETRY_LIMIT - 1:  # 最后一次尝试
                    # 返回一个空的元数据对象而不是抛出异常
                    return exp_id, {}, None
                await asyncio.sleep(1)

        return exp_id, {}, None


async def process_batch(semaphore, batch):
    tasks = [extract_metadata(semaphore, exp_id, exp_data) for exp_id, exp_data in batch]
    return await asyncio.gather(*tasks)


# ====== Main Entry ======
async def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    experiments = data["ai_data"]["hierarchical_data"]["GSE"]["experiments"]
    items = list(experiments.items())

    # 只处理前3个项目进行调试
    items = items[:3]

    output = {
        "metadata_extraction_timestamp": datetime.now().isoformat(),
        "projects": {}
    }

    # 初始化并发控制
    current_concurrency = 1  # 只使用1个并发以简化调试
    semaphore = asyncio.Semaphore(current_concurrency)
    
    # 用于跟踪响应时间的变量
    response_times = []
    batch_count = 0

    for i in range(0, len(items), current_concurrency):
        batch = items[i:i+current_concurrency]
        results = await process_batch(semaphore, batch)
        
        # 处理结果并收集响应时间
        for exp_id, meta, response_time in results:
            if response_time is not None:
                response_times.append(response_time)
            output["projects"][exp_id] = meta
        
        batch_count += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ Metadata extraction completed for {len(output['projects'])} projects.")


if __name__ == "__main__":
    asyncio.run(main())