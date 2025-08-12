import json
import re
import asyncio
from datetime import datetime
import os
import time

# Set the environment for Dynaconf to 'test' to load model configurations
os.environ["DYNACONF_ENV"] = "test"

from dotenv import load_dotenv
from SRAgent.agents.utils import set_model, load_settings

# ====== Config ======
INPUT_FILE = "/ssd2/xuyuan/output/output_sc_final.json"
OUTPUT_FILE = "/ssd2/xuyuan/output/metadata_extract.json"
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
def extract_json_from_response(text: str) -> str:
    """
    Extract JSON from model response text.
    - Remove <think>...</think> or 《...》 thinking chains
    - Prefer ```json { ... } ``` block
    - Otherwise match first {...} block
    - Fallback: If starts with "project_id", wrap in {}
    """
    if not isinstance(text, str):
        raise ValueError("model response is not a string")

    # remove possible thinking chains
    cleaned = re.sub(r"<think>.*?</think>|《.*?》", "", text, flags=re.DOTALL).strip()

    # prefer ```json ... ```
    m = re.search(r"```json\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    # match first {...}
    m = re.search(r"\{[\s\S]*\}", cleaned)
    if m:
        return m.group(0).strip()

    # fallback: "project_id": [...]
    m = re.search(r'"project_id"\s*:\s*\[.*', cleaned, flags=re.DOTALL)
    if m:
        candidate = m.group(0).strip()
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
    async with semaphore:
        meta_subset = {
            "shared_metadata": exp_data.get("shared_metadata", {}),
            "ai_targeted_metadata": exp_data.get("ai_targeted_metadata", {}),
            "sample_records": exp_data.get("individual_records", [])[:20]
        }
        meta_str = json.dumps(meta_subset, ensure_ascii=False)

        for attempt in range(1, RETRY_LIMIT + 1):
            try:
                start_time = time.time()

                # Interpolate meta_str into prompt_content
                prompt_content = PROMPT_TEMPLATE.replace("{metadata_json}", meta_str)

                resp = await model.ainvoke([{"role": "user", "content": prompt_content}])

                # resp might be an object (e.g., AIMessage)
                resp_text = getattr(resp, "content", None)
                if resp_text is None:
                    # If no content, try to convert resp to str
                    resp_text = str(resp)

                # Extract and parse JSON
                json_str = extract_json_from_response(resp_text)
                result = json.loads(json_str)

                response_time = time.time() - start_time
                return exp_id, result, response_time

            except Exception as e:
                # Reserve minimum logging for error location
                print(f"[Attempt {attempt}] {exp_id} error: {e}")
                if attempt < RETRY_LIMIT:
                    await asyncio.sleep(1)
                else:
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

    output = {
        "metadata_extraction_timestamp": datetime.now().isoformat(),
        "projects": {}
    }

    current_concurrency = INITIAL_CONCURRENCY
    semaphore = asyncio.Semaphore(current_concurrency)

    response_times = []
    batch_count = 0

    i = 0
    total = len(items)
    while i < total:
        batch_size = min(current_concurrency, total - i)
        batch = items[i:i + batch_size]

        # Create tasks for current batch (using current semaphore)
        results = await process_batch(semaphore, batch)

        # Store results and record response time
        for exp_id, meta, response_time in results:
            output["projects"][exp_id] = meta
            if response_time is not None:
                response_times.append(response_time)

        i += batch_size
        batch_count += 1

        # Every 3 batches, adjust concurrency based on average response time
        if batch_count % 3 == 0 and response_times:
            avg_rt = sum(response_times) / len(response_times)
            if avg_rt > 30 and current_concurrency > MIN_CONCURRENCY:
                current_concurrency = max(MIN_CONCURRENCY, current_concurrency - 1)
                print(f"Decreasing concurrency to {current_concurrency} (avg rt {avg_rt:.1f}s)")
            elif avg_rt < 10 and current_concurrency < MAX_CONCURRENCY:
                current_concurrency = min(MAX_CONCURRENCY, current_concurrency + 1)
                print(f"Increasing concurrency to {current_concurrency} (avg rt {avg_rt:.1f}s)")

            # Update semaphore
            semaphore = asyncio.Semaphore(current_concurrency)
            response_times = []
            await asyncio.sleep(0.3)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ Metadata extraction completed for {len(output['projects'])} projects.")

if __name__ == "__main__":
    asyncio.run(main())
