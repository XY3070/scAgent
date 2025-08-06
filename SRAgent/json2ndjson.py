#!/usr/bin/env python3
# json2ndjson.py

"""
Break down SRAgent's hierarchical JSON into:
- gse.ndjson  —— experiment / project level
- gsm.ndjson  —— sample / run level

Usage:
python json2ndjson.py hierarchical.json -o data/
# get data/gse.ndjson  and data/gsm.ndjson
"""

import json, jsonlines, pathlib, argparse

def main(infile: str, outdir: str):
    outdir = pathlib.Path(outdir)
    outdir.mkdir(exist_ok=True)

    with open(infile, "r") as f:
        root = json.load(f)

    gse_dict = root["hierarchical_data"]["GSE"]["experiments"]

    # 1. experiment-level
    with jsonlines.open(outdir / "gse.ndjson", "w") as writer:
        for pid, exp in gse_dict.items():
            obj = {"project_id": pid}
            obj.update(exp["shared_metadata"] or {})
            writer.write(obj)

    # 2. sample-level
    with jsonlines.open(outdir / "gsm.ndjson", "w") as writer:
        for pid, exp in gse_dict.items():
            for rec in exp["individual_records"]:
                rec = dict(rec)            # copy, don't modify the original
                rec["project_id"] = pid    # foreign key
                writer.write(rec)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="convert hierarchical JSON to NDJSON")
    p.add_argument("input",  help="input JSON (hierarchical)")
    p.add_argument("-o","--outdir", default=".", help="output directory")
    main(**vars(p.parse_args()))
    print("\nDone.")
    print(f"gse.ndjson: {outdir / 'gse.ndjson'}")
    print(f"gsm.ndjson: {outdir / 'gsm.ndjson'}")
