#!/usr/bin/env python3
"""
build_results_master.py — 建立论文唯一数据源 results_master.csv / .json
说明：AIIA 投稿修改说明 §1.1 要求所有正文/表/图/补充材料只从一个数据源导出。
字段：dataset, seed, method, pruning_ratio, fine_tuning, quantization,
      accuracy, std, model_size_mb, latency_ms, peak_memory_mb
数据出处（只读不改）：
  - results/m4/m4_seed*.json        PlantVillage 3-seed 剪枝/FT
  - results/m4_pd/m4_pd_seed*.json  PlantDoc 3-seed 剪枝/FT
  - results/m5/m5_curves.json       seed42 INT8/组合/FT曲线/McNemar
  - autodl_results/imagenet_baseline.json  ImageNet-1K
  - results/table_edge（revised_paper.tex tab:edge） Pi 3B 边缘数据
输出：results/results_master.csv + results_master.json（新文件，不覆盖旧数据）
"""
import json, csv
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
RES = BASE / "results"

# ── 1. PlantVillage / PlantDoc：3-seed 剪枝+FT ──────────────────────────
def agg_m4(subdir):
    out = {}
    for p in sorted((RES / subdir).glob("m4*_seed*.json")):
        d = json.loads(p.read_text())
        seed = d.get("seed")
        for k, v in d.get("results", {}).items():
            out.setdefault(k, []).append((seed, v))
    return out

METHODS = {
    "baseline":        ("baseline", None, False, False),
    "prune25_noft":    ("prune_25pct", 0.25, False, False),
    "prune50_noft":    ("prune_50pct", 0.50, False, False),
    "prune75_noft":    ("prune_75pct", 0.75, False, False),
    "prune25_ft":      ("prune_25pct_ft5ep", 0.25, True, False),
    "prune50_ft":      ("prune_50pct_ft5ep", 0.50, True, False),
    "prune75_ft":      ("prune_75pct_ft5ep", 0.75, True, False),
}

rows = []

def add_row(dataset, seed, method, pruning_ratio, ft, quant,
            acc, std=None, size=None, latency=None, peak_memory_mb=None):
    rows.append({
        "dataset": dataset, "seed": seed, "method": method,
        "pruning_ratio": pruning_ratio, "fine_tuning": ft, "quantization": quant,
        "accuracy": acc, "std": std, "model_size_mb": size,
        "latency_ms": latency, "peak_memory_mb": peak_memory_mb,
    })

for subdir, dataset, size in (("m4", "PlantVillage", 327.4), ("m4_pd", "PlantDoc", 327.4)):
    agg = agg_m4(subdir)
    for k, (method, ratio, ft, _) in METHODS.items():
        if k not in agg:
            continue
        vals = [v for _, v in agg[k]]
        seeds = [s for s, _ in agg[k]]
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        for seed, v in zip(seeds, vals):
            add_row(dataset, seed, method, ratio, ft, False, v,
                    std=round(s, 3), size=size)

# ── 1b. headkeep 对照（保留训练好分类头 + 直接剪枝，无 FT）───────
HK_METHODS = {
    "prune25_headkeep_noft": ("prune_25pct_headkeep_noft", 0.25),
    "prune50_headkeep_noft": ("prune_50pct_headkeep_noft", 0.50),
    "prune75_headkeep_noft": ("prune_75pct_headkeep_noft", 0.75),
}
HK_DIRS = (("m4_headkeep", "PlantVillage"), ("m4_headkeep_pd", "PlantDoc"))
for subdir, dataset in HK_DIRS:
    agg = agg_m4(subdir)
    for k, (method, ratio) in HK_METHODS.items():
        if k not in agg:
            continue
        vals = [v for _, v in agg[k]]
        seeds = [s for s, _ in agg[k]]
        s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        for seed, v in zip(seeds, vals):
            add_row(dataset, seed, method, ratio, "no_ft", False, v,
                    std=round(s, 3), size=327.4)

# ── 2. INT8（seed42）：m5_curves.json + ImageNet ────────────────────────
m5 = json.loads((RES / "m5" / "m5_curves.json").read_text())
for cfg, info in m5.get("int8", {}).items():
    label = {"baseline": "baseline", "prune25": "prune_25pct_ft5ep",
             "prune50": "prune_50pct_ft5ep", "prune75": "prune_75pct_ft5ep",
             "prune25_noft": "prune_25pct", "prune50_noft": "prune_50pct",
             "prune75_noft": "prune_75pct"}[cfg]
    ratio = {"baseline": None, "prune25": 0.25, "prune50": 0.50, "prune75": 0.75,
             "prune25_noft": 0.25, "prune50_noft": 0.50, "prune75_noft": 0.75}[cfg]
    ft = "ft5ep" in label or "ft" in cfg
    add_row("PlantVillage", 42, label + "_int8", ratio, ft, True,
            info.get("acc"), size=info.get("size_mb", 83.7))

# PlantDoc INT8（all_experiments.json 有）
ae = json.loads((RES / "all_experiments.json").read_text())
for r in ae:
    if r.get("dataset") == "PlantDoc" and "int8" in r.get("method", ""):
        m = r["method"]
        if m == "int8_quant":
            m = "baseline_int8"
        elif m.endswith("_int8"):
            m = m[:-5] + "_int8"  # prune_25pct_int8 -> prune_25pct_int8 保持
        add_row("PlantDoc", 42, m,
                r.get("pruning_ratio"), "ft5ep" in m, True,
                r["accuracy"], size=r.get("size_mb"))

# ── 3. ImageNet-1K ─────────────────────────────────────────────────────
im = json.loads((BASE / "autodl_results" / "imagenet_baseline.json").read_text())
imap = [
    ("baseline", None, False, "baseline_top1", 81.1),
    ("prune_25pct", 0.25, False, "prune_25pct_noft", 59.07),
    ("prune_50pct", 0.50, False, "prune_50pct_noft", 13.26),
    ("prune_75pct", 0.75, False, "prune_75pct_noft", 0.27),
    ("prune_25pct_ft5ep", 0.25, True, "prune_25pct_ft5ep", 83.49),
    ("prune_50pct_ft5ep", 0.50, True, "prune_50pct_ft5ep", 82.29),
    ("prune_75pct_ft5ep", 0.75, True, "prune_75pct_ft5ep", 81.40),
    ("int8_quant", None, False, "int8_quant_acc", 80.90),
]
for method, ratio, ft, key, fallback in imap:
    v = im.get(key, fallback)
    size = im.get("int8_quant_size_mb") if method == "int8_quant" else im.get("size_mb", 330.2)
    add_row("ImageNet-1K", 0, method, ratio, ft, method == "int8_quant", v, size=size)

# ── 4. 边缘部署（Pi 3B，revised_paper.tex tab:edge）────────────────────
edge = [
    # dataset, seed, method, ratio, ft, quant, acc, size, latency, peak
    ("Pi3B-50pics", 0, "baseline_fp32", None, False, False, None, 343, 5353, 495),
    ("Pi3B-50pics", 0, "prune_50pct_ft5ep_fp32", 0.50, True, False, None, 343, 5326, 513),
    ("Pi3B-50pics", 0, "baseline_int8", None, False, True, None, 83, 3528, 281),
    ("Pi3B-50pics", 0, "prune_50pct_ft5ep_int8", 0.50, True, True, None, 83, 3519, 278),
]
for ds, seed, method, ratio, ft, q, acc, size, lat, peak in edge:
    add_row(ds, seed, method, ratio, ft, q, acc, size=size, latency=lat, peak_memory_mb=peak)

# ── 写出 ──────────────────────────────────────────────────────────────
cols = ["dataset", "seed", "method", "pruning_ratio", "fine_tuning",
        "quantization", "accuracy", "std", "model_size_mb", "latency_ms", "peak_memory_mb"]
with open(RES / "results_master.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow(r)
with open(RES / "results_master.json", "w") as f:
    json.dump(rows, f, indent=2)

print(f"rows: {len(rows)}")
print("→ results/results_master.csv  (+ .json)")
