#!/usr/bin/env python3
"""
run_m4_headkeep.py — 说明文档 §1.3 要求的对照实验
"保留训练完成的分类头、直接剪枝"：baseline FT 完成后保存权重，
再在 FT 好的模型上 prune 25/50/75% head，立即 eval（分类头是训练好的）。
现有 m4 的 no-FT（1.29~5.35%）是在"随机初始化分类头+剪枝"上测的，
按说明文档只能称为 diagnostic control；本脚本补齐主证据。

输出: results/m4_headkeep/m4_headkeep_seed{seed}.json
     results/m4_headkeep.json  (3-seed mean±std 汇总)
"""
import json, time, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from safetensors.torch import load_file
import timm

BASE = Path("REPO_ROOT")
RES = BASE / "results"
OUT_DIR = RES / "m4_headkeep"
OUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 123, 2026]
BATCH = 32
EPOCHS = 5
LR = 5e-5
WD = 0.01

PT_WEIGHTS = Path("PRETRAINED_VIT")
DATASET_PATH = Path("DATASETS/plantvillage")

train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def make_model():
    m = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=38)
    sd = load_file(str(PT_WEIGHTS))
    sd = {k: v for k, v in sd.items() if not k.startswith("head.")}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if not k.startswith("head.")]
    assert not real_missing, f"missing: {real_missing[:5]}"
    return m

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    cor = tot = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        _, pred = model(imgs).max(1)
        cor += pred.eq(lbls).sum().item()
        tot += lbls.size(0)
    return cor / tot

def finetune(model, train_loader, val_loader, seed, tag, out):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")
    best = 0.0
    for ep in range(EPOCHS):
        model.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(model(imgs), lbls)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        va = evaluate(model, val_loader)
        best = max(best, va)
        print(f"    [{tag}] seed={seed} ep{ep+1}: val={va*100:.2f}%", flush=True)
        out["progress"][tag] = {"epoch": ep + 1, "val": round(va * 100, 2)}
        (OUT_DIR / f"m4_headkeep_seed{seed}.json").write_text(json.dumps(out, indent=2))
    return best

def prune_heads(model, ratio):
    pm = copy.deepcopy(model)
    for name, mod in pm.named_modules():
        if "attn" in name and hasattr(mod, "num_heads"):
            n_heads, hd = mod.num_heads, mod.head_dim
            qkv = mod.qkv.weight.data.view(3, n_heads, hd, -1)
            norms = qkv.norm(dim=2).norm(dim=2).mean(0)
            n_prune = int(n_heads * ratio)
            if n_prune == 0:
                continue
            for h in torch.argsort(norms)[:n_prune]:
                qkv[:, h] = 0
            mod.qkv.weight.data = qkv.reshape(mod.qkv.weight.shape)
            if hasattr(mod, "proj"):
                proj = mod.proj.weight.data.view(-1, n_heads, hd)
                for h in torch.argsort(norms)[:n_prune]:
                    proj[:, h] = 0
                mod.proj.weight.data = proj.reshape(mod.proj.weight.shape)
    return pm

def run_seed(seed):
    out_path = OUT_DIR / f"m4_headkeep_seed{seed}.json"
    if out_path.exists():
        out = json.loads(out_path.read_text())
        if out.get("done"):
            print(f"seed {seed} 已完成，跳过", flush=True)
            return out
    else:
        out = {"seed": seed, "progress": {}, "results": {}, "done": False}

    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed); np.random.seed(seed)
    train_ds = datasets.ImageFolder(str(DATASET_PATH / "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(str(DATASET_PATH / "valid"), transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    print(f"\n=== seed {seed} (headkeep) ===", flush=True)

    # 1) baseline FT（与 m4 相同协议），保存权重
    if "baseline" not in out["results"]:
        print(f"  [baseline FT5ep]", flush=True)
        m = make_model().to(DEVICE)
        best = finetune(m, train_loader, val_loader, seed, "baseline", out)
        out["results"]["baseline"] = round(best * 100, 2)
        out["progress"]["baseline"] = {"val": round(best * 100, 2), "done": True}
        torch.save(m.state_dict(), OUT_DIR / f"m4_headkeep_seed{seed}_baseline_ft.pth")
        (OUT_DIR / f"m4_headkeep_seed{seed}.json").write_text(json.dumps(out, indent=2))
    else:
        m = make_model().to(DEVICE)
        m.load_state_dict(torch.load(OUT_DIR / f"m4_headkeep_seed{seed}_baseline_ft.pth", map_location=DEVICE, weights_only=True))

    # 2) 在 FT 好的模型上 prune → 立即 eval（保留训练好的分类头）
    for ratio in [0.25, 0.50, 0.75]:
        tag = f"prune{int(ratio*100)}_headkeep_noft"
        if tag in out["results"]:
            continue
        print(f"  [{tag}]", flush=True)
        pm = prune_heads(m, ratio)
        va = evaluate(pm, val_loader)
        out["results"][tag] = round(va * 100, 2)
        print(f"    acc={va*100:.2f}%", flush=True)
        (OUT_DIR / f"m4_headkeep_seed{seed}.json").write_text(json.dumps(out, indent=2))

    out["done"] = True
    (OUT_DIR / f"m4_headkeep_seed{seed}.json").write_text(json.dumps(out, indent=2))
    print(f"seed {seed} 完成: {out['results']}", flush=True)
    return out

if __name__ == "__main__":
    t0 = time.time()
    all_out = {}
    for s in SEEDS:
        all_out[s] = run_seed(s)
    # 汇总 3-seed mean±std
    summary = {"baseline": {"vals": [], "mean": None, "std": None}}
    keys = ["prune25_headkeep_noft", "prune50_headkeep_noft", "prune75_headkeep_noft"]
    for k in keys:
        summary[k] = {"vals": []}
    for s, out in all_out.items():
        summary["baseline"]["vals"].append(out["results"]["baseline"])
        for k in keys:
            summary[k]["vals"].append(out["results"].get(k))
    for k, v in summary.items():
        a = np.array(v["vals"], dtype=float)
        v["mean"] = round(float(a.mean()), 2)
        v["std"] = round(float(a.std(ddof=1)) if len(a) > 1 else 0.0, 2)
        v["n"] = len(a)
    (RES / "m4_headkeep.json").write_text(json.dumps(summary, indent=2))
    print("\n=== 汇总 (保留训练好分类头 + 直接剪枝) ===")
    for k, v in summary.items():
        print(f"  {k:28s} {v['mean']:.2f} ± {v['std']:.2f} (n={v['n']})")
    print(f"\nALL DONE in {(time.time()-t0)/3600:.2f}h")
