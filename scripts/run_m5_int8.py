#!/usr/bin/env python3
"""
M5: 补 prune+FT+INT8 组合数据（审稿意见 M5）
阶段A: PlantVillage 3 个 FT 剪枝模型 → INT8 动态量化 → 全量验证集评估 (CPU)
阶段B: PlantDoc 训练 3 个剪枝模型 (CPU) + INT8 评估
输出: results/m5_pv_int8.json, results/m5_pd_int8.json
"""
import torch, torch.nn as nn, json, time, warnings, copy
import timm
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
torch.set_num_threads(8)

BASE = Path("REPO_ROOT")
PV_DATA = Path("DATASETS/plantvillage")
PD_DATA = Path("DATASETS/plantdoc")

val_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_val_loader(path, batch=64):
    ds = datasets.ImageFolder(str(path), transform=val_tf)
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0), ds

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    cor = tot = 0
    for imgs, lbls in loader:
        _, pred = model(imgs).max(1)
        cor += pred.eq(lbls).sum().item()
        tot += lbls.size(0)
    return cor / tot

def quant_eval(path, num_classes, val_path, tag):
    m = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=False)
    m.eval()
    mq = torch.ao.quantization.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)
    loader, ds = get_val_loader(val_path)
    t0 = time.time()
    acc = evaluate(mq, loader)
    print(f"[{tag}] INT8 acc={acc*100:.2f}% (n={len(ds)}, {time.time()-t0:.0f}s)", flush=True)
    return round(acc * 100, 2)

# ═══ 阶段 A: PlantVillage ═══
print("=== 阶段A: PlantVillage prune+FT+INT8 ===", flush=True)
pv_out = {}
for ratio in [25, 50, 75]:
    pv_out[f"prune{ratio}"] = quant_eval(
        BASE / "results" / f"prune_{ratio}_best.pth", 38,
        PV_DATA / "valid", f"PV prune{ratio}")
(BASE / "results" / "m5_pv_int8.json").write_text(json.dumps(pv_out, indent=2))
print("PV 完成:", pv_out, flush=True)

# ═══ 阶段 B: PlantDoc（训练剪枝模型 + INT8）═══
print("=== 阶段B: PlantDoc prune+FT+INT8 ===", flush=True)
import torch.optim as optim

train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# PlantDoc: train 全量 + 内部 80/20 划分（与 run_cross_dataset.py 一致）
full_train = datasets.ImageFolder(str(PD_DATA / "train"), transform=train_tf)
n = len(full_train); n_val = int(n * 0.2)
from torch.utils.data import random_split
train_ds, val_ds = random_split(full_train, [n - n_val, n_val], generator=torch.Generator().manual_seed(42))
val_ds.dataset.transform = val_tf
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
num_cls = len(full_train.classes)
print(f"PlantDoc: {len(train_ds)} train / {len(val_ds)} val / {num_cls} classes", flush=True)

def prune_heads(model, ratio):
    pm = copy.deepcopy(model)
    for name, mod in pm.named_modules():
        if "attn" in name and hasattr(mod, "num_heads"):
            n_heads, hd = mod.num_heads, mod.head_dim
            qkv = mod.qkv.weight.data.view(3, n_heads, hd, -1)
            norms = qkv.norm(dim=2).norm(dim=2).mean(0)
            n_prune = int(n_heads * ratio)
            if n_prune == 0: continue
            for h in torch.argsort(norms)[:n_prune]:
                qkv[:, h] = 0
            mod.qkv.weight.data = qkv.reshape(mod.qkv.weight.shape)
            if hasattr(mod, "proj"):
                proj = mod.proj.weight.data.view(-1, n_heads, hd)
                for h in torch.argsort(norms)[:n_prune]:
                    proj[:, h] = 0
                mod.proj.weight.data = proj.reshape(mod.proj.weight.shape)
    return pm

def finetune(model, epochs=5, tag=""):
    opt = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(epochs):
        model.train()
        for imgs, lbls in train_loader:
            opt.zero_grad()
            loss = crit(model(imgs), lbls)
            loss.backward(); opt.step()
        sched.step()
        va = evaluate(model, val_loader)
        best = max(best, va)
        print(f"    {tag} ep{ep+1}: val={va*100:.2f}%", flush=True)
    return best

base = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_cls)
base.load_state_dict(torch.load(BASE / "results" / "finetuned_plantdoc.pth", map_location="cpu", weights_only=True))
print(f"PD baseline 复现: {evaluate(base, val_loader)*100:.2f}%", flush=True)

pd_out = {}
for ratio in [25, 50, 75]:
    pm = prune_heads(copy.deepcopy(base), ratio / 100)
    va = evaluate(pm, val_loader)
    print(f"  PD prune{ratio} noFT: {va*100:.2f}%", flush=True)
    best = finetune(pm, 5, f"PD prune{ratio}")
    print(f"  PD prune{ratio} FT5ep: {best*100:.2f}%", flush=True)
    # INT8 评估
    mq = torch.ao.quantization.quantize_dynamic(pm, {nn.Linear}, dtype=torch.qint8)
    qa = evaluate(mq, val_loader)
    print(f"  PD prune{ratio} FT+INT8: {qa*100:.2f}%", flush=True)
    pd_out[f"prune{ratio}"] = {"noft": round(va*100, 2), "ft": round(best*100, 2), "ft_int8": round(qa*100, 2)}
(BASE / "results" / "m5_pd_int8.json").write_text(json.dumps(pd_out, indent=2))
print("PD 完成:", pd_out, flush=True)
print("ALL M5 DONE", flush=True)
