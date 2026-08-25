#!/usr/bin/env python3
"""R2 Q8: 轻量 CNN 基线对照（MobileNetV3-Large / Small）
数据集: PlantDoc (M4 协议: seed-42 80/20 划分) + PlantVillage (Kaggle train/valid 划分)
训练: 5ep, AdamW lr=1e-3 (CNN 常规 FT 学习率), wd=1e-4, batch 64, cosine, AMP
输出: results/cnn_baseline.json {dataset: {model: {acc, params_m, size_mb}}}
"""
import json, time, torch, torch.nn as nn, warnings
import timm
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = True

BASE = Path("REPO_ROOT")
PD_PATH = Path("DATASETS/plantdoc")
PV_PATH = Path("DATASETS/plantvillage")
OUT_PATH = BASE / "results" / "cnn_baseline.json"
DEVICE = torch.device("cuda")
EPOCHS, LR, WD, BATCH = 5, 1e-3, 1e-4, 64

train_tf = transforms.Compose([
    transforms.Resize((256, 256)), transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((256, 256)), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

MODELS = ["mobilenetv3_large_100", "mobilenetv3_small_100"]


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


def finetune(model, train_loader, val_loader, tag):
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
        print(f"    {tag} ep{ep+1}: val={va*100:.2f}%", flush=True)
    return best


def run_one(model_name, train_loader, val_loader, num_classes, tag):
    m = timm.create_model(model_name, pretrained=True, num_classes=num_classes).to(DEVICE)
    best = finetune(m, train_loader, val_loader, tag)
    params_m = sum(p.numel() for p in m.parameters()) / 1e6
    size_mb = sum(p.numel() * p.element_size() for p in m.parameters()) / 1024 / 1024
    # CPU 延迟（torch, batch 1）
    m.cpu().eval()
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        for _ in range(5):
            m(dummy)
        t0 = time.time()
        for _ in range(20):
            m(dummy)
        lat_ms = (time.time() - t0) / 20 * 1000
    print(f"{tag}: acc={best*100:.2f}% params={params_m:.2f}M size={size_mb:.1f}MB lat={lat_ms:.0f}ms", flush=True)
    return {"acc": round(best * 100, 2), "params_m": round(params_m, 2),
            "size_mb": round(size_mb, 1), "cpu_lat_ms": round(lat_ms, 1)}


def main():
    out = {}
    # PlantDoc (M4 协议)
    torch.manual_seed(42); np.random.seed(42)
    full_train = datasets.ImageFolder(str(PD_PATH / "train"), transform=train_tf)
    n_cls = len(full_train.classes)
    n_val = int(len(full_train) * 0.2)
    train_ds, val_ds = torch.utils.data.random_split(
        full_train, [len(full_train) - n_val, n_val],
        generator=torch.Generator().manual_seed(42))
    val_ds.dataset.transform = val_tf
    g = torch.Generator().manual_seed(42)
    tl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True, generator=g)
    vl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    print(f"PlantDoc: {len(train_ds)}/{len(val_ds)} {n_cls}cls", flush=True)
    out["PlantDoc"] = {}
    for mn in MODELS:
        out["PlantDoc"][mn] = run_one(mn, tl, vl, n_cls, f"PD-{mn}")
        json.dump(out, open(OUT_PATH, "w"), indent=2)

    # PlantVillage (Kaggle 划分)
    torch.manual_seed(42); np.random.seed(42)
    train_ds = datasets.ImageFolder(str(PV_PATH / "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(str(PV_PATH / "valid"), transform=val_tf)
    n_cls = len(train_ds.classes)
    g = torch.Generator().manual_seed(42)
    tl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True, generator=g)
    vl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    print(f"PlantVillage: {len(train_ds)}/{len(val_ds)} {n_cls}cls", flush=True)
    out["PlantVillage"] = {}
    for mn in MODELS:
        out["PlantVillage"][mn] = run_one(mn, tl, vl, n_cls, f"PV-{mn}")
        json.dump(out, open(OUT_PATH, "w"), indent=2)

    print("DONE ->", OUT_PATH, flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"{(time.time()-t0)/60:.1f} min", flush=True)
