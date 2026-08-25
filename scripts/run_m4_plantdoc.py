#!/usr/bin/env python3
"""
PlantDoc 3-seed unified-protocol experiments (same protocol as run_m4_seeds.py)
- 统一协议：所有模型从同一 ImageNet-1K 预训练权重出发，FT 5ep
  (AdamW, lr 5e-5, wd 0.01, batch 32, cosine, AMP, seed 控制 shuffle)
- PlantDoc: 2336 张 train 固定 80/20 划分（seed 42 random_split，与旧实验一致）
- seeds: 42, 123, 2026
- 每 seed: baseline FT + prune25/50/75 (noFT eval + FT5ep)
- 输出: results/m4_pd/m4_pd_seed{seed}.json（每阶段完成即写盘，可续跑）
"""
import torch, torch.nn as nn, json, time, os, copy, warnings
import timm
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = True

BASE = Path("REPO_ROOT")
DATASET_PATH = Path("DATASETS/plantdoc")
PT_WEIGHTS = Path("PRETRAINED_VIT")
OUT_DIR = BASE / "results" / "m4_pd"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda")
SEEDS = [42, 123, 2026]
BATCH = 32
EPOCHS = 5
LR = 5e-5
WD = 0.01
SPLIT_SEED = 42  # 固定 80/20 划分，与旧实验一致

train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
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


def make_model(num_classes):
    from safetensors.torch import load_file
    m = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
    sd = load_file(str(PT_WEIGHTS))
    # 剥离 1000 类分类头（本地 head 随机初始化，由 FT 学习）
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
        (OUT_DIR / f"m4_pd_seed{seed}.json").write_text(json.dumps(out, indent=2))
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
    out_path = OUT_DIR / f"m4_pd_seed{seed}.json"
    if out_path.exists():
        out = json.loads(out_path.read_text())
        if out.get("done"):
            print(f"seed {seed} 已完成，跳过", flush=True)
            return out
    else:
        out = {"seed": seed, "progress": {}, "results": {}, "done": False}

    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed); np.random.seed(seed)

    full_train = datasets.ImageFolder(str(DATASET_PATH / "train"), transform=train_tf)
    num_classes = len(full_train.classes)
    n_val = int(len(full_train) * 0.2)
    train_ds, val_ds = torch.utils.data.random_split(
        full_train, [len(full_train) - n_val, n_val],
        generator=torch.Generator().manual_seed(SPLIT_SEED)
    )
    val_ds.dataset.transform = val_tf
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    out["num_classes"] = num_classes
    out["split"] = {"train": len(train_ds), "val": len(val_ds), "split_seed": SPLIT_SEED}
    print(f"\n=== seed {seed} | {len(train_ds)} train / {len(val_ds)} val, {num_classes} classes ===", flush=True)

    # baseline FT
    if "baseline" not in out["results"]:
        print(f"  [baseline FT5ep]", flush=True)
        m = make_model(num_classes).to(DEVICE)
        best = finetune(m, train_loader, val_loader, seed, "baseline", out)
        out["results"]["baseline"] = round(best * 100, 2)
        out["progress"]["baseline"] = {"val": round(best * 100, 2), "done": True}
        (OUT_DIR / f"m4_pd_seed{seed}.json").write_text(json.dumps(out, indent=2))

    # prune ratios
    for ratio in [0.25, 0.50, 0.75]:
        tag = f"prune{int(ratio*100)}"
        if f"{tag}_ft" in out["results"]:
            continue
        print(f"  [{tag}]", flush=True)
        m = make_model(num_classes).to(DEVICE)
        pm = prune_heads(m, ratio)
        if f"{tag}_noft" not in out["results"]:
            va = evaluate(pm, val_loader)
            out["results"][f"{tag}_noft"] = round(va * 100, 2)
            print(f"    noFT: {va*100:.2f}%", flush=True)
            (OUT_DIR / f"m4_pd_seed{seed}.json").write_text(json.dumps(out, indent=2))
        best = finetune(pm, train_loader, val_loader, seed, tag, out)
        out["results"][f"{tag}_ft"] = round(best * 100, 2)
        (OUT_DIR / f"m4_pd_seed{seed}.json").write_text(json.dumps(out, indent=2))

    out["done"] = True
    (OUT_DIR / f"m4_pd_seed{seed}.json").write_text(json.dumps(out, indent=2))
    print(f"seed {seed} 完成: {out['results']}", flush=True)
    return out


if __name__ == "__main__":
    t0 = time.time()
    for s in SEEDS:
        run_seed(s)
    print(f"\nALL DONE in {(time.time()-t0)/3600:.2f}h", flush=True)
