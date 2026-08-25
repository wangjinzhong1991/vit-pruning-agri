#!/usr/bin/env python3
"""PlantDoc combined-configuration evaluation (prune + FT + INT8, M4-aligned protocol)
流程与 run_m4_plantdoc.py 完全一致：timm 预训练权重 -> prune(25/50/75) -> FT5ep -> INT8 eval。
预期 FT 复现 ~67.67/59.53/48.61（M4 seed-42），INT8 损失应 <0.5pp。
输出：results/m5_pd_combo_m4.json（与 Table 3 同源）
"""
import json, time, torch, torch.nn as nn, copy, warnings
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
OUT_PATH = BASE / "results" / "m5_pd_combo_m4.json"
DEVICE = torch.device("cuda")
BATCH, EPOCHS, LR, WD, SPLIT_SEED = 32, 5, 5e-5, 0.01, 42

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


def make_model(num_classes):
    from safetensors.torch import load_file
    m = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
    sd = load_file(str(PT_WEIGHTS))
    sd = {k: v for k, v in sd.items() if not k.startswith("head.")}
    missing, _ = m.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if not k.startswith("head.")]
    assert not real_missing, f"missing: {real_missing[:5]}"
    return m


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


@torch.no_grad()
def evaluate(model, loader, device=DEVICE):
    model.eval()
    cor = tot = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        _, pred = model(imgs).max(1)
        cor += pred.eq(lbls).sum().item()
        tot += lbls.size(0)
    return cor / tot


def finetune(model, tag):
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


def main():
    torch.manual_seed(42); np.random.seed(42)
    full_train = datasets.ImageFolder(str(DATASET_PATH / "train"), transform=train_tf)
    num_classes = len(full_train.classes)
    n_val = int(len(full_train) * 0.2)
    train_ds, val_ds = torch.utils.data.random_split(
        full_train, [len(full_train) - n_val, n_val],
        generator=torch.Generator().manual_seed(SPLIT_SEED))
    val_ds.dataset.transform = val_tf
    g = torch.Generator().manual_seed(42)
    global train_loader, val_loader
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    print(f"train {len(train_ds)} / val {len(val_ds)} / {num_classes} classes", flush=True)

    out = {}
    for ratio in [25, 50, 75]:
        m = make_model(num_classes).to(DEVICE)
        pm = prune_heads(m, ratio / 100)
        best = finetune(pm, f"PD prune{ratio} FT")
        qm = torch.ao.quantization.quantize_dynamic(copy.deepcopy(pm).to("cpu"), {nn.Linear}, dtype=torch.qint8)
        qa = evaluate(qm, val_loader, device=torch.device("cpu"))
        print(f"PD prune{ratio}: FT={best*100:.2f}% FT+INT8={qa*100:.2f}%", flush=True)
        out[f"prune{ratio}"] = {"ft": round(best * 100, 2), "ft_int8": round(qa * 100, 2)}
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print("DONE ->", OUT_PATH, flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"{(time.time()-t0)/60:.1f} min", flush=True)
