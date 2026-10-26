# -*- coding: utf-8 -*-

import os
import csv
from copy import deepcopy

import numpy as np
import torch
import scipy.stats as stats
import yaml
from ultralytics import YOLO
from custom_layers import register_custom_layers

# ===================== 全局配置 =====================
DATASET_TEMPLATE = "/hy-tmp/split_datasets/seed_{seed}/dataset.yaml"
MODELS_DIR = "/hy-tmp/models/attention_universality_n"
PROJECT_DIR = "/hy-tmp/runs/attention_universality_n"
DETAIL_CSV = os.path.join(PROJECT_DIR, "universality_detail.csv")
EPOCHS = 200
SEEDS = [42, 123, 256, 512, 1024]
NUM_CLASSES = 3

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PROJECT_DIR, exist_ok=True)

BASE_TRAIN_CFG = {
    "epochs": EPOCHS, "batch": 32, "imgsz": 640,
    "deterministic": True, "verbose": False, "plots": True,
    "cache": False, "workers": 8, "cos_lr": True,
    "warmup_epochs": 3, "optimizer": "SGD", "lr0": 0.01,
    "momentum": 0.937, "weight_decay": 0.0005, "patience": 0, "iou": 0.7,
}

# ===================== 实验矩阵 =====================

BACKBONES = {
    "yolov5n": "yolov5nu.pt",
    "yolov8n": "yolov8n.pt",
    "yolov10n": "yolov10n.pt",
    "yolo11n": "yolo11n.pt",
}


ATTENTION_CONFIGS = {
    "None":  None,
    "GGCA":  ("GGCA",                [1024, 16, 4], [256, 16, 4]),
    "CBAM":  ("CBAM",                [1024],        [256]),
    "EMA":   ("EMA",                 [1024],        [256]),
    "CA":    ("CoordinateAttention", [1024],        [256]),
    "SimAM": ("SimAM",               [1024],        [256]),
}

# ===================== 结构提取与注意力插入 =====================

FALLBACK_SCALE = {
    "yolov5n": [0.33, 0.25], "yolov8n": [0.33, 0.25],
    "yolov10n": [0.33, 0.25], "yolo11n": [0.50, 0.25],
}

def get_official_cfg(source):
    """"""
    if source.endswith((".yaml", ".yml")):
        return yaml.safe_load(open(source, encoding="utf-8"))
    m = YOLO(source)
    cfg = deepcopy(m.model.yaml)
    del m
    torch.cuda.empty_cache()
    return cfg


def _fix_refs(layers, pos):
    """"""
    for i in range(pos + 1, len(layers)):
        f = layers[i][0]
        if isinstance(f, int):
            if f >= pos:
                layers[i][0] = f + 1
        else:
            layers[i][0] = [x + 1 if (isinstance(x, int) and x >= pos) else x for x in f]


def build_variant_yaml(backbone_name, att_name):
    """"""
    cfg = get_official_cfg(BACKBONES[backbone_name])
    cfg["nc"] = NUM_CLASSES
    
    scales = cfg.get("scales") or {}
    d, w = (list(scales.values())[0][:2] if scales else FALLBACK_SCALE[backbone_name])
    cfg["scales"] = {"n": [d, w, 1024], "s": [d, w, 1024]}

    att = ATTENTION_CONFIGS[att_name]
    if att is not None:
        layers = cfg["backbone"] + cfg["head"]
        nb = len(cfg["backbone"])

        
        sppf_i = next(i for i, l in enumerate(layers) if l[2] == "SPPF")
        layers.insert(sppf_i, [-1, 1, att[0], att[1]])
        _fix_refs(layers, sppf_i)

       
        det = layers[-1]
        assert "Detect" in det[2], f"末层不是检测头: {det[2]}"
        p3 = det[0][0]
        layers.insert(p3 + 1, [-1, 1, att[0], att[2]])
        _fix_refs(layers, p3 + 1)
        layers[-1][0][0] = p3 + 1

        cfg["backbone"] = layers[:nb + 1]
        cfg["head"] = layers[nb + 1:]

    yaml_path = os.path.join(MODELS_DIR, f"{backbone_name}_{att_name}.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return yaml_path

# ===================== 单次训练与评估 =====================
def train_eval_one(backbone_name, att_name, seed):
    dataset_yaml = DATASET_TEMPLATE.format(seed=seed)
    exp_name = f"{backbone_name}_{att_name}_seed{seed}"
    cfg = BASE_TRAIN_CFG.copy()
    cfg.update({"seed": seed, "project": PROJECT_DIR})

    model = YOLO(build_variant_yaml(backbone_name, att_name))
    try:
        model.load(BACKBONES[backbone_name])   
    except Exception as e:
        print(f"  ⚠️ 预训练权重部分加载异常（已忽略）: {str(e)[:80]}")
    model.train(data=dataset_yaml, name=exp_name, **cfg)

    best_pt = os.path.join(PROJECT_DIR, exp_name, "weights", "best.pt")
    best = YOLO(best_pt)
    res = best.val(data=dataset_yaml, split="test", imgsz=640, verbose=False)

    P = float(res.box.mp) * 100
    R = float(res.box.mr) * 100
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    ap50 = np.atleast_1d(res.box.ap50)

    row = {
        "backbone": backbone_name, "attention": att_name, "seed": seed,
        "mAP@0.5": round(float(res.box.map50) * 100, 2),
        "mAP@0.5:0.95": round(float(res.box.map) * 100, 2),
        "Precision": round(P, 2), "Recall": round(R, 2), "F1": round(F1, 2),
        "head_AP@0.5": round(float(ap50[0]) * 100, 2),
        "tail_AP@0.5": round(float(ap50[1]) * 100, 2),
        "limb_AP@0.5": round(float(ap50[2]) * 100, 2),
        "Params(M)": round(sum(p.numel() for p in best.model.parameters()) / 1e6, 4),
        "Latency(ms)": round(float(res.speed["inference"]), 4),
    }
    del model, best
    torch.cuda.empty_cache()
    return row

# ===================== 主流程 =====================
if __name__ == "__main__":
    register_custom_layers()

    
    print("架构自检（None 组参数量，应与 Table 3 同模型接近）：")
    for bb in BACKBONES:
        _m = YOLO(build_variant_yaml(bb, "None"))
        print(f"  {bb}: {sum(p.numel() for p in _m.model.parameters())/1e6:.3f}M")
        del _m
    torch.cuda.empty_cache()

    all_rows = []
    first = True
    total = len(BACKBONES) * len(ATTENTION_CONFIGS) * len(SEEDS)
    cur = 0

    for bb in BACKBONES:
        for att_name in ATTENTION_CONFIGS:
            print(f"\n{'='*66}\n🚀 {bb} + {att_name}\n{'='*66}")
            for seed in SEEDS:
                cur += 1
                try:
                    row = train_eval_one(bb, att_name, seed)
                    all_rows.append(row)
                    with open(DETAIL_CSV, "a", newline="", encoding="utf-8-sig") as f:
                        w = csv.DictWriter(f, fieldnames=list(row.keys()))
                        if first:
                            w.writeheader(); first = False
                        w.writerow(row)
                    print(f"[{cur}/{total}] {bb}+{att_name} seed{seed} | "
                          f"mAP@0.5={row['mAP@0.5']}  P={row['Precision']}  R={row['Recall']}")
                except Exception as e:
                    print(f"  ❌ [{bb}+{att_name} | seed {seed}] 失败: {str(e)[:120]}")

    # ===================== 汇总 1：各组合 mean±SD =====================
    def ms(vals):
        v = np.array(vals, dtype=float)
        return f"{v.mean():.2f}±{v.std(ddof=1):.2f}", v.mean()

    print("\n" + "=" * 78)
    print("各组合汇总（mean ± SD over five runs）：")
    print(f"{'Backbone':<10}{'Attention':<10}{'mAP@0.5':>15}{'mAP@0.5:0.95':>15}"
          f"{'Precision':>13}{'Recall':>13}{'Params(M)':>11}")
    print("-" * 78)
    for bb in BACKBONES:
        for att_name in ATTENTION_CONFIGS:
            sub = [r for r in all_rows if r["backbone"] == bb and r["attention"] == att_name]
            if len(sub) < len(SEEDS):
                print(f"{bb:<10}{att_name:<10} 仅有 {len(sub)}/{len(SEEDS)} 次成功")
                continue
            s5, _ = ms([r["mAP@0.5"] for r in sub])
            s95, _ = ms([r["mAP@0.5:0.95"] for r in sub])
            sp, _ = ms([r["Precision"] for r in sub])
            sr, _ = ms([r["Recall"] for r in sub])
            pm = np.mean([r["Params(M)"] for r in sub])
            print(f"{bb:<10}{att_name:<10}{s5:>15}{s95:>15}{sp:>13}{sr:>13}{pm:>10.3f}M")

    # ===================== 汇总 2：通用性矩阵（ΔmAP@0.5 vs None + 配对 t 检验） =====================
    print("\n" + "=" * 78)
    print("通用性矩阵：各注意力相对 None 的 ΔmAP@0.5（配对 t 检验，* p<0.05）")
    atts = [a for a in ATTENTION_CONFIGS if a != "None"]
    print(f"{'Backbone':<10}{'None mAP':>10}" + "".join(f"{a:>16}" for a in atts))
    print("-" * 78)
    for bb in BACKBONES:
        none_rows = sorted([r for r in all_rows if r["backbone"] == bb and r["attention"] == "None"],
                           key=lambda r: r["seed"])
        if len(none_rows) < len(SEEDS):
            print(f"{bb:<10} None 组不完整，跳过")
            continue
        n5 = np.array([r["mAP@0.5"] for r in none_rows])
        line = f"{bb:<10}{n5.mean():>9.2f} "
        for att_name in atts:
            sub = sorted([r for r in all_rows if r["backbone"] == bb and r["attention"] == att_name],
                         key=lambda r: r["seed"])
            if len(sub) < len(SEEDS):
                line += f"{'N/A':>16}"
                continue
            a5 = np.array([r["mAP@0.5"] for r in sub])
            _, p = stats.ttest_rel(a5, n5)
            sig = "*" if p < 0.05 else ""
            line += f"{a5.mean()-n5.mean():>+8.2f}({p:.3f}){sig:<2}"
        print(line)

    print(f"\n📁 明细: {DETAIL_CSV}")