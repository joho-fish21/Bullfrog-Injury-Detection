# -*- coding: utf-8 -*-


import os
import csv
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import scipy.stats as stats
import yaml
from ultralytics import YOLO
from custom_layers import register_custom_layers

# ===================== 全局配置（与损失消融一致） =====================
DATASET_TEMPLATE = "/hy-tmp/split_datasets/seed_{seed}/dataset.yaml"
MODELS_DIR = "/hy-tmp/models/backbone_ablation_v8n_cbam"
PROJECT_DIR = "/hy-tmp/runs/backbone_ablation_v8n_cbam"
DETAIL_CSV = os.path.join(PROJECT_DIR, "backbone_detail.csv")
ALL_CSV = os.path.join(PROJECT_DIR, "backbone_ablation_all.csv")
# 基线来源：损失消融最终合并表中的 CIoU 行（= yolov8n+CBAM 原骨干）
BASELINE_CSV = "/hy-tmp/runs/loss_ablation_v8n_cbam/loss_ablation_all.csv"
BASELINE_NAME = "Baseline(CBAM+CIoU)"

EPOCHS = 200
SEEDS = [42, 123, 256, 512, 1024]
NUM_CLASSES = 3
IMG_SIZE = 640
BACKBONE_PT = "yolov8n.pt"
FALLBACK_SCALE = [0.33, 0.25]  # yolov8n 官方 n 档 [depth, width]

# 变体 -> custom_layers 中的模块名
# 注意：GhostNetV2 必须用 GhostBottleneckV2（独立子类名）——
# 直接用 GhostBottleneck 会与 ultralytics 原生同名模块冲突，
# 类对象混入 parse_model 的 base_modules 集合导致通道被双重缩放（1024→256→64）
VARIANTS = {
    "FasterNet": "FasterNetBlock",
    "MobileNetV3": "MBConv",
    "ShuffleNetV2": "ShuffleBottleneck",
    "GhostNetV2": "GhostBottleneckV2",
}

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PROJECT_DIR, exist_ok=True)

BASE_TRAIN_CFG = {
    "epochs": EPOCHS, "batch": 32, "imgsz": 640,
    "deterministic": True, "verbose": False, "plots": True,
    "cache": False, "workers": 8, "cos_lr": True,
    "warmup_epochs": 3, "optimizer": "SGD", "lr0": 0.01,
    "momentum": 0.937, "weight_decay": 0.0005, "patience": 0, "iou": 0.7,
}

# ===================== 结构生成 =====================
def get_official_cfg(source):
    if source.endswith((".yaml", ".yml")):
        return yaml.safe_load(open(source, encoding="utf-8"))
    m = YOLO(source)
    cfg = deepcopy(m.model.yaml)
    del m
    torch.cuda.empty_cache()
    return cfg


def _fix_refs(layers, pos):
    """在 pos 处插入一层后，其后所有层的绝对层号引用 +1（f=-1 相对引用不变）"""
    for i in range(pos + 1, len(layers)):
        f = layers[i][0]
        if isinstance(f, int):
            if f >= pos:
                layers[i][0] = f + 1
        else:
            layers[i][0] = [x + 1 if (isinstance(x, int) and x >= pos) else x for x in f]


def build_variant_yaml(variant):
    """官方 yolov8n 结构 → 替换 backbone C2f 为轻量块 → 插入 CBAM×2
    （顺序：先替换后插入，插入位置按名称定位，不受替换影响）"""
    cfg = get_official_cfg(BACKBONE_PT)
    cfg["nc"] = NUM_CLASSES
    scales = cfg.get("scales") or {}
    d, w = (list(scales.values())[0][:2] if scales else FALLBACK_SCALE)
    cfg["scales"] = {"n": [d, w, 1024], "s": [d, w, 1024]}  # 双键同值

    # ① 替换 backbone 中的 C2f（保留输出通道参数，模块名换成轻量块）
    block = VARIANTS[variant]
    n_replaced = 0
    for i, layer in enumerate(cfg["backbone"]):
        if layer[2] == "C2f":
            c2 = layer[3][0]  # 原始尺度通道（128/256/512/1024）
            cfg["backbone"][i] = [-1, 1, block, [c2]]
            n_replaced += 1
    assert n_replaced == 4, f"预期替换 4 个 backbone C2f，实际 {n_replaced}"

    # ② 插入 CBAM×2（与损失消融完全相同的定位逻辑）
    layers = cfg["backbone"] + cfg["head"]
    nb = len(cfg["backbone"])
    sppf_i = next(i for i, l in enumerate(layers) if l[2] == "SPPF")
    layers.insert(sppf_i, [-1, 1, "CBAM", [1024]])
    _fix_refs(layers, sppf_i)
    det = layers[-1]
    assert "Detect" in det[2]
    p3 = det[0][0]
    layers.insert(p3 + 1, [-1, 1, "CBAM", [256]])
    _fix_refs(layers, p3 + 1)
    layers[-1][0][0] = p3 + 1
    cfg["backbone"], cfg["head"] = layers[:nb + 1], layers[nb + 1:]

    yaml_path = os.path.join(MODELS_DIR, f"yolov8n_CBAM_{variant}.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return yaml_path


# ===================== 单次训练与评估 =====================
def train_eval_one(variant, seed):
    exp_name = f"{variant}_seed{seed}"
    best_pt = os.path.join(PROJECT_DIR, exp_name, "weights", "best.pt")
    if os.path.exists(best_pt):
        print(f"  已存在 {exp_name}/weights/best.pt，跳过（如需重练请删除该目录）")
        return None

    dataset_yaml = DATASET_TEMPLATE.format(seed=seed)
    cfg = BASE_TRAIN_CFG.copy()
    cfg.update({"seed": seed, "project": PROJECT_DIR})

    model = YOLO(build_variant_yaml(variant))
    try:
        model.load(BACKBONE_PT)   # 保留层迁移权重；替换块随机初始化
    except Exception as e:
        print(f"  ⚠️ 预训练权重部分加载异常（已忽略）: {str(e)[:80]}")
    model.train(data=dataset_yaml, name=exp_name, **cfg)

    best = YOLO(best_pt)
    res = best.val(data=dataset_yaml, split="test", imgsz=IMG_SIZE, verbose=False)
    P = float(res.box.mp) * 100
    R = float(res.box.mr) * 100
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    ap50 = np.atleast_1d(res.box.ap50)
    row = {
        "backbone": variant, "seed": seed,
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

    # ---- 架构自检：各变体参数量（基线 yolov8n+CBAM ≈ 3.02M） ----
    print("架构自检（参数量；基线 yolov8n+CBAM ≈ 3.02M）：")
    for v in VARIANTS:
        _m = YOLO(build_variant_yaml(v))
        n_param = sum(p.numel() for p in _m.model.parameters()) / 1e6
        print(f"  {v:<14} {n_param:.3f}M")
        del _m
    torch.cuda.empty_cache()

    all_rows = []
    first = not os.path.exists(DETAIL_CSV)
    total = len(VARIANTS) * len(SEEDS)
    cur = 0
    for variant in VARIANTS:
        print(f"\n{'=' * 66}\n🚀 backbone 替换 = {variant}\n{'=' * 66}")
        for seed in SEEDS:
            cur += 1
            try:
                row = train_eval_one(variant, seed)
                if row is None:
                    continue
                all_rows.append(row)
                with open(DETAIL_CSV, "a", newline="", encoding="utf-8-sig") as f:
                    w = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if first:
                        w.writeheader()
                        first = False
                    w.writerow(row)
                print(f"[{cur}/{total}] {variant} seed{seed} | "
                      f"mAP@0.5={row['mAP@0.5']}  mAP@0.5:0.95={row['mAP@0.5:0.95']}  "
                      f"P={row['Precision']}  R={row['Recall']}  Params={row['Params(M)']}M")
            except Exception as e:
                print(f"  ❌ [{variant} | seed {seed}] 失败: {str(e)[:150]}")

    # ===================== 汇总：基线 + 4 变体，配对 t 检验 =====================
    frames = []
    if os.path.exists(DETAIL_CSV):
        frames.append(pd.read_csv(DETAIL_CSV))
    if os.path.exists(BASELINE_CSV):
        b = pd.read_csv(BASELINE_CSV)
        b = b[b["loss"] == "CIoU"].copy()
        if len(b) > 0:
            b = b.drop(columns=["loss"], errors="ignore")
            b.insert(0, "backbone", BASELINE_NAME)
            frames.append(b)
            print(f"\n已载入基线 {len(b)} 条（{BASELINE_CSV} 中 loss==CIoU 行）")
        else:
            print(f"\n⚠️ {BASELINE_CSV} 中没有 loss==CIoU 的行")
    else:
        print(f"\n⚠️ 未找到 {BASELINE_CSV}，基线缺失（请先运行 CIoU 基线补测脚本）")

    if frames:
        df_all = pd.concat(frames, ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["backbone", "seed"], keep="last")
        df_all.to_csv(ALL_CSV, index=False, encoding="utf-8-sig")

        METRICS = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1",
                   "head_AP@0.5", "tail_AP@0.5", "limb_AP@0.5", "Params(M)"]

        def ms(vals):
            v = np.array(vals, dtype=float)
            s = v.std(ddof=1) if len(v) > 1 else 0.0
            return f"{v.mean():.2f}±{s:.2f}"

        print("\n" + "=" * 108)
        print("Backbone 替换消融汇总（mean ± SD，按实际完成 run 数统计）：")
        print(f"{'Backbone':<22}" + "".join(f"{m:>15}" for m in METRICS))
        print("-" * 108)
        order = [BASELINE_NAME] + list(VARIANTS)
        for v in order:
            sub = df_all[df_all["backbone"] == v]
            if len(sub) == 0:
                print(f"{v:<22} 暂无记录")
                continue
            cells = []
            for m in METRICS:
                if m == "Params(M)":
                    cells.append(f"{sub[m].mean():>14.3f} ")
                else:
                    cells.append(f"{ms(sub[m]):>15}")
            suffix = f"  (n={len(sub)}/{len(SEEDS)})" if len(sub) < len(SEEDS) else ""
            print(f"{v:<22}" + "".join(cells) + suffix)

        base = df_all[df_all["backbone"] == BASELINE_NAME].sort_values("seed")
        if len(base) >= 2:
            print(f"\n相对 {BASELINE_NAME} 的配对 t 检验（逐 seed 配对，* p<0.05）：")
            for v in VARIANTS:
                sub = df_all[df_all["backbone"] == v].sort_values("seed")
                common = sorted(set(base["seed"]) & set(sub["seed"]))
                if len(common) < 2:
                    print(f"  {v:<14} 配对样本不足（{len(common)} 对），跳过")
                    continue
                b_paired = base[base["seed"].isin(common)]
                s_paired = sub[sub["seed"].isin(common)]
                parts = []
                for m in ["mAP@0.5", "mAP@0.5:0.95", "head_AP@0.5", "tail_AP@0.5", "limb_AP@0.5"]:
                    a = b_paired[m].values.astype(float)
                    bb = s_paired[m].values.astype(float)
                    if np.allclose(a, bb):
                        parts.append(f"Δ{m}=+0.00(p=1.0)")
                        continue
                    t, p = stats.ttest_rel(bb, a)
                    star = "*" if p < 0.05 else ""
                    parts.append(f"Δ{m}={np.mean(bb - a):+.2f}(p={p:.4f}){star}")
                note = "" if len(common) == len(SEEDS) else f"  [仅 {len(common)} 对，试跑参考]"
                print(f"  {v:<14} " + "  ".join(parts) + note)
        print(f"\n📁 明细: {DETAIL_CSV}")
        print(f"📁 合并: {ALL_CSV}")