import argparse
import glob
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("viz")

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.manifold import TSNE
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt

import src.config as config
from src.models import get_model

# -------- Config (same as your inference) --------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP")


def get_test_transforms():
    return A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE, interpolation=cv2.INTER_LANCZOS4),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def resolve_device(req: str) -> str:
    req = (req or "").strip().lower()
    # if req == "cuda" and torch.cuda.is_available():
    #     return req
    # return "cpu"
    if req == "cpu": return "cpu"
    return req if torch.cuda.is_available() else "cpu"


def list_images(input_path: str, limit: int | None):
    if os.path.isdir(input_path):
        paths = []
        for ext in SUPPORTED_EXT:
            paths += glob.glob(os.path.join(input_path, "**", f"*{ext}"), recursive=True)
    else:
        paths = [input_path]
    paths = [p for p in paths if os.path.isfile(p)]
    if limit: paths = paths[:limit]
    return paths


# --- Plotly helpers ---
def _filename_only(p: str) -> str:
    import os
    return os.path.basename(p)


def save_plotly_embed(Z, labels, paths, class_names, out_html: str, title="Embedding map"):
    """
    Z: np.array shape [N,2] (2D embedding)
    labels: np.array shape [N] with class indices
    paths: list[str] length N (original file paths)
    class_names: list[str] length C
    out_html: path to save interactive html
    """
    df = pd.DataFrame({
        "x": Z[:, 0],
        "y": Z[:, 1],
        "label_idx": labels,
        "label": [class_names[i] if 0 <= i < len(class_names) else f"class_{i}" for i in labels],
        "file": [_filename_only(p) for p in paths],
        "path": paths,
    })

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="label",
        hover_name="file",  # big title on hover
        hover_data={"path": True,  # show full path
                    "label": True,
                    "label_idx": True,
                    "x": ':.3f', "y": ':.3f'},
        title=title,
        template="plotly_white",
    )
    fig.update_traces(marker=dict(size=6, opacity=0.8))
    fig.update_layout(legend_title_text="Class", hovermode="closest")
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"[saved] Interactive plot: {out_html}")


def save_plotly_stats(mean_scalar, std_scalar, paths, out_html: str, title="Normalized pixel stats"):
    """
    mean_scalar, std_scalar: arrays length N (per-image mean/std after Normalize)
    paths: list[str] of file paths
    """
    df = pd.DataFrame({
        "mean": mean_scalar,
        "std": std_scalar,
        "file": [os.path.basename(p) for p in paths],
        "path": paths
    })
    fig = px.scatter(
        df, x="mean", y="std",
        hover_name="file",
        hover_data={"path": True, "mean": ':.4f', "std": ':.4f'},
        title=title, template="plotly_white"
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.add_hline(y=1, line_dash="dash", line_color="gray")
    fig.update_traces(marker=dict(size=6, opacity=0.8))
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"[saved] Interactive plot: {out_html}")


def load_and_normalize(paths, transform, device, batch_size):
    """Yield (batch_tensor[B,3,224,224], batch_labels or None, batch_paths).
       Labels optional: infer from parent folder if it matches CLASS_NAMES.
    """
    batch_imgs, batch_lbls, batch_pths = [], [], []
    name_to_idx = {n: i for i, n in enumerate(getattr(config, "CLASS_NAMES", []))}
    for p in tqdm(paths):
        img = cv2.imread(p)
        if img is None:
            logger.warning(f"Unreadable: {p}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = transform(image=img)["image"]  # CHW float32 normalized
        batch_imgs.append(t)
        # optional label from folder name
        parent = Path(p).parent.name
        batch_lbls.append(name_to_idx.get(parent, -1))
        batch_pths.append(p)
        if len(batch_imgs) == batch_size:
            bt = torch.stack(batch_imgs).to(device)
            yield bt, batch_lbls, batch_pths
            batch_imgs, batch_lbls, batch_pths = [], [], []
    if batch_imgs:
        bt = torch.stack(batch_imgs).to(device)
        yield bt, batch_lbls, batch_pths


# -------------------- MODE 1: Normalization stats --------------------
def viz_norm_stats(paths, device, batch_size, out_png):
    logger.info("Computing per-image normalized pixel stats...")
    tfm = get_test_transforms()
    means, stds, labels = [], [], []
    proc_paths = []

    for bt, bl, bp in tqdm(load_and_normalize(paths, tfm, device, batch_size)):
        # bt is normalized already (Imagenet). Compute per-image per-channel stats in normalized space.
        # Convert CHW -> NCHW stats
        b = bt.detach().cpu().numpy()  # [B,3,224,224]
        m = b.mean(axis=(2, 3))  # [B,3]
        s = b.std(axis=(2, 3))  # [B,3]
        means.append(m)
        stds.append(s)
        labels += bl
        proc_paths.extend(bp)

    if not means:
        logger.error("No images processed for stats.")
        return

    means = np.concatenate(means, axis=0)  # [N,3]
    stds = np.concatenate(stds, axis=0)  # [N,3]
    labels = np.array(labels)

    # Scatter plot: mean vs std per image (averaged across channels) as quick sanity
    mean_scalar = means.mean(axis=1)  # ~0 if normalization ok
    std_scalar = stds.mean(axis=1)  # ~1 if normalization ok

    save_plotly_stats(
        mean_scalar=mean_scalar,
        std_scalar=std_scalar,
        paths=proc_paths,
        out_html=out_png.replace('.png', '.html') if out_png else "norm_stats.html",
        title="Per-image normalized pixel stats (interactive)"
    )

    plt.figure(figsize=(8, 6))
    plt.scatter(mean_scalar, std_scalar, s=10, alpha=0.6)
    plt.axvline(0, color='k', lw=1, ls='--')
    plt.axhline(1, color='k', lw=1, ls='--')
    plt.title("Per-image normalized stats")
    plt.xlabel("Mean across channels (expected ~0)")
    plt.ylabel("Std across channels (expected ~1)")
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=160)
        print(f"[saved] {out_png}")
    else:
        plt.show()
    plt.close()


# -------------------- MODE 2: Embedding scatter --------------------
def get_feature_extractor(model):
    """
    Returns a callable f(x)->features.
    Fallback: use logits (still fine for a quick 2D map).
    If model has 'forward_features' (e.g., efficientnet/vit), prefer that.
    """
    if hasattr(model, "forward_features") and callable(model.forward_features):
        def _f(x):
            with torch.no_grad():
                feats = model.forward_features(x)
                # pool if needed to [B, C]
                if feats.ndim == 4:
                    feats = feats.mean(dim=(2, 3))
                return feats

        return _f
    else:
        # fallback: use logits
        logger.warning(f"Model has no 'forward_features'; using logits as features.")

        def _f(x):
            with torch.no_grad():
                return model(x)  # [B, num_classes]

        return _f


def reduce_2d(X, method="umap"):
    """Return 2D array using UMAP/t-SNE/PCA depending on availability."""
    method = method.lower()
    logger.info(f"Reducing features to 2D using {method.upper()}...")
    if method == "umap":
        try:
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
            return reducer.fit_transform(X)
        except Exception as e:
            logger.error(f"UMAP reduction failed - {e}, falling back to PCA.")

    if method == "tsne":
        try:
            return TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30, random_state=42).fit_transform(
                X)
        except Exception as e:
            logger.error(f"t-SNE reduction failed - {e}, falling back to PCA.")

    # PCA fallback (always available via sklearn)
    return PCA(n_components=2, random_state=42).fit_transform(X)


def viz_embeddings(paths, checkpoint, model_name, device, batch_size, method, out_png, color_by="auto"):
    num_classes = int(getattr(config, "OUTPUT_FEATURES", 3))
    class_names = getattr(config, "CLASS_NAMES", [str(i) for i in range(num_classes)])

    # load model
    model = get_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()

    tfm = get_test_transforms()
    extractor = get_feature_extractor(model)

    feats_list, lbls_list, proc_paths = [], [], []
    preds_list = []

    for bt, bl, bp in load_and_normalize(paths, tfm, device, batch_size):
        with torch.no_grad():
            # logits for predictions
            logits = model(bt)  # [B, C]
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            # features for embedding
            f = extractor(bt).detach().cpu().numpy()

        feats_list.append(f)
        lbls_list += bl
        preds_list += preds
        proc_paths.extend(bp)

    if not feats_list:
        logger.error("No features extracted.");
        return

    X = np.concatenate(feats_list, axis=0)  # [N, D]
    y_lbl = np.array(lbls_list)  # ground truth (may be -1)
    y_pred = np.array(preds_list)  # model predictions (0..C-1)
    assert len(proc_paths) == X.shape[0] == y_lbl.shape[0] == y_pred.shape[0]

    # Choose coloring
    if color_by == "label":
        labels_plot = y_lbl
        legend_names = class_names + ["unlabeled"]
        # keep all points; matplotlib needs an unlabeled bucket
    elif color_by == "pred":
        labels_plot = y_pred
        legend_names = class_names
    else:  # auto
        if np.any(y_lbl != -1):
            labels_plot = y_lbl
            legend_names = class_names + ["unlabeled"]
        else:
            labels_plot = y_pred
            legend_names = class_names

    # 2D reduction on ALL processed samples (don’t drop unlabeled)
    Z = reduce_2d(X, method=method)  # [N,2]

    # ---- Plotly (interactive) ----
    # Map -1 to "unlabeled" only when plotting ground-truth labels
    if labels_plot is y_lbl:
        label_text = [class_names[i] if (0 <= i < len(class_names)) else "unlabeled" for i in labels_plot]
        label_idx_for_df = labels_plot  # can include -1
    else:
        label_text = [class_names[i] for i in labels_plot]
        label_idx_for_df = labels_plot

    save_plotly_embed(
        Z=Z,
        labels=label_idx_for_df,
        paths=proc_paths,
        class_names=class_names,
        out_html=out_png.replace('.png', '.html') if out_png else "embedding_map.html",
        title=f"Embedding map (interactive) — colored by {'labels' if labels_plot is y_lbl else 'predictions'}"
    )

    # ---- Matplotlib (static) ----
    plt.figure(figsize=(8, 7))
    cmap = plt.cm.get_cmap('tab10', num_classes)

    if labels_plot is y_lbl:
        # draw labeled classes
        for ci in range(num_classes):
            sel = (labels_plot == ci)
            if sel.any():
                plt.scatter(Z[sel, 0], Z[sel, 1], s=10, alpha=0.7, label=class_names[ci], c=[cmap(ci)])
        # draw unlabeled (=-1) in gray
        unl = (labels_plot == -1)
        if unl.any():
            plt.scatter(Z[unl, 0], Z[unl, 1], s=10, alpha=0.6, label="unlabeled", c=["#999999"])
        title_suffix = "colored by labels"
    else:
        # colored by predictions
        for ci in range(num_classes):
            sel = (labels_plot == ci)
            if sel.any():
                plt.scatter(Z[sel, 0], Z[sel, 1], s=10, alpha=0.7, label=class_names[ci], c=[cmap(ci)])
        title_suffix = "colored by predictions"

    plt.legend(markerscale=2, fontsize=9, loc="best")
    plt.title(f"Embedding map ({method.upper()}) — {title_suffix}")
    plt.xlabel("dim-1");
    plt.ylabel("dim-2")
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=180);
        print(f"[saved] {out_png}")
    else:
        plt.show()
    plt.close()


# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="Dataset visualization after normalization.")
    ap.add_argument("-i", "--input", required=True, help="Image file or directory")
    ap.add_argument("-c", "--checkpoint", required=False, help="Model .pth (required for --mode embed)")
    ap.add_argument("-m", "--model_name", required=False,
                    choices=['vit_b_16', 'vit_l_32', 'efficientnet_v2_l', 'efficientnet_v2_s', 'resnet18', 'resnet34'],
                    help="Model arch (required for --mode embed)")
    ap.add_argument("--mode", choices=["stats", "embed"], default="stats",
                    help="'stats' = pixel normalization sanity; 'embed' = 2D feature map")
    ap.add_argument("-o", "--output_png", default=None, help="If set, save plot to this PNG path")
    ap.add_argument("-d", "--device", default="cuda", help="cpu | cuda")
    ap.add_argument("-b", "--batch_size", type=int, default=1)
    ap.add_argument("-l", "--limit", type=int, default=None)
    ap.add_argument("--method", default="umap", choices=["umap", "tsne", "pca"],
                    help="Dimensionality reduction for --mode embed")
    ap.add_argument("--color_by", default="auto", choices=["auto", "label", "pred"],
                    help="Color points by: ground-truth 'label', model 'pred', or 'auto' (use label if present else pred).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    device = resolve_device(args.device)
    paths = list_images(args.input, args.limit)
    if not paths:
        logger.error("No images found.")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output_png), exist_ok=True)

    if args.mode == "stats":
        viz_norm_stats(paths, device, args.batch_size, args.output_png)
    else:
        if not args.checkpoint or not args.model_name:
            logger.error("--mode embed requires --checkpoint and --model_name")
            sys.exit(1)
        viz_embeddings(paths, args.checkpoint, args.model_name, device, args.batch_size, args.method, args.output_png, color_by=args.color_by)


if __name__ == "__main__":
    main()
