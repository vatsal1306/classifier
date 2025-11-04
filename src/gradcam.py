# inference.py
import argparse
import glob
import logging
import os
import sys
from typing import List, Optional

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

import src.config as config
from src.models import get_model

logger = logging.getLogger(__name__)

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
DISPLAY_SIZE = 512
PANEL_WIDTH = 260
CAM_BLEND = 0.45


def get_test_transforms():
    return A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE, interpolation=cv2.INTER_LANCZOS4),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# -------------------- Plain (no CAM) visualization --------------------
def save_annotated_prediction(image_path, output_dir, pred_name, probs):
    """Save [original 512 | right text panel] (no CAM)."""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        logger.warning(f"Could not read image {image_path}, skipping.")
        return
    image_bgr = cv2.resize(image_bgr, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_LANCZOS4)
    h, w, _ = image_bgr.shape

    text_panel = np.full((h, PANEL_WIDTH, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, color, thick = 0.55, (0, 0, 0), 1
    y = 26
    cv2.putText(text_panel, f"Pred: {pred_name}", (10, y), font, scale, color, thick)
    y += 26
    for i, p in enumerate(probs):
        cv2.putText(text_panel, f"P[{config.CLASS_NAMES[i]}]: {p:.3f}", (10, y), font, scale, color, thick)
        y += 22

    canvas = np.hstack([image_bgr, text_panel])
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(out_path, canvas)


# -------------------- In-script CAM implementations --------------------
def _is_vit(model_name: str) -> bool:
    return model_name.startswith("vit_")


def _find_last_conv2d(model: nn.Module) -> Optional[nn.Module]:
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last


class GradCAMConv:
    """
    Minimal Grad-CAM for CNNs.
    - Hooks the last Conv2d layer to capture activations and gradients.
    - CAM = ReLU( sum_k (avg_pool(grad_k)) * act_k ).
    Returns heatmap in [0,1] at conv spatial size; caller can upsample.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module, device: str):
        self.model = model
        self.target_layer = target_layer
        self.device = device

        self.activations = None
        self.gradients = None

        def fwd_hook(m, inp, out):
            self.activations = out.detach()

        def bwd_hook(m, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._fh = target_layer.register_forward_hook(fwd_hook)
        self._bh = target_layer.register_full_backward_hook(bwd_hook)

    def cam_for_batch(self, input_batch: torch.Tensor, class_ids: List[int]) -> List[np.ndarray]:
        """
        input_batch: [B,3,H,W] on device
        class_ids: length B (target class per sample)
        Returns list of CAM heatmaps (Hc,Wc) normalized to [0,1]
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_batch)  # [B,C]
        loss = 0.0
        for i, cid in enumerate(class_ids):
            loss = loss + logits[i, cid]
        loss.backward(retain_graph=False)

        acts = self.activations  # [B, K, Hc, Wc]
        grads = self.gradients  # [B, K, Hc, Wc]
        cams = []

        B, K, Hc, Wc = acts.shape
        for i in range(B):
            a = acts[i]  # [K,Hc,Wc]
            g = grads[i]  # [K,Hc,Wc]
            # global-average-pool grads over spatial dims
            weights = g.mean(dim=(1, 2))  # [K]
            cam = (weights[:, None, None] * a).sum(dim=0)  # [Hc,Wc]
            cam = torch.relu(cam)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            cams.append(cam.detach().cpu().numpy())
        return cams


class ViTAttentionRollout:
    """
    Basic Attention Rollout for torchvision ViT models.
    - Multiplies attention matrices across layers to get a token-level importance.
    - Drops CLS token and reshapes to (h,w).
    Returns heatmap in [0,1] at token grid size (e.g., 14x14 for 224/16).
    """

    def __init__(self, vit: nn.Module):
        self.vit = vit
        # try to infer patch grid
        patch = vit.patch_size if isinstance(vit.patch_size, int) else vit.patch_size[0]
        self.grid = IMAGE_SIZE // patch  # e.g., 14

        self.attn_mats: List[torch.Tensor] = []

        def enc_block_hook(mod, inp, out):
            # hook on the multihead self-attention to get attention probs
            # In torchvision ViT, block.attn.attn_drop comes after softmax
            # safer: hook block.attn.attn_drop's input (which is already softmax)
            pass  # we'll attach per-block below

        # Attach a forward hook on each encoder block to collect attn
        self._hooks = []
        try:
            for blk in vit.encoder.layers:
                # each block has blk.attn: MultiheadAttention-like wrapper
                # we hook into blk.attn.attn_drop (its input is the attention probs)
                def make_hook():
                    def _hook(module, input, output):
                        # input is a tuple; take the first arg (attention probs)
                        attn = input[0]  # [B, heads, Tokens, Tokens]
                        self.attn_mats.append(attn.detach())

                    return _hook

                h = blk.attn.attn_drop.register_forward_hook(make_hook())
                self._hooks.append(h)
        except Exception as e:
            # If structure differs, we won't break; rollout simply won't run.
            logger.warning(f"Could not attach ViT attention hooks: {e}")

    def __del__(self):
        for h in getattr(self, "_hooks", []):
            try:
                h.remove()
            except Exception:
                pass

    @torch.no_grad()
    def rollout(self, input_batch: torch.Tensor) -> List[np.ndarray]:
        """
        Returns a list of rollout heatmaps (grid x grid) per sample, in [0,1].
        """
        self.attn_mats.clear()
        _ = self.vit(input_batch)  # forward pass to collect attention maps

        if not self.attn_mats:
            return [None] * input_batch.shape[0]

        # Multiply attention across layers (using average over heads per layer)
        # attn_mats: list of [B, heads, T, T]
        B = self.attn_mats[0].shape[0]
        T = self.attn_mats[0].shape[-1]
        rollouts = []
        for b in range(B):
            attn_avg = None
            for A in self.attn_mats:
                A_b = A[b].mean(dim=0)  # [T,T]
                A_b = A_b + torch.eye(T, device=A_b.device)  # residual
                A_b = A_b / A_b.sum(dim=-1, keepdim=True)
                attn_avg = A_b if attn_avg is None else attn_avg @ A_b
            # drop CLS token (index 0), keep patch tokens: T-1
            mask = attn_avg[0, 1:]  # importance from CLS to patches, [T-1]
            mask = mask / (mask.max() + 1e-8)
            mask = mask.reshape(self.grid, self.grid)  # (h,w)
            rollouts.append(mask.detach().cpu().numpy())
        return rollouts


def _overlay_cam_on_image(base_bgr_512: np.ndarray, cam_small: np.ndarray) -> np.ndarray:
    H, W = base_bgr_512.shape[:2]
    cam_resized = cv2.resize(cam_small, (W, H), interpolation=cv2.INTER_LINEAR)
    cam_uint8 = np.uint8(255 * np.clip(cam_resized, 0, 1))
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(heatmap, CAM_BLEND, base_bgr_512, 1.0 - CAM_BLEND, 0)


def _render_right_panel(pred_name: str, probs: List[float]) -> np.ndarray:
    canvas = np.full((DISPLAY_SIZE, PANEL_WIDTH, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, color, thick = 0.55, (0, 0, 0), 1
    y = 26
    cv2.putText(canvas, f"Pred: {pred_name}", (10, y), font, scale, color, thick)
    y += 26
    for i, p in enumerate(probs):
        cv2.putText(canvas, f"P[{config.CLASS_NAMES[i]}]: {p:.3f}", (10, y), font, scale, color, thick)
        y += 22
    return canvas


def save_visualization_with_cam(image_path, output_dir, pred_name, probs, cam_map: Optional[np.ndarray]):
    """Save [original 512 | CAM overlay 512 | right panel]."""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        logger.warning(f"Could not read image {image_path}, skipping.")
        return
    orig_512 = cv2.resize(image_bgr, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_LANCZOS4)
    cam_overlay = _overlay_cam_on_image(orig_512, cam_map) if cam_map is not None else orig_512.copy()
    right_panel = _render_right_panel(pred_name, probs)
    composite = np.hstack([orig_512, cam_overlay, right_panel])
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(out_path, composite)


# -------------------- Predictor --------------------
class Predictor:
    """Encapsulates model, transforms, in-script CAM, and prediction logic."""

    def __init__(self, model_name, checkpoint_path, device="cuda", enable_cam: bool = False):
        self.device = device
        self.model_name = model_name
        self.transform = get_test_transforms()
        self.enable_cam = enable_cam

        logger.info(f"Loading model '{model_name}' from {checkpoint_path}")
        self.model = get_model(model_name, pretrained=False, num_classes=config.OUTPUT_FEATURES)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully.")

        # In-script CAM objects (created only if requested)
        self.cnn_cam: Optional[GradCAMConv] = None
        self.vit_rollout: Optional[ViTAttentionRollout] = None

        if self.enable_cam:
            try:
                if _is_vit(self.model_name):
                    # For torchvision ViT we’ll do attention rollout (no gradients)
                    self.vit_rollout = ViTAttentionRollout(self.model)
                    logger.info("CAM: ViT Attention Rollout enabled.")
                else:
                    last_conv = _find_last_conv2d(self.model)
                    if last_conv is None:
                        logger.warning("No Conv2d layer found; CAM disabled.")
                        self.enable_cam = False
                    else:
                        self.cnn_cam = GradCAMConv(self.model, last_conv, self.device)
                        logger.info(f"CAM: Grad-CAM on {last_conv.__class__.__name__} enabled.")
            except Exception as e:
                logger.warning(f"CAM init failed ({e}). Continuing without CAM.")
                self.enable_cam = False

    def _prep_tensor(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image_rgb)
        return transformed['image']  # CHW tensor

    def _compute_cam_single(self, input_tensor: torch.Tensor, pred_id: int) -> Optional[np.ndarray]:
        """
        Compute CAM map for a single item [1,3,224,224].
        - CNN: Grad-CAM returns conv-size map; upsample later.
        - ViT : Attention rollout (grid x grid).
        """
        if not self.enable_cam:
            return None
        try:
            if self.cnn_cam is not None:
                cams = self.cnn_cam.cam_for_batch(input_tensor, [pred_id])
                return cams[0] if cams else None
            if self.vit_rollout is not None:
                maps = self.vit_rollout.rollout(input_tensor)
                return maps[0] if maps else None
        except Exception as e:
            logger.warning(f"CAM failed: {e}")
        return None

    def _compute_cam_batch(self, batch: torch.Tensor, pred_ids: List[int]) -> List[Optional[np.ndarray]]:
        if not self.enable_cam:
            return [None] * batch.shape[0]
        try:
            if self.cnn_cam is not None:
                return self.cnn_cam.cam_for_batch(batch, pred_ids)
            if self.vit_rollout is not None:
                return self.vit_rollout.rollout(batch)
        except Exception as e:
            logger.warning(f"Batch CAM failed: {e}")
        return [None] * batch.shape[0]

    def predict_image(self, image_path):
        """
        Single-image inference:
          returns (pred_id, pred_name, probs, cam_small or None)
        """
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            logger.warning(f"Could not read image {image_path}, skipping.")
            return None, None, None, None

        image_tensor = self._prep_tensor(image_bgr).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

        pred_id = int(np.argmax(probs))
        cam_map = self._compute_cam_single(image_tensor, pred_id)
        pred_name = config.CLASS_NAMES[pred_id]
        return pred_id, pred_name, probs, cam_map

    def predict_batch(self, image_paths):
        """
        Batched inference (preserves order, skips unreadable).
          returns list of dicts:
            { "path": str, "pred_id": int, "pred_name": str, "probs": list[float], "cam_map": Optional[np.ndarray] }
        """
        tensors, keep_paths = [], []
        for p in image_paths:
            img = cv2.imread(p)
            if img is None:
                logger.warning(f"Could not read image {p}, skipping.")
                continue
            tensors.append(self._prep_tensor(img))
            keep_paths.append(p)

        if not tensors:
            return []

        batch = torch.stack(tensors, dim=0).to(self.device)  # [B,3,224,224]
        with torch.no_grad():
            logits = self.model(batch)
            probs_all = F.softmax(logits, dim=1).cpu().tolist()

        pred_ids = [int(np.argmax(p)) for p in probs_all]
        cam_maps = self._compute_cam_batch(batch, pred_ids)

        results = []
        for p, probs, cam in zip(keep_paths, probs_all, cam_maps):
            pred_id = int(np.argmax(probs))
            results.append({
                "path": p,
                "pred_id": pred_id,
                "pred_name": config.CLASS_NAMES[pred_id],
                "probs": probs,
                "cam_map": cam
            })
        return results


# -------------------- Device & main --------------------
def _resolve_device(arg_device: str) -> str:
    req = (arg_device or "").strip().lower()
    if req == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return arg_device  # allow "cuda" or "cuda:0"
    logger.warning("CUDA requested but not available; falling back to CPU.")
    return "cpu"


def main(args):
    if not os.path.exists(args.input):
        logger.error(f"Input path does not exist: {args.input}")
        return
    os.makedirs(args.output, exist_ok=True)

    device = _resolve_device(args.device)

    # Collect images
    if os.path.isdir(args.input):
        image_paths = []
        image_paths.extend(glob.glob(os.path.join(args.input, '**', '*.[jJ][pP]*[gG]'), recursive=True))
        image_paths.extend(glob.glob(os.path.join(args.input, '**', '*.[pP][nN][gG]'), recursive=True))
        logger.info(f"Found {len(image_paths)} images in directory: {args.input}")
    else:
        image_paths = [args.input]
        logger.info(f"Processing single image: {args.input}")

    if not image_paths:
        logger.warning("No images found to process.")
        return

    if args.limit:
        image_paths = image_paths[:args.limit]

    # Initialize predictor (CAM optional)
    predictor = Predictor(args.model_name, args.checkpoint, device, enable_cam=args.save_cam)

    # Inference
    bs = max(1, int(args.batch_size))
    if device != "cpu" and bs > 1:
        logger.info(f"Running batched inference on {device} with batch_size={bs}")
        for i in tqdm(range(0, len(image_paths), bs), desc="Running Inference (batches)"):
            chunk = image_paths[i:i + bs]
            results = predictor.predict_batch(chunk)
            for r in results:
                if args.save_cam and r["cam_map"] is not None:
                    save_visualization_with_cam(r["path"], args.output, r["pred_name"], r["probs"], r["cam_map"])
                elif args.save_cam:
                    # CAM requested but unavailable for this image/model → fallback without CAM
                    save_annotated_prediction(r["path"], args.output, r["pred_name"], r["probs"])
                else:
                    save_annotated_prediction(r["path"], args.output, r["pred_name"], r["probs"])
    else:
        if device == "cpu":
            logger.info("Running single-image inference on CPU")
        else:
            logger.info("Running single-image inference (batch_size=1)")
        for p in tqdm(image_paths, desc="Running Inference"):
            pred_id, pred_name, probs, cam_map = predictor.predict_image(p)
            if pred_name is None:
                continue
            if args.save_cam and cam_map is not None:
                save_visualization_with_cam(p, args.output, pred_name, probs, cam_map)
            elif args.save_cam:
                save_annotated_prediction(p, args.output, pred_name, probs)
            else:
                save_annotated_prediction(p, args.output, pred_name, probs)

    logger.info(f"Inference complete. Results saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone inference script for 3-class classification.")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to a single image or a directory of images.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Path to the directory where outputs will be saved.")
    parser.add_argument("-c", "--checkpoint", type=str, required=True,
                        help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("-m", "--model_name", type=str, required=True,
                        choices=['vit_b_16', 'vit_l_32', 'efficientnet_v2_l', 'efficientnet_v2_s', 'resnet18',
                                 'resnet34'],
                        help="Name of the model architecture to use.")
    parser.add_argument("-d", "--device", type=str, default="cuda",
                        help="Device: 'cpu', 'cuda', or 'cuda:N' (falls back to CPU if unavailable).")
    parser.add_argument("-l", "--limit", type=int, help="Limit how many images to process")
    parser.add_argument("-b", "--batch_size", type=int, default=1,
                        help="Batch size for inference. If >1 and using CUDA, runs in batches.")
    parser.add_argument("--save_cam", action="store_true",
                        help="If set, saves [original | CAM overlay | probs panel]. Otherwise saves [original | probs panel].")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
