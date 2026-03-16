"""
Analytic studies for PEEK variance and loss geometry.
"""

from __future__ import annotations

import csv
import glob
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from .models import build_imagenette_model
from .utils.paths import repo_path


IMAGENETTE_SYNSETS = [
    "n01440764",
    "n02102040",
    "n02979186",
    "n03000684",
    "n03028079",
    "n03394916",
    "n03417042",
    "n03425413",
    "n03445777",
    "n03888257",
]

IMAGENETTE_SYNSET_TO_NAME = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


@dataclass
class GeometryConfig:
    p_keep: float = 0.20
    topk_min: int = 16
    topk_frac: float = 0.10
    eps_z_l: float = 1e-3
    eps_z_sig: float = 1e-3
    grid_steps: int = 9
    alpha_max: float = 1e-3
    beta_max: float = 1e-3
    peek_eps: float = 1e-8


def imagenette_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def list_imagenette_val_images(
    val_dir: Union[str, Path],
    *,
    one_per_class: bool = False,
    synsets: Optional[list[str]] = None,
) -> list[Path]:
    val_dir_p = repo_path(val_dir)
    class_ids = synsets or IMAGENETTE_SYNSETS
    paths: list[Path] = []
    for synset in class_ids:
        matches = sorted(Path(p) for p in glob.glob(str(val_dir_p / synset / "*")))
        if one_per_class:
            if matches:
                paths.append(matches[0])
        else:
            paths.extend(matches)
    return paths


def load_image_tensor(
    image_path: Union[str, Path],
    *,
    transform: Optional[transforms.Compose] = None,
) -> tuple[Image.Image, torch.Tensor]:
    image = Image.open(repo_path(image_path)).convert("RGB")
    tensor = (transform or imagenette_eval_transform())(image).unsqueeze(0)
    return image, tensor


def torch_peek_conv(a: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    PEEK over channels at each spatial coordinate.
    """
    a_reshaped = a.reshape(a.size(0), -1)
    mins = a_reshaped.min(dim=1, keepdim=True).values.view(a.size(0), 1, 1, 1)
    a_plus = a - mins + eps
    return -(a_plus * torch.log(a_plus)).sum(dim=1)


def sigma2_from_z(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = torch_peek_conv(z, eps=eps)
    p = p - p.mean()
    return (p * p).mean()


def center_spatial_np(X: np.ndarray, W: Optional[np.ndarray] = None) -> np.ndarray:
    if W is None:
        mu = X.mean(axis=(1, 2), keepdims=True)
    else:
        wsum = W.sum(axis=(1, 2), keepdims=True) + 1e-300
        mu = (X * W).sum(axis=(1, 2), keepdims=True) / wsum
    return X - mu


def s2_ip(A: np.ndarray, B: np.ndarray, W: np.ndarray) -> float:
    return float(((A * B) * W).sum())


def rb_overlay(mask: np.ndarray, H: int, W: int) -> np.ndarray:
    rb = np.zeros((H, W, 3), dtype=np.float32)
    rb[mask] = [1.0, 0.0, 0.0]
    rb[~mask] = [0.0, 0.0, 1.0]
    return (rb * 255).astype(np.uint8)


def _resize_array(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    import cv2

    return cv2.resize(arr, (width, height), interpolation=cv2.INTER_LINEAR)


def _forward_spatial_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "features"):
        return model.features(x)
    if hasattr(model, "backbone"):
        return model.backbone(x)
    raise ValueError("Model must expose `.features` or `.backbone` for dense-head analysis.")


def _get_dense_head(model: nn.Module) -> nn.Linear:
    if hasattr(model, "fc"):
        return model.fc
    if hasattr(model, "head"):
        return model.head
    raise ValueError("Model must expose `.fc` or `.head` for dense-head analysis.")


def _reset_linear_head(linear: nn.Linear, random_seed: int) -> None:
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(random_seed)
    nn.init.kaiming_uniform_(linear.weight, a=np.sqrt(5))
    if linear.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
        bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(linear.bias, -bound, bound)
    torch.random.set_rng_state(generator_state)


def build_dense_head_pair(
    model_name: str,
    checkpoint_path: Union[str, Path],
    *,
    num_classes: int = len(IMAGENETTE_SYNSETS),
    dropout: float = 0.2,
    device: Optional[Union[str, torch.device]] = None,
    random_seed: int = 1337,
) -> tuple[nn.Module, nn.Module]:
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(repo_path(checkpoint_path), map_location=device_obj)

    trained = build_imagenette_model(
        model_name,
        num_classes=num_classes,
        pretrained=False,
        dropout=dropout,
    ).to(device_obj)
    trained.load_state_dict(checkpoint, strict=False)
    trained.eval()

    random_head = build_imagenette_model(
        model_name,
        num_classes=num_classes,
        pretrained=False,
        dropout=dropout,
    ).to(device_obj)
    random_head.load_state_dict(checkpoint, strict=False)
    _reset_linear_head(_get_dense_head(random_head), random_seed)
    random_head.eval()

    return trained, random_head


def build_vgg16_dense_head_pair(
    checkpoint_path: Union[str, Path],
    *,
    num_classes: int = len(IMAGENETTE_SYNSETS),
    dropout: float = 0.2,
    device: Optional[Union[str, torch.device]] = None,
    random_seed: int = 1337,
) -> tuple[nn.Module, nn.Module]:
    return build_dense_head_pair(
        "vgg16",
        checkpoint_path,
        num_classes=num_classes,
        dropout=dropout,
        device=device,
        random_seed=random_seed,
    )


def compute_dense_head_geometry_from_features(
    z: torch.Tensor,
    fc: nn.Linear,
    y_idx: int,
    class_ids: list[str],
    *,
    image_hw: Optional[tuple[int, int]] = None,
    config: Optional[GeometryConfig] = None,
    criterion: Optional[nn.Module] = None,
    compute_maps: bool = True,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """
    Compute loss/variance geometry for a fixed conv feature tensor and dense head.
    """
    cfg = config or GeometryConfig()
    loss_fn = criterion or nn.CrossEntropyLoss()
    z = z.detach().clone().requires_grad_(True)

    flat = torch.flatten(z, 1)
    logits = fc(flat)
    q = torch.softmax(logits, dim=1)[0]
    pred_idx = int(torch.argmax(logits, dim=1))
    y = torch.tensor([y_idx], device=z.device)

    loss = loss_fn(logits, y)
    gL = torch.autograd.grad(loss, z, create_graph=False, retain_graph=True)[0]

    p_conv = torch_peek_conv(z, eps=cfg.peek_eps)
    mu_conv = p_conv.mean()
    sigma2 = ((p_conv - mu_conv) ** 2).mean()
    gS = torch.autograd.grad(sigma2, z, create_graph=False, retain_graph=True)[0]

    gL64 = gL.detach().reshape(-1).double().cpu().numpy()
    gS64 = gS.detach().reshape(-1).double().cpu().numpy()
    nL = float(np.linalg.norm(gL64))
    nS = float(np.linalg.norm(gS64))
    cos_global = (
        float(np.clip(np.dot(gL64 / nL, gS64 / nS), -1.0, 1.0))
        if nL > 0 and nS > 0
        else float("nan")
    )

    fmap = np.moveaxis(z.detach().cpu().numpy()[0], 0, -1).astype(np.float32)
    hc, wc, channels = fmap.shape
    a_chw = z.detach().cpu().numpy()[0]
    a_min = a_chw.min()
    a_plus = a_chw - a_min + cfg.peek_eps

    T2_conv = (p_conv - mu_conv).detach().cpu().numpy()[0].astype(np.float32)
    T3_conv = np.sum(np.log(a_plus) + 1.0, axis=0).astype(np.float32)

    W_fc = fc.weight.detach().cpu()
    W_cls = (
        W_fc.view(len(class_ids), channels, hc, wc)
        .permute(0, 2, 3, 1)
        .contiguous()
        .numpy()
    )
    delta = torch.zeros_like(q)
    delta[y_idx] = 1.0
    coeff = (q - delta).detach().cpu().numpy()[..., None, None, None]
    G = np.sum(coeff * W_cls, axis=0)
    T1_conv = G.sum(axis=-1).astype(np.float32)

    num_closed = float(np.sum(T1_conv * T2_conv * T3_conv))
    den_closed = float(np.sum((T2_conv**2) * (T3_conv**2)) + 1e-12)
    dL_dsigma2_closed = num_closed / den_closed if den_closed != 0.0 else 0.0

    sal = np.abs(T2_conv * T3_conv).ravel()
    thr = np.quantile(sal, 1.0 - cfg.p_keep) if cfg.p_keep < 1.0 else -np.inf
    keep = (sal >= thr).reshape(T2_conv.shape)

    gL_chw = gL.detach().cpu().double().numpy()[0]
    gS_chw = gS.detach().cpu().double().numpy()[0]

    v1 = gL_chw[:, keep].reshape(-1)
    v2 = gS_chw[:, keep].reshape(-1)
    cos_saliency = (
        float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-300))
        if v1.size
        else float("nan")
    )

    num_map = (gL_chw * gS_chw).sum(axis=0)
    den_map = np.linalg.norm(gL_chw, axis=0) * np.linalg.norm(gS_chw, axis=0) + 1e-300
    cos_map = num_map / den_map
    cos_loc = float(cos_map[keep].mean()) if keep.any() else float("nan")

    logit_y = logits[0, y_idx]
    g_logit = torch.autograd.grad(logit_y, z, retain_graph=True)[0]
    w_cam = g_logit.mean(dim=(2, 3)).abs()[0].cpu().numpy()
    K = max(cfg.topk_min, int(cfg.topk_frac * w_cam.shape[0]))
    top_idx = np.argsort(-w_cam)[:K]

    S_chw = (np.log(a_plus) + 1.0).astype(np.float64)
    LwK = gL_chw[top_idx] * S_chw[top_idx]
    gSK = gS_chw[top_idx]
    Wmet = S_chw[top_idx] ** 2
    LwK_c = center_spatial_np(LwK, Wmet)
    gSK_c = center_spatial_np(gSK, Wmet)
    numK = s2_ip(LwK_c, gSK_c, Wmet)
    denK = np.sqrt(s2_ip(LwK_c, LwK_c, Wmet) * s2_ip(gSK_c, gSK_c, Wmet)) + 1e-300
    cos_sigma_topk = float(numK / denK)

    dot = float((gL * gS).sum().item())
    gS_sq = float((gS * gS).sum().item())
    proj_mag = dot / (np.sqrt(gS_sq) + 1e-300)
    local_coupling = dot / (gS_sq + 1e-300)

    gS_hat = gS / (gS.norm() + 1e-300)
    gL_par = (gL * gS_hat).sum() * gS_hat
    gL_orth = gL - gL_par
    n_par = float(gL_par.norm().item())
    n_orth = float(gL_orth.norm().item())

    gL_hat = gL / (gL.norm() + 1e-300)
    with torch.no_grad():
        s2_base = float(sigma2_from_z(z, eps=cfg.peek_eps).item())
        L_base = float(loss_fn(fc(torch.flatten(z, 1)), y).item())

        z1 = z - cfg.eps_z_l * gL_hat
        s2_1 = float(sigma2_from_z(z1, eps=cfg.peek_eps).item())
        L_1 = float(loss_fn(fc(torch.flatten(z1, 1)), y).item())

        z2 = z + cfg.eps_z_sig * gS_hat
        s2_2 = float(sigma2_from_z(z2, eps=cfg.peek_eps).item())
        L_2 = float(loss_fn(fc(torch.flatten(z2, 1)), y).item())

    alphas = np.linspace(-cfg.alpha_max, cfg.alpha_max, cfg.grid_steps)
    betas = np.linspace(-cfg.beta_max, cfg.beta_max, cfg.grid_steps)
    grid_L = np.zeros((cfg.grid_steps, cfg.grid_steps), dtype=np.float32)
    grid_S2 = np.zeros((cfg.grid_steps, cfg.grid_steps), dtype=np.float32)
    with torch.no_grad():
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                z_ab = z + alpha * gL_hat + beta * gS_hat
                grid_L[i, j] = float(loss_fn(fc(torch.flatten(z_ab, 1)), y).item()) - L_base
                grid_S2[i, j] = float(sigma2_from_z(z_ab, eps=cfg.peek_eps).item()) - s2_base

    metrics = {
        "pred_index": pred_idx,
        "pred_synset": class_ids[pred_idx],
        "score": float(logits[0, pred_idx].item()),
        "margin": float(torch.topk(logits[0], k=2).values.diff().abs().item()),
        "cos": cos_global,
        "cos_sal(20%)": cos_saliency,
        "cos_loc": cos_loc,
        "cos_sigma@topK": cos_sigma_topk,
        "norm_ratio(‖∇σ²‖/‖∇L‖)": nS / (nL + 1e-300),
        "proj(∇L on ∇σ²)": proj_mag,
        "frac_parallel(∇L)": cos_global,
        "local_dL/dsigma2": local_coupling,
        "||parallel||": n_par,
        "||orthogonal||": n_orth,
        "dL/dsigma2_closed": float(dL_dsigma2_closed),
        "nudge_-gL_dS2": float(s2_1 - s2_base),
        "nudge_-gL_dL": float(L_1 - L_base),
        "nudge_+gS_dS2": float(s2_2 - s2_base),
        "nudge_+gS_dL": float(L_2 - L_base),
        "grid_max|ΔL|": float(np.max(np.abs(grid_L))),
        "grid_max|Δσ²|": float(np.max(np.abs(grid_S2))),
        "||gL||": nL,
        "||gS||": nS,
    }

    maps: dict[str, np.ndarray] = {}
    if compute_maps:
        maps["T1_conv"] = T1_conv.astype(np.float32)
        maps["T2_conv"] = T2_conv.astype(np.float32)
        maps["T3_conv"] = T3_conv.astype(np.float32)
        maps["grid_L"] = grid_L.astype(np.float32)
        maps["grid_S2"] = grid_S2.astype(np.float32)
        maps["alphas"] = alphas.astype(np.float32)
        maps["betas"] = betas.astype(np.float32)

        if image_hw is not None:
            H, W = image_hw
            T1_up = _resize_array(T1_conv, W, H)
            T3_up = _resize_array(T3_conv, W, H)
            peek_map = _resize_array(T2_conv, W, H) + float(np.mean(T2_conv))
            T2_up = peek_map - float(np.mean(peek_map))
            prod_sign = (
                np.where(T1_up > 0, 1, -1)
                * np.where(T2_up > 0, 1, -1)
                * np.where(T3_up > 0, 1, -1)
            ) > 0
            maps["PEEK_img"] = peek_map.astype(np.float32)
            maps["T1_up"] = T1_up.astype(np.float32)
            maps["T2_up"] = T2_up.astype(np.float32)
            maps["T3_up"] = T3_up.astype(np.float32)
            maps["prod_sign"] = prod_sign.astype(np.uint8)

    return metrics, maps


def analyze_dense_head_image(
    model: nn.Module,
    image_path: Union[str, Path],
    *,
    y_idx: Optional[int] = None,
    class_ids: Optional[list[str]] = None,
    transform: Optional[transforms.Compose] = None,
    config: Optional[GeometryConfig] = None,
    criterion: Optional[nn.Module] = None,
    compute_maps: bool = True,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """
    Analyze one image for a model exposing a spatial backbone and dense head.
    """
    class_ids_resolved = class_ids or IMAGENETTE_SYNSETS
    image, tensor = load_image_tensor(image_path, transform=transform)
    tensor = tensor.to(next(model.parameters()).device)

    if y_idx is None:
        synset = repo_path(image_path).parent.name
        if synset not in class_ids_resolved:
            raise ValueError(f"Could not infer target class from path: {image_path}")
        y_idx = class_ids_resolved.index(synset)

    with torch.enable_grad():
        z = _forward_spatial_features(model, tensor)
        z.retain_grad()

    metrics, maps = compute_dense_head_geometry_from_features(
        z,
        _get_dense_head(model),
        y_idx,
        class_ids_resolved,
        image_hw=(image.size[1], image.size[0]),
        config=config,
        criterion=criterion,
        compute_maps=compute_maps,
    )
    return metrics, maps


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_dense_head_analytic_study(
    model_name: str,
    val_dir: Union[str, Path],
    checkpoint_path: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    one_per_class: bool = False,
    config: Optional[GeometryConfig] = None,
    device: Optional[Union[str, torch.device]] = None,
    random_seed: int = 1337,
    save_npz: bool = True,
) -> dict[str, Path]:
    """
    Reproduce the trained-vs-random-head dense-head analysis as reusable repo code.
    """
    cfg = config or GeometryConfig()
    out_dir_p = repo_path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    map_dir = out_dir_p / "maps"
    if save_npz:
        map_dir.mkdir(parents=True, exist_ok=True)

    trained, random_head = build_dense_head_pair(
        model_name,
        checkpoint_path,
        num_classes=len(IMAGENETTE_SYNSETS),
        dropout=0.2,
        device=device,
        random_seed=random_seed,
    )

    rows_train: list[dict[str, Any]] = []
    rows_random: list[dict[str, Any]] = []
    transform = imagenette_eval_transform()
    image_paths = list_imagenette_val_images(val_dir, one_per_class=one_per_class)

    for image_path in image_paths:
        image, tensor = load_image_tensor(image_path, transform=transform)
        tensor = tensor.to(next(trained.parameters()).device)
        synset = repo_path(image_path).parent.name
        y_idx = IMAGENETTE_SYNSETS.index(synset)
        true_name = IMAGENETTE_SYNSET_TO_NAME[synset]
        image_name = Path(image_path).name
        image_hw = (image.size[1], image.size[0])

        with torch.enable_grad():
            z_train = _forward_spatial_features(trained, tensor)
            z_train.retain_grad()
        metrics_train, maps_train = compute_dense_head_geometry_from_features(
            z_train,
            _get_dense_head(trained),
            y_idx,
            IMAGENETTE_SYNSETS,
            image_hw=image_hw,
            config=cfg,
            compute_maps=save_npz,
        )
        metrics_train.update(
            {
                "image": image_name,
                "true": true_name,
                "pred": IMAGENETTE_SYNSET_TO_NAME[metrics_train["pred_synset"]],
                "image_path": str(repo_path(image_path)),
            }
        )

        with torch.enable_grad():
            z_random = _forward_spatial_features(random_head, tensor)
            z_random.retain_grad()
        metrics_random, maps_random = compute_dense_head_geometry_from_features(
            z_random,
            _get_dense_head(random_head),
            y_idx,
            IMAGENETTE_SYNSETS,
            image_hw=image_hw,
            config=cfg,
            compute_maps=save_npz,
        )
        metrics_random.update(
            {
                "image": image_name,
                "true": true_name,
                "pred": IMAGENETTE_SYNSET_TO_NAME[metrics_random["pred_synset"]],
                "image_path": str(repo_path(image_path)),
            }
        )

        rows_train.append(metrics_train)
        rows_random.append(metrics_random)

        if save_npz:
            out_npz = map_dir / f"{image_name}.npz"
            np.savez_compressed(
                out_npz,
                image=image_name,
                synset=synset,
                true_name=true_name,
                H=np.int16(image_hw[0]),
                W=np.int16(image_hw[1]),
                T1_train=maps_train["T1_conv"],
                T1_rand=maps_random["T1_conv"],
                T2=maps_train["T2_conv"],
                T3=maps_train["T3_conv"],
                PEEK_img=maps_train.get("PEEK_img"),
                T1_up_train=maps_train.get("T1_up"),
                T1_up_rand=maps_random.get("T1_up"),
                T2_up=maps_train.get("T2_up"),
                T3_up=maps_train.get("T3_up"),
                prod_sign_train=maps_train.get("prod_sign"),
                prod_sign_rand=maps_random.get("prod_sign"),
                grid_L_train=maps_train["grid_L"],
                grid_S2_train=maps_train["grid_S2"],
                grid_L_rand=maps_random["grid_L"],
                grid_S2_rand=maps_random["grid_S2"],
                alphas=maps_train["alphas"],
                betas=maps_train["betas"],
                image_path=str(repo_path(image_path)),
            )

    train_csv = out_dir_p / "peek_loss_geometry_summary_basictrain.csv"
    rand_csv = out_dir_p / "peek_loss_geometry_summary_random_head.csv"
    config_json = out_dir_p / "geometry_config.json"

    _write_csv(train_csv, rows_train)
    _write_csv(rand_csv, rows_random)
    config_json.write_text(json.dumps(asdict(cfg), indent=2))

    return {
        "train_csv": train_csv,
        "random_csv": rand_csv,
        "map_dir": map_dir,
        "config": config_json,
    }


def run_vgg16_dense_head_analytic_study(
    val_dir: Union[str, Path],
    checkpoint_path: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    one_per_class: bool = False,
    config: Optional[GeometryConfig] = None,
    device: Optional[Union[str, torch.device]] = None,
    random_seed: int = 1337,
    save_npz: bool = True,
) -> dict[str, Path]:
    return run_dense_head_analytic_study(
        "vgg16",
        val_dir,
        checkpoint_path,
        out_dir,
        one_per_class=one_per_class,
        config=config,
        device=device,
        random_seed=random_seed,
        save_npz=save_npz,
    )
