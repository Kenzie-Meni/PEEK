"""
General PyTorch model latent extraction.

Extracts intermediate activations ("latents after a module") from arbitrary
PyTorch models using forward hooks.

Supports torchvision models (resnet50, convnext, etc.) and custom models.

Output:
    One pickle per image:
        {module_index: torch.Tensor}
"""

from __future__ import annotations

import glob
import operator
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch
import torchvision.models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from peek.extractors.hooks import LatentExtractor
from peek.utils.paths import repo_path


_ACTIVATION_MODULES = (
    torch.nn.ELU,
    torch.nn.GELU,
    torch.nn.Hardsigmoid,
    torch.nn.Hardswish,
    torch.nn.LeakyReLU,
    torch.nn.Mish,
    torch.nn.PReLU,
    torch.nn.ReLU,
    torch.nn.ReLU6,
    torch.nn.SELU,
    torch.nn.SiLU,
)

_PASSTHROUGH_MODULES = (
    torch.nn.AdaptiveAvgPool2d,
    torch.nn.AvgPool2d,
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.Dropout,
    torch.nn.Dropout2d,
    torch.nn.Dropout3d,
    torch.nn.Identity,
    torch.nn.LayerNorm,
    torch.nn.MaxPool2d,
)

_ACTIVATION_FUNCTIONS = {
    torch.nn.functional.elu,
    torch.nn.functional.gelu,
    torch.nn.functional.hardsigmoid,
    torch.nn.functional.hardswish,
    torch.nn.functional.leaky_relu,
    torch.nn.functional.mish,
    torch.nn.functional.relu,
    torch.nn.functional.relu6,
    torch.nn.functional.selu,
    torch.nn.functional.silu,
}

_PASSTHROUGH_FUNCTIONS = {
    operator.add,
    operator.getitem,
    operator.mul,
    torch.add,
    torch.mul,
}

_PASSTHROUGH_METHODS = {
    "contiguous",
    "flatten",
    "mean",
    "permute",
    "reshape",
    "squeeze",
    "transpose",
    "unsqueeze",
    "view",
}


def _resolve_glob(pattern: str) -> str:
    """
    Resolves an images glob.

    - If absolute: use as-is.
    - If relative: interpret relative to repo root.
    """
    if os.path.isabs(pattern):
        return pattern
    return str((repo_path(".") / pattern).resolve())


def _load_torchvision_model(model_name: str, pretrained: bool = True) -> torch.nn.Module:
    """
    Load a torchvision model by name.
    """
    try:
        if pretrained:
            # Use modern weights API
            # Map common model names to their weights classes
            weights_mapping = {
                'resnet18': 'ResNet18_Weights',
                'resnet34': 'ResNet34_Weights', 
                'resnet50': 'ResNet50_Weights',
                'resnet101': 'ResNet101_Weights',
                'resnet152': 'ResNet152_Weights',
                'vgg11': 'VGG11_Weights',
                'vgg13': 'VGG13_Weights',
                'vgg16': 'VGG16_Weights',
                'vgg19': 'VGG19_Weights',
                'convnext_tiny': 'ConvNeXt_Tiny_Weights',
                'convnext_small': 'ConvNeXt_Small_Weights',
                'convnext_base': 'ConvNeXt_Base_Weights',
                'convnext_large': 'ConvNeXt_Large_Weights',
                'vit_b_16': 'ViT_B_16_Weights',
                'vit_b_32': 'ViT_B_32_Weights',
                'vit_l_16': 'ViT_L_16_Weights',
                'vit_l_32': 'ViT_L_32_Weights',
                'vit_h_14': 'ViT_H_14_Weights',
                'efficientnet_b0': 'EfficientNet_B0_Weights',
                'efficientnet_b1': 'EfficientNet_B1_Weights',
                'efficientnet_b2': 'EfficientNet_B2_Weights',
                'efficientnet_b3': 'EfficientNet_B3_Weights',
                'efficientnet_b4': 'EfficientNet_B4_Weights',
                'efficientnet_b5': 'EfficientNet_B5_Weights',
                'efficientnet_b6': 'EfficientNet_B6_Weights',
                'efficientnet_b7': 'EfficientNet_B7_Weights',
            }
            
            weights_class_name = weights_mapping.get(model_name)
            if weights_class_name:
                weights_class = getattr(torchvision.models, weights_class_name)
                weights = weights_class.DEFAULT
                model = getattr(torchvision.models, model_name)(weights=weights)
            else:
                # Fallback to old API if weights class not found
                model = getattr(torchvision.models, model_name)(pretrained=True)
        else:
            model = getattr(torchvision.models, model_name)(pretrained=False)
    except AttributeError:
        raise ValueError(f"Unknown torchvision model: {model_name}")
    return model


def _preprocess_image(image_path: str, img_size: int = 224) -> torch.Tensor:
    """
    Load and preprocess an image for PyTorch models.
    """
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension


def _is_activation_module(module: torch.nn.Module) -> bool:
    return isinstance(module, _ACTIVATION_MODULES)


def _is_convnext_model(model: torch.nn.Module) -> bool:
    return model.__class__.__name__ == "ConvNeXt"


def _collect_conv_activation_nodes(model: torch.nn.Module) -> List[str]:
    """
    Return feature-extractor node names for activations whose preceding op is conv-related.
    """
    _, eval_names = get_graph_node_names(model)
    modules_by_name = dict(model.named_modules())
    selected: List[str] = []

    def resolve_module_name(node_name: str) -> Optional[str]:
        if node_name in modules_by_name:
            return node_name

        match = re.match(r"^(.*)_\d+$", node_name)
        if match and match.group(1) in modules_by_name:
            return match.group(1)

        return None

    for index, eval_name in enumerate(eval_names):
        module_name = resolve_module_name(eval_name)
        if module_name is None:
            continue
        module = modules_by_name[module_name]
        if not _is_activation_module(module):
            continue
        if index == 0:
            continue
        predecessor_name = eval_names[index - 1]
        predecessor_module_name = resolve_module_name(predecessor_name)
        if predecessor_module_name is None:
            continue
        predecessor_module = modules_by_name[predecessor_module_name]
        if isinstance(predecessor_module, (torch.nn.Conv2d, _PASSTHROUGH_MODULES)):
            selected.append(eval_name)

    return selected


def _collect_convnext_block_names(model: torch.nn.Module) -> List[str]:
    """
    Return ConvNeXt CNBlock module names.

    These block outputs are spatially coherent feature stacks and are a more
    meaningful PEEK target for ConvNeXt than trying to force a generic
    conv-post-activation rule onto its MLP-style block internals.
    """
    return [
        name
        for name, module in model.named_modules()
        if module.__class__.__name__ == "CNBlock"
    ]


def _extract_torch_conv_activation_latents(
    model: torch.nn.Module,
    paths: Sequence[str],
    out_dir_p: Path,
    device: str,
    img_size: int,
    modules: Optional[List[int]],
    fp16: bool,
    to_cpu: bool,
    return_first: bool,
    verbose: bool,
) -> Optional[Dict[int, torch.Tensor]]:
    if _is_convnext_model(model):
        block_names = _collect_convnext_block_names(model)
        if not block_names:
            raise ValueError("No ConvNeXt spatial block outputs were found for this model.")

        selected_indices = list(range(len(block_names))) if modules is None else list(modules)
        modules_by_name = dict(model.named_modules())
        targets = [modules_by_name[block_names[i]] for i in selected_indices]

        handles = []
        first_cache = None

        for p in paths:
            input_tensor = _preprocess_image(p, img_size).to(device)
            cache: Dict[int, torch.Tensor] = {}

            def make_hook(index: int):
                def hook(_module, _inputs, output):
                    y = output.detach()
                    if fp16:
                        y = y.half()
                    if to_cpu:
                        y = y.cpu()
                    cache[index] = y
                return hook

            for index, target in zip(selected_indices, targets):
                handles.append(target.register_forward_hook(make_hook(index)))

            try:
                _ = model(input_tensor)
            finally:
                for handle in handles:
                    handle.remove()
                handles.clear()

            if return_first and first_cache is None:
                first_cache = dict(cache)

            out_pkl = out_dir_p / f"{Path(p).stem}.pkl"
            with open(out_pkl, "wb") as f:
                pickle.dump(cache, f)

            if verbose:
                print(f"[torch] saved: {out_pkl}")

        return first_cache if return_first else None

    node_names = _collect_conv_activation_nodes(model)
    if not node_names:
        raise ValueError("No post-activation conv nodes were found for this model.")

    if modules is None:
        selected = list(enumerate(node_names))
    else:
        selected = [(i, node_names[i]) for i in modules]

    return_nodes = {node_name: str(index) for index, node_name in selected}
    feature_model = create_feature_extractor(model, return_nodes=return_nodes).to(device)
    feature_model.eval()

    first_cache = None

    for p in paths:
        input_tensor = _preprocess_image(p, img_size).to(device)
        outputs = feature_model(input_tensor)

        cache: Dict[int, torch.Tensor] = {}
        for index_str, tensor in outputs.items():
            y = tensor.detach()
            if fp16:
                y = y.half()
            if to_cpu:
                y = y.cpu()
            cache[int(index_str)] = y

        if return_first and first_cache is None:
            first_cache = dict(cache)

        out_pkl = out_dir_p / f"{Path(p).stem}.pkl"
        with open(out_pkl, "wb") as f:
            pickle.dump(cache, f)

        if verbose:
            print(f"[torch] saved: {out_pkl}")

    return first_cache if return_first else None


@torch.no_grad()
def extract_torch_latents(
    model: Union[str, torch.nn.Module],
    images_glob: str,
    out_dir: Union[str, Path],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    img_size: int = 224,
    modules: Optional[List[int]] = None,
    fp16: bool = False,
    to_cpu: bool = True,
    limit: int = 0,
    return_first: bool = False,
    verbose: bool = False,
    capture_mode: str = "all_modules",
) -> Optional[Dict[int, torch.Tensor]]:
    """
    Run PyTorch model inference and save latents collected via forward hooks.

    Args:
        model         Model name (e.g., "resnet50") or torch.nn.Module instance
        images_glob   Glob for input images (repo-relative OK; interpreted from repo root)
        out_dir       Output folder for .pkl files (repo-relative OK)
        device        Device to run on ("cpu", "cuda", etc.)
        img_size      Input image size (square)
        modules       None for all modules, or list of module indices
        fp16          Store captured latents as float16
        to_cpu        Move captured latents to CPU before saving
        limit         Max number of images (0 = all)
        return_first  Return first cache dict for inspection
        verbose       Print progress lines
        capture_mode  "all_modules" or "conv_post_activation"

    Returns:
        None, or first cache dict if return_first=True.
    """
    # Load model
    if isinstance(model, str):
        model = _load_torchvision_model(model)
    elif not isinstance(model, torch.nn.Module):
        raise ValueError("model must be a string (torchvision model name) or torch.nn.Module")

    model = model.to(device)
    model.eval()

    # Resolve IO paths
    out_dir_p = repo_path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    # Collect input images
    g = _resolve_glob(images_glob)
    paths = sorted(glob.glob(g))
    if limit and limit > 0:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No images matched: {images_glob} (resolved: {g})")

    if capture_mode == "conv_post_activation":
        return _extract_torch_conv_activation_latents(
            model=model,
            paths=paths,
            out_dir_p=out_dir_p,
            device=device,
            img_size=img_size,
            modules=modules,
            fp16=fp16,
            to_cpu=to_cpu,
            return_first=return_first,
            verbose=verbose,
        )
    if capture_mode != "all_modules":
        raise ValueError(f"Unsupported capture_mode: {capture_mode}")

    # Attach hooks
    extractor = LatentExtractor(
        model,
        modules=modules,
        to_cpu=to_cpu,
        fp16=fp16,
        target_mode="recursive",
    )
    extractor.start()

    first_cache = None

    for p in paths:
        # Preprocess image
        input_tensor = _preprocess_image(p, img_size).to(device)

        # Forward pass populates extractor.cache
        _ = model(input_tensor)

        if return_first and first_cache is None:
            first_cache = dict(extractor.cache)

        # Save per-image latents
        stem = Path(p).stem
        out_pkl = out_dir_p / f"{stem}.pkl"
        with open(out_pkl, "wb") as f:
            pickle.dump(dict(extractor.cache), f)

        extractor.clear()

        if verbose:
            print(f"[torch] saved: {out_pkl}")

    extractor.stop()
    return first_cache if return_first else None
