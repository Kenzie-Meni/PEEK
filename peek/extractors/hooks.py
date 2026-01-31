"""
Forward-hook latent capture.

This module captures intermediate activations ("latents after a module")
by registering forward hooks on a PyTorch model.

Design intent:
- For YOLOv5, latents correspond to model.model[i] blocks (top-level stages).
- For generic models, latents correspond to top-level children.

Saved output per forward pass:
    {module_index: torch.Tensor}
"""

from __future__ import annotations

from pathlib import Path
import pickle
import torch


class LatentExtractor:
    """
    Collect intermediate activations using forward hooks.

    Typical usage:
        extractor = LatentExtractor(model)
        extractor.start()
        _ = model(x)
        extractor.save("out.pkl")
        extractor.clear()
        extractor.stop()

    Notes:
        - Hooks capture outputs *after* each module executes.
        - Captured tensors are detached and optionally moved to CPU / fp16.
        - One extractor instance corresponds to one model.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        modules=None,
        to_cpu: bool = True,
        fp16: bool = False,
    ):
        # Model to instrument
        self.model = model

        # Optional subset of module indices to hook
        self.modules = modules

        # Post-processing options for captured tensors
        self.to_cpu = to_cpu
        self.fp16 = fp16

        # Storage for captured activations
        self.cache = {}

        # Active hook handles (needed for cleanup)
        self.handles = []

        # Resolve which submodules are considered "top-level"
        self.targets = self._resolve_targets()

    def _resolve_targets(self):
        """
        Determine which modules should be indexed and hooked.

        YOLOv5 convention:
            model.model[i] corresponds to semantic network stages.

        Fallback:
            Use top-level children of the model.
        """
        if (
            hasattr(self.model, "model")
            and isinstance(
                self.model.model,
                (list, torch.nn.ModuleList, torch.nn.Sequential),
            )
        ):
            return list(self.model.model)

        # Generic PyTorch model fallback
        return list(self.model.children())

    def start(self):
        """
        Register forward hooks on selected modules.

        If modules=None, hooks all resolved targets.
        Otherwise, hooks only the specified indices.
        """
        if self.modules is None:
            indices = list(range(len(self.targets)))
        else:
            indices = list(self.modules)

        for i in indices:
            handle = self.targets[i].register_forward_hook(
                self._make_hook(i)
            )
            self.handles.append(handle)

    def _make_hook(self, index: int):
        """
        Create a forward hook function for a specific module index.

        The hook stores the module output in self.cache[index].
        """

        def hook(module, inputs, output):
            # Some modules return tuples (e.g., detection heads)
            y = output[0] if isinstance(output, (list, tuple)) else output

            # Only tensors are meaningful latents
            if not torch.is_tensor(y):
                return

            # Detach from autograd graph
            y = y.detach()

            # Optional precision / device normalization
            if self.fp16:
                y = y.half()
            if self.to_cpu:
                y = y.cpu()

            self.cache[index] = y

        return hook

    def clear(self):
        """
        Clear stored activations without removing hooks.
        """
        self.cache.clear()

    def stop(self):
        """
        Remove all registered hooks from the model.
        """
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def save(self, path: str | Path):
        """
        Serialize the current cache to disk as a pickle.

        The file contains:
            {module_index: torch.Tensor}
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self.cache, f)
