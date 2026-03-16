"""
PEEK package public API.
"""

from .analytic import (
    GeometryConfig,
    analyze_dense_head_image,
    build_dense_head_pair,
    build_vgg16_dense_head_pair,
    run_dense_head_analytic_study,
    run_vgg16_dense_head_analytic_study,
)
from .analytic_plots import (
    compare_geometry_csvs,
    plot_dense_head_alignment_maps,
    render_geometry_directory,
)
from .analytic_train import (
    ImagenetteFineTuneConfig,
    launch_imagenette_training_tmux,
    launch_vgg16_dense_head_training_tmux,
    train_imagenette_model,
    train_vgg16_dense_head_imagenette,
)
from .core import PEEK, peek_mean_variance, relative_variance_contribution
from .metrics import compute_feature_folder_metrics
from .models import ConvNeXtBaseOneDenseNoPool, ResNet50OneDenseNoPool, VGG16OneDenseNoPool
from .plotting import plot_PEEK

__all__ = [
    "GeometryConfig",
    "ImagenetteFineTuneConfig",
    "PEEK",
    "ConvNeXtBaseOneDenseNoPool",
    "ResNet50OneDenseNoPool",
    "VGG16OneDenseNoPool",
    "analyze_dense_head_image",
    "build_dense_head_pair",
    "build_vgg16_dense_head_pair",
    "compare_geometry_csvs",
    "compute_feature_folder_metrics",
    "launch_imagenette_training_tmux",
    "launch_vgg16_dense_head_training_tmux",
    "peek_mean_variance",
    "plot_dense_head_alignment_maps",
    "plot_PEEK",
    "relative_variance_contribution",
    "render_geometry_directory",
    "run_dense_head_analytic_study",
    "run_vgg16_dense_head_analytic_study",
    "train_imagenette_model",
    "train_vgg16_dense_head_imagenette",
]
