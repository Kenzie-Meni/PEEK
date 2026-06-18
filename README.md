# PEEK

This is the official GitHub repository for Probabilistic Explanations for Entropic Knowledge extraction (PEEK), a method for visualizing CNN decisionmaking processes.

The implementation currently works for YOLO (v5, v8, v11, 26). More will be added in the future. Requests for implementations for specific architectures should go to the owner of the repo and lead developer, Mackenzie Meni.

Use of the PEEK method should cite the original paper:

>M. Meni, T. Mahendrakar, O. D. Raney, R. T. White, M. L. Mayo, and K. R. Pilkiewicz (2024). Taking a PEEK into YOLOv5 for Satellite Component Recognition via Entropy-based Visual Explanations. *AIAA SCITECH 2024 Forum*. https://arc.aiaa.org/doi/abs/10.2514/6.2024-2766

Bibtex:

    @inbook{doi:10.2514/6.2024-2766,
    author = {Mackenzie Meni and Trupti Mahendrakar and Olivia D. Raney and Ryan T. White and Michael L. Mayo and Kevin R. Pilkiewicz},
    title = {Taking a PEEK into YOLOv5 for Satellite Component Recognition via Entropy-based Visual Explanations},
    booktitle = {AIAA SCITECH 2024 Forum},
    doi = {10.2514/6.2024-2766},
    URL = {https://arc.aiaa.org/doi/abs/10.2514/6.2024-2766},
    eprint = {https://arc.aiaa.org/doi/pdf/10.2514/6.2024-2766},
        abstract = { The escalating risk of collisions and the accumulation of space debris in Low Earth Orbit (LEO) has reached critical concern due to the ever increasing number of spacecraft. Addressing this crisis, especially in dealing with non-cooperative and unidentified space debris, is of paramount importance. This paper contributes to efforts in enabling autonomous swarms of small chaser satellites for target geometry determination and safe flight trajectory planning for proximity operations in LEO. Our research explores on-orbit use of the You Only Look Once v5 (YOLOv5) object detection model trained to detect satellite components. While this model has shown promise, its inherent lack of interpretability hinders human understanding, a critical aspect of validating algorithms for use in safety-critical missions. To analyze the decision processes, we introduce Probabilistic Explanations for Entropic Knowledge extraction (PEEK), a method that utilizes information theoretic analysis of the latent representations within the hidden layers of the model. Through both synthetic in hardware-in-the-loop experiments, PEEK illuminates the decision-making processes of the model, helping identify its strengths, limitations and biases. }
    }

## Third-party models

This repository uses external model implementations as Git submodules to avoid vendoring large upstream codebases.

- YOLOv5: `third_party/yolov5` (Ultralytics, v7.0)
- Ultralytics: `third_party/ultralytics` (for YOLOv8/11/26)

All modifications for latent extraction and PEEK are implemented via
PyTorch forward hooks (see `peek/`), not by modifying upstream code.

## PEEK-assisted tracking

The hook refactor also supports an online tracking path for YOLO26-style
Ultralytics models. `YOLOPEEKTracker` runs YOLO inference, captures selected
latent modules with hooks, proposes PEEK recovery regions, and associates both
YOLO detections and PEEK regions into bbox/mask tracks.

```python
from peek.tracking import YOLOPEEKTracker, draw_tracks

with YOLOPEEKTracker("weights/yolo26s.pt", peek_modules=[16, 19, 22]) as tracker:
    for frame_result in tracker.track_video("input.mp4"):
        # frame_result.tracks contains bbox tracks; masks are present for
        # segmentation-capable YOLO weights.
        print(frame_result.frame_index, len(frame_result.tracks), frame_result.latency_ms)
```

PEEK recovery tracks are marked with `source="peek"` so experiments can compare
plain YOLO tracking against PEEK-assisted track continuity.

For the current YOLO26 bbox model, `weights/yolo26s_peek_bbox_best.pt` points to
the best 640px checkpoint from the mAP sweep. A runnable tracker CLI writes an
annotated video and JSONL track records:

```bash
python tools/track_yolo26_peek.py \
  --source /path/to/video_or_frame_directory \
  --weights weights/yolo26s_peek_bbox_best.pt \
  --output runs/track/peek_yolo26_tracking.mp4 \
  --jsonl runs/track/peek_yolo26_tracking.jsonl \
  --device 0
```
