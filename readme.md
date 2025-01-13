# PEEK

This is the official GitHub repository for Probabilistic Explanations for Entropic Knowledge extraction (PEEK), a method for visualizing CNN decisionmaking processes.

The implementation currently works for YOLOv5 and arbitrary Keras CNNs. More will be added in the future. Requests for implementations for specific architectures should go to the owner of the repo and lead developer, Mackenzie Meni.

Currently, we provide a modified YOLOv5 implementation with the repo as well as a notebook demonstrating use of PEEK with YOLOv5.

UPDATES:

* Jan 2025: New notebook with a demo of PEEK used with a VGG16 image classifier pretrained on ImageNet (implemented in PyTorch), added sample images locally.

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