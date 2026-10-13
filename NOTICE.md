# Third-party software and models

Human Generator is Apache-2.0 (see `LICENSE`). It ships **no third-party model weights or assets**: `install.ps1`
installs the libraries below from their official sources, and they download their own files on first use.
Each keeps its own licence; read them before commercial use.

| Component | Used for | Licence | Source |
|---|---|---|---|
| Kimodo (code) | text-to-motion | Apache-2.0 | https://github.com/nv-tlabs/kimodo |
| Kimodo-SOMA-RP-v1 (model weights) | text-to-motion | NVIDIA Open Model License | https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1 |
| Meta Llama 3 8B Instruct | Kimodo's text encoder (gated: request access) | Meta Llama 3 Community License | https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct |
| SOMA-X (`py-soma-x`) and its assets | body model, skeleton, skinning | Apache-2.0 (assets: see the model card) | https://github.com/NVlabs/SOMA-X · https://huggingface.co/nvidia/SOMA-X |
| Anny | anthropometric body (age, height, weight...) | Apache-2.0 (NAVER) | https://github.com/naver/anny |
| PyTorch | GPU runtime | BSD-3-Clause | https://pytorch.org |

"Built with Meta Llama 3": Kimodo's text encoder is based on Meta Llama 3. Motions you generate are subject to the
NVIDIA Open Model License and the Meta Llama 3 Community License terms that apply to model outputs.
