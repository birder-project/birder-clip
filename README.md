# Birder CLIP

Birder CLIP is an early-stage Birder extension for CLIP-style image-text models, focused on practical inference and fine-tuning workflows.

- [Introduction](#introduction)
- [Setup](#setup)
- [Getting Started](#getting-started)
- [Training](#training)
- [Project Status and Contributions](#project-status-and-contributions)
- [Licenses](#licenses)
- [Acknowledgments](#acknowledgments)

## Introduction

Birder CLIP extends [Birder](https://gitlab.com/birder/birder) with image-text models for zero-shot classification, image-text retrieval style workflows, caption generation and related multimodal computer vision tasks.

The project is aimed at image-text modeling rather than general vision-language model (VLM) chat or instruction-following systems.
It currently includes CLIP-style components, tokenizers, model registry utilities, inference scripts and training code.
Full training is supported, but for large-scale CLIP pretraining you are probably better served by [OpenCLIP](https://github.com/mlfoundations/open_clip).

## Setup

1. Ensure your environment meets the minimum requirements:
   - Python 3.11 or newer
   - PyTorch 2.10 or newer (installed for your hardware/driver stack)
   - Birder 0.7.0 or newer

1. Install the latest Birder CLIP version:

```sh
pip install birder-clip
```

## Getting Started

List available image-text models:

```sh
python -m birder_clip.tools list-models --image-text
```

List available pretrained weights:

```sh
python -m birder_clip.tools list-models --pretrained --verbose
```

Run zero-shot classification on a directory of images:

```sh
python -m birder_clip.scripts.zero_shot -n laion_clip_vit_l14 --classes eagle hawk falcon --template-set default --gpu data/images
```

For detailed options, run:

```sh
python -m birder_clip.scripts.zero_shot --help
python -m birder_clip.tools --help
```

## Training

Birder CLIP includes training support for image-text datasets in CSV and WebDataset formats, including CLIP, CoCa and LiT-style workflows.

## Project Status and Contributions

Birder CLIP is an early alpha project. APIs, model names, checkpoints, training recipes and command-line options may change without notice.

This is currently a personal project in active development. Suggestions, bug reports and feedback are welcome through the project's issue tracker, but the project is not yet stable enough for broad external contributions.

## Licenses

The code in this project is primarily licensed under Apache 2.0. See [LICENSE](LICENSE) for details.

Some model implementations, pretrained weights, tokenizers and converted artifacts may be derived from or depend on projects and datasets with their own licenses and usage restrictions.

**You are responsible for ensuring compliance with all licenses and conditions of any dependent licenses.**

### Disclaimer

If you intend to use Birder CLIP, its pretrained weights, or any associated datasets in a commercial product, we strongly recommend seeking legal advice to ensure compliance with all relevant licenses and terms of use.

It's the user's responsibility to ensure that their use of this project, including any pretrained weights or datasets, complies with all applicable licenses and legal requirements.

## Acknowledgments

Birder CLIP owes much to the work of others in computer vision, image-text representation learning and open-source machine learning.

Special thanks to the [OpenCLIP](https://github.com/mlfoundations/open_clip) project, which serves as the main reference implementation and inspiration for much of the CLIP-style modeling and training work here. The same principle as in Birder applies: this project stands on the shoulders of many open-source projects, papers and datasets. If an attribution is missing, please open an issue.
