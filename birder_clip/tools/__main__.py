import argparse

from birder.common import cli

from birder_clip.tools import convert_model
from birder_clip.tools import download_tokenizer
from birder_clip.tools import list_models
from birder_clip.tools import model_info
from birder_clip.tools import show_iterator
from birder_clip.tools import stats


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m birder_clip.tools",
        allow_abbrev=False,
        description="Tool to run Birder CLIP auxiliary commands",
        epilog=(
            "Usage examples:\n"
            "python -m birder_clip.tools download-tokenizer --tokenizer timm/ViT-SO400M-14-SigLIP2\n"
            "python -m birder_clip.tools convert-model --network openai_clip_vit_l14 --resize 336\n"
            "python -m birder_clip.tools list-models --pretrained\n"
            "python -m birder_clip.tools model-info --network openai_clip_vit_l14 --epoch 0\n"
            "python -m birder_clip.tools show-iterator --mode training --aug-level 1 --data-path data/training.csv\n"
            "python -m birder_clip.tools stats --prompt-tokens --tokenizer openai_clip_bpe "
            "--data-path data/training.csv\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    convert_model.set_parser(subparsers)
    download_tokenizer.set_parser(subparsers)
    list_models.set_parser(subparsers)
    model_info.set_parser(subparsers)
    show_iterator.set_parser(subparsers)
    stats.set_parser(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
