import argparse

from birder.common import cli

from birder_clip.tools import download_tokenizer
from birder_clip.tools import show_iterator


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m birder_clip.tools",
        allow_abbrev=False,
        description="Tool to run Birder CLIP auxiliary commands",
        epilog=(
            "Usage examples:\n"
            "python -m birder_clip.tools download-tokenizer --tokenizer timm/ViT-SO400M-14-SigLIP2\n"
            "python -m birder_clip.tools show-iterator --mode training --aug-level 1 --data-path data/training.csv\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    download_tokenizer.set_parser(subparsers)
    show_iterator.set_parser(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
