import argparse
import fnmatch
from typing import Any

from birder.common import cli
from birder.model_registry.model_registry import group_sort
from rich.columns import Columns
from rich.console import Console
from rich.table import Table

from birder_clip.model_registry import Task
from birder_clip.model_registry import registry


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "list-models",
        allow_abbrev=False,
        help="list available models",
        description="list available models",
        epilog=(
            "Usage examples:\n"
            "python -m birder_clip.tools list-models\n"
            "python -m birder_clip.tools list-models --image-text\n"
            "python -m birder_clip.tools list-models --text\n"
            "python -m birder_clip.tools list-models --pretrained\n"
            "python -m birder_clip.tools list-models --pretrained --image-text --verbose\n"
            "python -m birder_clip.tools list-models --pretrained --verbose --filter '*clip*'\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )

    task_group = subparser.add_mutually_exclusive_group(required=False)
    task_group.add_argument("--image-text", default=False, action="store_true", help="list image-text models")
    task_group.add_argument("--text", default=False, action="store_true", help="list text models")

    subparser.add_argument("--pretrained", default=False, action="store_true", help="list pretrained models")
    subparser.add_argument("--filter", type=str, help="filter results with a fnmatch type filter)")
    subparser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="enable verbose output with additional model details",
    )
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    # Determine the task based on the selected flags
    task = None
    if args.image_text is True:
        task = Task.IMAGE_TEXT
    elif args.text is True:
        task = Task.TEXT

    if args.pretrained is True:
        model_list = registry.list_pretrained_models(task=task)
    else:
        model_list = registry.list_models(task=task)

    model_list = group_sort(model_list)
    if args.filter is not None:
        model_list = fnmatch.filter(model_list, args.filter)

    console = Console()
    if args.verbose is True:
        if args.pretrained is True:
            table = Table(show_header=True, header_style="bold dark_magenta")
            table.add_column("Model name")
            table.add_column("Format", style="dim")
            table.add_column("File size", justify="right")
            table.add_column("Resolution", justify="right")
            table.add_column("Context", justify="right")
            table.add_column("Description")
            for model_name in model_list:
                model_metadata = registry.get_pretrained_metadata(model_name)
                for format_name, format_info in model_metadata["formats"].items():
                    table.add_row(
                        model_name,
                        format_name,
                        f"{format_info['file_size']}MB",
                        "x".join(str(x) for x in model_metadata["resolution"]),
                        str(model_metadata["context_length"]),
                        model_metadata["description"],
                    )

            console.print(table)

        else:
            raise NotImplementedError

    else:
        console.print(
            Columns(
                model_list,
                padding=(0, 3),
                equal=True,
                column_first=True,
                title=f"[bold]{len(model_list)} Models[/bold]",
            )
        )
