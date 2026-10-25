import argparse
import csv
import logging
from pathlib import Path
from typing import Any
from typing import Optional

from birder.common import cli

from birder_clip.data.datasets.csv import IMAGE_PATH_COLUMN

logger = logging.getLogger(__name__)


def _remap_path(path: str, from_prefix: str, to_prefix: str) -> Optional[str]:
    if path == from_prefix:
        return to_prefix

    prefix = from_prefix.rstrip("/") + "/"
    if path.startswith(prefix) is False:
        return None

    return to_prefix.rstrip("/") + "/" + path[len(prefix) :]


def remap_csv_paths(args: argparse.Namespace) -> None:
    data_path = Path(args.data_path).expanduser()
    output_path = Path(args.output).expanduser()
    if output_path.exists() is True and args.force is False:
        logger.warning(f"File {output_path} already exists... aborting")
        raise SystemExit(1)
    if output_path.exists() is True and data_path.samefile(output_path) is True:
        raise cli.ValidationError("Input and output CSV files must be different")

    num_rows = 0
    num_matched = 0
    with open(data_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader, None)
        if header is None or IMAGE_PATH_COLUMN not in header:
            raise cli.ValidationError(f"CSV must contain an '{IMAGE_PATH_COLUMN}' column")

        path_index = header.index(IMAGE_PATH_COLUMN)
        logger.info(f"Saving CSV file at {output_path}...")
        try:
            with open(output_path, "w", encoding="utf-8", newline="") as output_handle:
                writer = csv.writer(output_handle)
                writer.writerow(header)
                for num_rows, row in enumerate(reader, start=1):
                    if len(row) != len(header):
                        raise cli.ValidationError(f"CSV row {num_rows} has {len(row)} fields, expected {len(header)}")

                    new_path = _remap_path(row[path_index], args.from_prefix, args.to_prefix)
                    if new_path is not None:
                        row[path_index] = new_path
                        num_matched += 1

                    writer.writerow(row)

            if num_matched == 0:
                raise cli.ValidationError(f"No image paths match prefix '{args.from_prefix}'")

        except (Exception, KeyboardInterrupt):
            logger.error(
                f"Failed to remap CSV file to {output_path}, any partial output has been left in place for inspection"
            )
            raise

    logger.info(f"Remapped {num_matched:,} of {num_rows:,} rows ({num_rows - num_matched:,} unmatched)")


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "remap-csv-paths",
        allow_abbrev=False,
        help="remap image path prefixes in an image-text CSV dataset",
        description="remap image path prefixes in an image-text CSV dataset",
        epilog=(
            "Usage examples:\n"
            "python -m birder_clip.tools remap-csv-paths --from-prefix ../path/data/ --to-prefix /mnt/data/ "
            "--output captions_local.csv captions.csv\n"
            "python -m birder_clip.tools remap-csv-paths --from-prefix ../path/data/ --to-prefix /mnt/data/ "
            "--output captions_local.csv --force captions.csv\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument("--from-prefix", type=str, required=True, help="directory prefix to replace")
    subparser.add_argument(
        "--to-prefix",
        type=str,
        required=True,
        help="replacement directory prefix (relative paths are resolved from the output CSV directory)",
    )
    subparser.add_argument("--output", type=str, required=True, metavar="FILE", help="destination CSV file path")
    subparser.add_argument("-f", "--force", action="store_true", help="override existing CSV file")
    subparser.add_argument("data_path", help="image-text CSV file path")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    if len(args.from_prefix) == 0:
        raise cli.ValidationError("--from-prefix must not be empty")
    if len(args.to_prefix) == 0:
        raise cli.ValidationError("--to-prefix must not be empty")

    args.from_prefix = args.from_prefix.rstrip("/") or "/"
    args.to_prefix = args.to_prefix.rstrip("/") or "/"

    remap_csv_paths(args)
