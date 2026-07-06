"""Crop the extra canvas around the SPHEREPACK-style README figure.

Matplotlib's ``bbox_inches="tight"`` does not always remove the blank area
around 3-D axes after the axes are hidden.  This script trims the final PNG by
finding the bounding box of pixels that differ from the background color.

Run from the repository root:

    python scripts/crop_spherepack_docs_figure.py

By default, this overwrites ``readme/spherepack_docs_style.png`` in place.
Use ``--output`` to keep the original file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def find_content_bbox(
    image: Image.Image,
    *,
    tolerance: int = 8,
    background: tuple[int, int, int] | None = None,
) -> tuple[int, int, int, int]:
    """Return the bounding box of visible pixels different from background.

    If the image has transparent margins, alpha is used.  Otherwise, pixels are
    compared against ``background``.  When ``background`` is omitted, the script
    samples the top-left pixel, which matches the blank canvas for this figure.
    """

    rgba = image.convert("RGBA")
    arr = np.asarray(rgba)
    rgb = arr[:, :, :3].astype(np.int16)
    alpha = arr[:, :, 3]

    if np.any(alpha == 0):
        mask = alpha > tolerance
    else:
        if background is None:
            background_array = rgb[0, 0]
        else:
            background_array = np.asarray(background, dtype=np.int16)
        mask = np.abs(rgb - background_array).max(axis=2) > tolerance

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Could not find any content pixels to crop around.")

    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def pad_bbox(
    bbox: tuple[int, int, int, int],
    *,
    padding: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Expand a bounding box by ``padding`` pixels without leaving the image."""

    left, top, right, bottom = bbox
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(width, right + padding),
        min(height, bottom + padding),
    )


def crop_image(
    input_path: Path,
    output_path: Path,
    *,
    tolerance: int = 8,
    padding: int = 20,
) -> tuple[int, int, int, int]:
    """Crop ``input_path`` and write the result to ``output_path``."""

    image = Image.open(input_path)
    bbox = find_content_bbox(image, tolerance=tolerance)
    padded_bbox = pad_bbox(
        bbox,
        padding=padding,
        width=image.width,
        height=image.height,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(padded_bbox).save(output_path)
    return padded_bbox


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("readme/spherepack_docs_style.png"),
        help="PNG path to crop.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("readme/spherepack_docs_style_cropped.png"),
        help="PNG path to write. Defaults to overwriting --input.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=20,
        help="Pixels of whitespace to keep around the detected content.",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=8,
        help="Maximum RGB/alpha difference treated as background.",
    )
    args = parser.parse_args()

    output = args.output or args.input
    bbox = crop_image(
        args.input,
        output,
        tolerance=args.tolerance,
        padding=args.padding,
    )
    print(f"cropped {args.input} -> {output}")
    print(f"bbox with padding: {bbox}")


if __name__ == "__main__":
    main()
