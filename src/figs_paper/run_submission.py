"""Re-render Figures 4--11 from data as PDF, PNG and compressed TIFF.

Run: python -m src.figs_paper.run_submission --out-dir outputs/paper_submission
The original plotting/data-selection code is reused without retraining.
Line figures: >=1000 pixels/inch at 7.48 inches wide.
Field/line combinations: >=600 pixels/inch at 7.48 inches wide.
Raster dimensions come from fresh Matplotlib rendering, never image enlargement.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from PIL import Image

from src.figs_paper import run_all_final


@contextmanager
def publication_exports(width_inches: float):
    original_savefig = Figure.savefig
    records = []

    def export(fig, filename, *args, **kwargs):
        path = Path(filename)
        match = re.match(r'fig(\d+)', path.name)
        if not match or path.suffix.lower() != '.png':
            return original_savefig(fig, filename, *args, **kwargs)
        number = int(match.group(1))
        target_dpi = 1000 if number in (4, 5, 7, 10) else 600
        target_pixels = math.ceil(width_inches * target_dpi)
        fig.canvas.draw()
        bounds = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.1)
        # A fixed bounding box gives PDF and raster exports identical geometry.
        bounds = Bbox.from_bounds(*bounds.bounds)
        render_dpi = math.ceil((target_pixels + 4) / bounds.width)
        options = dict(kwargs)
        options.pop('dpi', None)
        options.pop('bbox_inches', None)
        options.update(bbox_inches=bounds, facecolor='white')
        original_savefig(fig, path, dpi=render_dpi, **options)
        original_savefig(fig, path.with_suffix('.pdf'), dpi=render_dpi, **options)
        with Image.open(path) as image:
            rgb = image.convert('RGB')
            # The print-size metadata describes the actual rendered pixels;
            # no resampling occurs when writing either lossless raster file.
            print_dpi = rgb.width / width_inches
            rgb.save(path, dpi=(print_dpi, print_dpi), optimize=True)
            rgb.save(path.with_suffix('.tif'), compression='tiff_lzw',
                     dpi=(print_dpi, print_dpi))
            assert rgb.width >= target_pixels
            record = dict(figure=number, file=path.name, pixels=list(rgb.size),
                          reference_width_inches=width_inches,
                          target_dpi=target_dpi, effective_dpi=print_dpi,
                          render_dpi=render_dpi, data_replotted=True)
        records.append(record)
        print('[Publication export]', json.dumps(record), flush=True)

    Figure.savefig = export
    try:
        yield records
    finally:
        Figure.savefig = original_savefig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default='outputs/paper_submission')
    parser.add_argument('--width-inches', type=float, default=7.48)
    args = parser.parse_args()
    if args.width_inches <= 0:
        parser.error('--width-inches must be positive')
    with publication_exports(args.width_inches) as records:
        output = run_all_final.main(args.out_dir)
    assert sorted(r['figure'] for r in records) == list(range(4, 12))
    (output / 'export_manifest.json').write_text(
        json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
