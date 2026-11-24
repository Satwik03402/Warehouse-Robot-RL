"""Reproduce the transcription of the paper's Figure 5 maps (provenance for the report).

Render page 6 of the CQ-learning paper at 300 dpi, then run this script on it:

    pdftoppm -f 6 -l 6 -r 300 -png Learning_multi-agent_state_space_representations.pdf page
    python3 -m marl.maps.extract_fig5 page-6.png

Each map is drawn on a regular grid. Grid lines between two free cells are thin (1 px);
walls are thick (4-6 px). The script classifies every cell boundary as none / thin / thick
by sampling it at three points, with a majority vote that skips robot icons. Cells that touch
a thin line are free. Thick lines between two free cells are internal walls (TunnelToGoal).
The grid calibration below (origin, pitch and size in pixels) was measured from scanlines of
the rendered figure. Free-cell counts must equal the Indep #states column of Table 1.
Needs Pillow (not in requirements.txt).
"""
import argparse
import numpy as np

# name: (x0, pitch_x, ncols, y0, pitch_y, nrows, Table 1 free cells), in pixels of the cropped figure
CALIBRATION = {
    'TTG': (274.5, 71.1, 5, 36.5, 71.2, 5, 25),
    'TR': (252.5, 57.14, 7, 628.5, 56.8, 7, 36),
    'ISR': (887.0, 55.72, 9, 470.5, 55.75, 10, 43),
    'CIT': (1557.0, 46.54, 14, 469.8, 46.58, 12, 69),
    'CMU': (707.0, 47.48, 34, 59.0, 47.43, 7, 133),
}


def _run_length_near(line, center, radius=6):
    seg = line[max(0, center - radius):center + radius + 1]
    best, i = 0, 0
    while i < len(seg):
        if seg[i]:
            j = i
            while j < len(seg) and seg[j]:
                j += 1
            best, i = max(best, j - i), j
        else:
            i += 1
    return best


def _classify(dark, samples):
    votes = ['none' if n == 0 else 'thin' if n <= 3 else 'thick'
             for n in (_run_length_near(line, pos) for line, pos in samples)]
    return max(set(votes), key=votes.count)


def extract(dark, x0, px, nc, y0, py, nr):
    """Return (grid rows top-to-bottom, internal walls as ((r, c), (r, c)) pairs)."""
    vert, horz = {}, {}
    for r in range(nr):
        for c in range(-1, nc):
            x = round(x0 + (c + 1) * px)
            vert[(r, c)] = _classify(dark, [(dark[round(y0 + (r + f) * py)], x) for f in (0.3, 0.5, 0.7)])
    for r in range(-1, nr):
        for c in range(nc):
            y = round(y0 + (r + 1) * py)
            horz[(r, c)] = _classify(dark, [(dark[:, round(x0 + (c + f) * px)], y) for f in (0.3, 0.5, 0.7)])
    free = set()
    for (r, c), kind in vert.items():
        if kind == 'thin':
            free |= {(r, c), (r, c + 1)}
    for (r, c), kind in horz.items():
        if kind == 'thin':
            free |= {(r, c), (r + 1, c)}
    walls = [((r, c), (r, c + 1)) for (r, c), k in vert.items()
             if k == 'thick' and {(r, c), (r, c + 1)} <= free]
    walls += [((r, c), (r + 1, c)) for (r, c), k in horz.items()
              if k == 'thick' and {(r, c), (r + 1, c)} <= free]
    grid = [''.join('.' if (r, c) in free else '#' for c in range(nc)) for r in range(nr)]
    return grid, walls


def main():
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument('page_png', help='page 6 rendered at 300 dpi')
    args = ap.parse_args()
    page = Image.open(args.page_png).convert('L')
    w, h = page.size
    dark = np.asarray(page.crop((0, int(h * 0.06), w, int(h * 0.43)))) < 128
    for name, (x0, px, nc, y0, py, nr, expected) in CALIBRATION.items():
        grid, walls = extract(dark, x0, px, nc, y0, py, nr)
        n = sum(row.count('.') for row in grid)
        print(f'== {name}: free={n} expected={expected} {"OK" if n == expected else "MISMATCH"} walls={len(walls)}')
        print('\n'.join(grid))
        if walls:
            print('walls (row, col):', walls)


if __name__ == '__main__':
    main()
