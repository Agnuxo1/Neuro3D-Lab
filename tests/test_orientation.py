"""Regression test for the image/physical orientation convention.

+x is the column index increasing; +y points up, i.e. row 0 (top of the image)
has the largest y. A region at (cx=0, cy=+c) must light the top rows.
Found after the first release: detector_masks had +y pointing down (fe de erratas
in docs/CONTRACTS/CONTRATO-001-nucleo.md).
"""
import numpy as np

from optlab.elements import detector_masks


def _rows_cols(mask):
    rows, cols = np.nonzero(mask)
    return rows, cols


def test_positive_y_region_lights_top_rows():
    n, pixel = 32, 1e-6
    m = detector_masks(n, pixel, [{"cx_m": 0.0, "cy_m": 8e-6, "w_m": 2e-6, "h_m": 2e-6}])[0]
    rows, cols = _rows_cols(m)
    assert rows.size > 0
    assert rows.max() < n // 2
    assert abs(cols.mean() - (n - 1) / 2) < 1.0


def test_positive_x_region_lights_right_columns():
    n, pixel = 32, 1e-6
    m = detector_masks(n, pixel, [{"cx_m": 8e-6, "cy_m": 0.0, "w_m": 2e-6, "h_m": 2e-6}])[0]
    rows, cols = _rows_cols(m)
    assert cols.size > 0
    assert cols.min() > n // 2
    assert abs(rows.mean() - (n - 1) / 2) < 1.0
