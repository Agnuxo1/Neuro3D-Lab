"""Example 2: window and tied-subpixel convergence of fixed random phase masks, with observed order.

Run: python examples/02_convergence_study.py
The phases are random (seed 0), not trained. Only the numerical window (padding) or the
quadrature (refinement) changes; the 3 x 128 x 128 physical phases never change.
"""
from neuro3d_completion.convergence import observed_order_from_differences
from neuro3d_completion.core import Optics
from neuro3d_completion.validation import boundary_convergence, pixel_convergence


def main():
    base = Optics(n=128, padding=2, refinement=1, boundary="aperture")
    window = boundary_convergence(base, paddings=(2, 4, 8))
    quadrature = pixel_convergence(base, refinements=(1, 2, 4))
    for name, record in (("window", window), ("refinement", quadrature)):
        for key in (k for k in record if "_vs_" in k):
            print(name, key, {m: round(v, 5) for m, v in record[key].items()})
    a = quadrature["refinement_1_vs_2"]["scores_relative_L2"]
    b = quadrature["refinement_2_vs_4"]["scores_relative_L2"]
    print("observed order of the scores (1-2-4):", observed_order_from_differences(a, b, 2))


if __name__ == "__main__":
    main()
