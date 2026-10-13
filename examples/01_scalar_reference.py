"""Example 1: propagate a smooth Gaussian with the finite-support core and compare with direct RS.

Run: python examples/01_scalar_reference.py [--n 16]
Units: metres. The comparison uses point values; nothing is fitted or renormalised.
"""
import argparse

from neuro3d_completion.validation import scalar_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=16, help="physical pixels per side (even)")
    args = parser.parse_args()
    result = scalar_reference(n=args.n)
    for key in ("quadrature_relative_L2", "source_window_relative_L2", "ASM_vs_RS_complex_relative_L2"):
        print(f"{key:32s} {result[key]:.3e}")
    print("scope:", result["reference_scope"])


if __name__ == "__main__":
    main()
