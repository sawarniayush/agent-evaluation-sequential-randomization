"""Reproduce the paper's analyses from the included numerical data."""

import argparse
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .analysis import ROOT, STUDIES, fit, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("reproduce", "figures", "verify-data"))
    parser.add_argument("--study", choices=("all", *STUDIES, "retail_mc"), default="all")
    parser.add_argument(
        "--refit-nuisance",
        action="store_true",
        help="Refit and check the five-fold outcome regressions as well.",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    if args.command == "verify-data":
        from .integrity import verify_data

        verify_data()
        return
    if args.command == "figures":
        from .figures import render

        render(args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    selected = (
        STUDIES if args.study == "all" else () if args.study == "retail_mc" else (args.study,)
    )
    with threadpool_limits(limits=2):
        for study in selected:
            print(f"Fitting {study}...", flush=True)
            frame, contributions = fit(study, args.refit_nuisance)
            count = verify(study, frame)
            frame.to_csv(args.output / f"{study}.csv", index=False)
            np.savez_compressed(args.output / f"{study}_contributions.npz", **contributions)
            print(
                f"Verified {count} result rows, including intervals and support: {study}",
                flush=True,
            )
        if args.study in ("all", "retail_mc"):
            from .monte_carlo import reproduce

            reproduce(args.output)


if __name__ == "__main__":
    main()
