"""Matplotlib set-up shared by every figure.

Matplotlib >= 3.10 shapes right-to-left Arabic-script text itself, so Urdu
strings are passed through unchanged. DejaVu Sans covers Latin text and
Noto Naskh Arabic is the fallback font for Urdu glyphs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:          # scripts/tests: no display needed
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

plt.rcParams.update({
    "font.family": ["DejaVu Sans", "Noto Naskh Arabic"],
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b",
           "#e377c2", "#17becf", "#bcbd22", "#7f7f7f", "#393b79", "#637939"]


def check_labels(fig) -> None:
    """The brief: every figure needs axis labels and a title."""
    sup = fig._suptitle.get_text() if fig._suptitle is not None else ""
    for ax in fig.axes:
        if not (ax.get_title() or sup):
            raise ValueError("figure without a title")
        if not ax.get_xlabel() or not ax.get_ylabel():
            raise ValueError(f"axes '{ax.get_title()}' is missing an axis label")


def savefig(fig, name: str) -> Path:
    check_labels(fig)
    path = C.FIG / name
    fig.savefig(path)
    return path
