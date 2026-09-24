"""Portable math-text typography shared by the reproduction figures."""

FIGURE_STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "mathtext.fontset": "dejavuserif",
    "text.usetex": False,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
}

TIME_LABEL = r"Time (s)"
SAMPLE_SD_LABEL = r"Mean $\pm$ one sample SD"
SQUARED_ERROR_LABEL = r"Squared estimation error $(z-\hat z)^2$"


def panel_titles(algorithm1_source):
    """Use article-like panel names while marking the published reference."""
    algorithm = ("Algorithm 1" if algorithm1_source == "computed"
                 else "Algorithm 1 (published reference)")
    return "(a) Theorem 1", f"(b) {algorithm}"
