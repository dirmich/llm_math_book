"""llm_math.viz: visualization helpers.

Common plotting helpers for vectors, matrices, gradients, and learning curves.
"""

from __future__ import annotations

from typing import Sequence, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import os
_FONT_OK = False
try:
    fallback_paths = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    for path in fallback_paths:
        if os.path.exists(path):
            try:
                fm.fontManager.addfont(path)
            except Exception:
                pass
    
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    _FONT_OK = True
except Exception:  # pragma: no cover
    _FONT_OK = False


def plot_vector_2d(
    ax: plt.Axes,
    v: Sequence[float],
    origin: Sequence[float] = (0, 0),
    color: str = 'blue',
    label: Optional[str] = None,
    annotate: bool = True,
) -> None:
    """Draw a 2D vector as an arrow."""
    v = np.asarray(v, dtype=float)
    origin = np.asarray(origin, dtype=float)
    ax.quiver(
        origin[0], origin[1], v[0], v[1],
        angles='xy', scale_units='xy', scale=1,
        color=color, label=label,
    )
    if annotate:
        ax.annotate(
            f'({v[0]:.1f}, {v[1]:.1f})',
            xy=(origin[0] + v[0], origin[1] + v[1]),
            xytext=(5, 5), textcoords='offset points',
            color=color,
        )


def setup_axes_2d(ax: plt.Axes, xlim=(-5, 5), ylim=(-5, 5)) -> None:
    """Configure 2D axes, aspect ratio, ticks, and grid."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(0, color='black', linewidth=0.8)


def plot_heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    title: str = '',
    cmap: str = 'viridis',
    annotate: bool = False,
    fmt: str = '.2f',
) -> None:
    """Draw a heatmap for a 2D matrix or attention weights."""
    im = ax.imshow(data, cmap=cmap, aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        ax.set_title(title)
    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, format(data[i, j], fmt),
                        ha='center', va='center', color='white', fontsize=8)


def plot_learning_curve(
    ax: plt.Axes,
    losses: Sequence[float],
    val_losses: Optional[Sequence[float]] = None,
    title: str = 'Learning Curve',
    ylabel: str = 'Loss',
) -> None:
    """Draw a learning curve."""
    ax.plot(losses, label='train', color='C0')
    if val_losses is not None:
        ax.plot(val_losses, label='val', color='C1', linestyle='--')
    ax.set_xlabel('Epoch / Step')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_contour_and_path(
    ax: plt.Axes,
    f,
    path: Sequence[tuple],
    xlim=(-3, 3),
    ylim=(-3, 3),
    levels: int = 30,
    title: str = 'Optimization Path',
) -> None:
    """Draw an optimization path over contour lines."""
    x = np.linspace(*xlim, 200)
    y = np.linspace(*ylim, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[f(np.array([xi, yi])) for xi in x] for yi in y])
    ax.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.7)
    path = np.asarray(path)
    ax.plot(path[:, 0], path[:, 1], 'r.-', label='path', linewidth=1.5, markersize=4)
    ax.plot(path[0, 0], path[0, 1], 'go', markersize=8, label='start')
    ax.plot(path[-1, 0], path[-1, 1], 'r*', markersize=12, label='end')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_bar_comparison(
    ax: plt.Axes,
    labels: Sequence[str],
    values: Sequence[float],
    title: str = '',
    ylabel: str = 'Time (ms)',
    colors: Optional[Sequence[str]] = None,
) -> None:
    """Draw a bar chart comparison."""
    if colors is None:
        colors = ['C0', 'C1', 'C2', 'C3'][:len(labels)]
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{v:.2f}', ha='center', va='bottom', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')


def plot_distribution(
    ax: plt.Axes,
    samples: np.ndarray,
    title: str = '',
    bins: int = 50,
    density: bool = True,
) -> None:
    """Draw a probability distribution histogram."""
    ax.hist(samples, bins=bins, density=density, alpha=0.7, color='C0', edgecolor='black')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_multiple_lines(
    ax: plt.Axes,
    x: Sequence,
    ys: dict[str, Sequence],
    title: str = '',
    xlabel: str = '',
    ylabel: str = '',
) -> None:
    """Draw multiple lines on one axis. ``ys`` maps labels to values."""
    for label, y in ys.items():
        ax.plot(x, y, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def set_default_font() -> bool:
    """Return whether the default font setup succeeded."""
    return _FONT_OK


set_korean_font = set_default_font
