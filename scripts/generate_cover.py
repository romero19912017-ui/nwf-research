# -*- coding: utf-8 -*-
"""
Генерация графической обложки для статьи NWF на Хабре.
Семантическое поле: потенциалы, заряды, кластеризация.

Запуск: python scripts/generate_cover.py [--3d]
  --3d  - альтернативная 3D-поверхность (медленнее)
"""
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

OUT = Path(__file__).resolve().parent.parent / "grafici"
OUT.mkdir(exist_ok=True)

# Параметры выхода: 4K (3840x2160)
WIDTH, HEIGHT = 3840, 2160
DPI = 150  # figsize = W/DPI x H/DPI
FIGW, FIGH = WIDTH / DPI, HEIGHT / DPI

def make_field(n, n_charges=15, seed=42):
    """Генерирует поле потенциалов и координаты зарядов."""
    np.random.seed(seed)
    x = np.linspace(-4, 4, n)
    y = np.linspace(-2.5, 2.5, int(n * HEIGHT / WIDTH))
    X, Y = np.meshgrid(x, y)
    centers = np.random.randn(n_charges, 2) * 2
    centers[:, 0] *= 0.8
    centers[:, 1] *= 0.6
    strengths = 0.8 + 0.4 * np.random.rand(n_charges)
    sigmas = 0.3 + 0.25 * np.random.rand(n_charges)
    Z = np.zeros_like(X)
    for i in range(n_charges):
        cx, cy = centers[i]
        s = sigmas[i]
        a = strengths[i]
        Z += a * np.exp(-0.5 * ((X - cx)**2 + (Y - cy)**2) / (s**2))
    Z = gaussian_filter(Z, sigma=2)
    Z = Z / (Z.max() + 1e-9)
    return X, Y, Z, centers, n_charges, x, y

def render_2d(save_4k=True, save_hd=True):
    X, Y, Z, centers, n_charges, x, y = make_field(400)
    
    # Кастомная colormap: тёмный фон -> синий -> фиолетовый -> оранжевый
    colors_hex = ['#0a0a12', '#0d1b2a', '#1b263b', '#415a77', '#778da9',
                  '#e0e1dd', '#9d4edd', '#c77dff', '#ff6b35', '#ff9f1c']
    cmap = LinearSegmentedColormap.from_list('nwf_cover', colors_hex, N=256)
    
    fig, ax = plt.subplots(figsize=(FIGW, FIGH), facecolor='#050508')
    ax.set_facecolor('#050508')
    
    # Рисуем поле (контурная заливка)
    levels = np.linspace(0.05, 1, 40)
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=0.95)
    
    # Контурные линии для глубины (полупрозрачные)
    ax.contour(X, Y, Z, levels=levels[::4], colors='white', alpha=0.08, linewidths=0.5)
    
    # Заряды: эффект свечения (несколько слоёв с убывающей яркостью)
    for i in range(n_charges):
        cx, cy = centers[i]
        # Внешнее свечение
        for r, alpha in [(0.8, 0.15), (0.4, 0.35), (0.15, 0.6)]:
            ax.scatter([cx], [cy], s=8000*r, c='white', alpha=alpha, edgecolors='none')
        # Ядро заряда - цвет по кластеру (циклически)
        cluster_color = plt.cm.plasma(i / max(n_charges, 1))[:3]
        ax.scatter([cx], [cy], s=120, c=[cluster_color], alpha=1, edgecolors='white', 
                   linewidths=1.5, zorder=10)
    
    # Линии связей между близкими зарядами (сеть)
    dist_threshold = 2.0
    for i in range(n_charges):
        for j in range(i + 1, n_charges):
            d = np.sqrt((centers[i, 0] - centers[j, 0])**2 + (centers[i, 1] - centers[j, 1])**2)
            if d < dist_threshold:
                alpha = 0.2 * (1 - d / dist_threshold)
                ax.plot([centers[i, 0], centers[j, 0]], [centers[i, 1], centers[j, 1]], 
                        color='white', alpha=alpha, linewidth=1)
    
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout(pad=0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Сохранение
    out_path = OUT / "cover_nwf_4k.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='#050508', 
                edgecolor='none', pad_inches=0)
    plt.close()
    
    print(f"Обложка сохранена: {out_path}")
    
    # Дополнительно: версия 1920x1080 для быстрой загрузки
    fig2, ax2 = plt.subplots(figsize=(1920/150, 1080/150), facecolor='#050508')
    ax2.set_facecolor('#050508')
    ax2.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=0.95)
    ax2.contour(X, Y, Z, levels=levels[::4], colors='white', alpha=0.08, linewidths=0.5)
    for i in range(n_charges):
        cx, cy = centers[i]
        for r, alpha in [(0.8, 0.15), (0.4, 0.35), (0.15, 0.6)]:
            ax2.scatter([cx], [cy], s=4000*r, c='white', alpha=alpha, edgecolors='none')
        cluster_color = plt.cm.plasma(i / max(n_charges, 1))[:3]
        ax2.scatter([cx], [cy], s=80, c=[cluster_color], alpha=1, edgecolors='white', 
                    linewidths=1, zorder=10)
    for i in range(n_charges):
        for j in range(i + 1, n_charges):
            d = np.sqrt((centers[i, 0] - centers[j, 0])**2 + (centers[i, 1] - centers[j, 1])**2)
            if d < dist_threshold:
                alpha = 0.2 * (1 - d / dist_threshold)
                ax2.plot([centers[i, 0], centers[j, 0]], [centers[i, 1], centers[j, 1]], 
                         color='white', alpha=alpha, linewidth=0.8)
    ax2.set_xlim(x.min(), x.max())
    ax2.set_ylim(y.min(), y.max())
    ax2.set_aspect('equal')
    ax2.axis('off')
    plt.tight_layout(pad=0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    out_hd = OUT / "cover_nwf_1920.png"
    plt.savefig(out_hd, dpi=150, bbox_inches='tight', facecolor='#050508', 
                edgecolor='none', pad_inches=0)
    plt.close()
    print(f"Обложка HD: {out_hd}")


def render_3d(save_4k=True):
    """Альтернатива: 3D-поверхность семантического поля."""
    X, Y, Z, centers, n_charges, x, y = make_field(120)  # Меньше точек для 3D
    fig = plt.figure(figsize=(FIGW, FIGH), facecolor='#050508')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#050508')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    cmap = LinearSegmentedColormap.from_list('nwf_3d', 
        ['#0a0a12', '#1b263b', '#778da9', '#9d4edd', '#ff6b35', '#ff9f1c'], N=256)
    ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.95, antialiased=True, rstride=2, cstride=2)
    for i in range(n_charges):
        cx, cy = centers[i]
        zi = min(np.argmin(np.abs(y - cy)), Z.shape[0] - 1)
        zj = min(np.argmin(np.abs(x - cx)), Z.shape[1] - 1)
        cz = float(Z[zi, zj])
        ax.scatter([cx], [cy], [cz], s=200, c=[plt.cm.plasma(i/n_charges)[:3]], 
                   alpha=1, edgecolors='white', linewidths=1)
    ax.view_init(elev=25, azim=45)
    out_3d = OUT / "cover_nwf_3d_4k.png"
    plt.savefig(out_3d, dpi=DPI, bbox_inches='tight', facecolor='#050508', pad_inches=0)
    plt.close()
    print(f"Обложка 3D: {out_3d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--3d", dest="use_3d", action="store_true", help="Сгенерировать 3D-вариант")
    args = ap.parse_args()
    render_2d()
    if args.use_3d:
        render_3d()
