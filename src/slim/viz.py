import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import networkx as nx
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch

# from copy import deepcopy


def plot_internal_model(data, **kwargs):
    ax = kwargs.get('ax')
    if ax is None:
        ax = plt.subplot(111)
    title = kwargs.get('title', 'Internal Representation')
    states = kwargs.get('states', np.arange(data.shape[0]))
    annot = kwargs.get('annot', False)
    cbar = kwargs.get('cbar', False)
    rotx = kwargs.get('rotx', 0)
    fmt = kwargs.get('fmt', '.2f')

    sns.heatmap(data, ax=ax, annot=annot, fmt=fmt, linewidths=1, linecolor='black', cmap='coolwarm', cbar=cbar)
    ax.set_title(title, fontsize=20, pad=20)
    ax.set_xticklabels(states, rotation=rotx, fontsize=14, fontweight='bold')
    ax.set_yticklabels(states, rotation=0, fontsize=14, fontweight='bold')

def lineplot(data, **kwargs):
    ax = kwargs.get('ax')
    if ax is None:
        ax = plt.subplot(111)
    ax.set_xlim(-0.5, 9.5)

    color = kwargs.get('color')
    alpha = kwargs.get('alpha')
    label = kwargs.get('label')
    points = kwargs.get('as_points')
    linewidth = kwargs.get('linewidth')
    linestyle = kwargs.get('linestyle')

    sns.lineplot(data, ax=ax, color=color, alpha=alpha, label=label, marker=points, linewidth=linewidth,
                 linestyle=linestyle)

    ax.set_title(kwargs.get('title', ''), fontsize=20, pad=20)
    ax.set_xticks(range(len(data)), range(len(data)), rotation=0)

def barplot(data, **kwargs):
    ax = kwargs.get('ax')
    if ax is None:
        ax = plt.subplot(111)
    ax.set_xlim(-0.5, 9.5)
    marker = kwargs.get('marker')
    if marker is not None:
        sns.barplot(data, ax=ax, color='red', marker=marker, linestyle='None')
    sns.barplot(data, ax=ax, color='red')

    ax.set_title(kwargs.get('title', ''), fontsize=20, pad=20)
    ax.set_xticks(range(len(data)), range(len(data)), rotation=0)


def plot_history(data, **kwargs):
    """

    :param data:
    :param kwargs:
    Extra Parameters
    ----------
    cmap
    grid
    states
    title
    :return:
    """
    if isinstance(data, list):
        data = np.array(data)

    ax = kwargs.get('ax')
    if ax is None:
        ax = plt.subplot(111)

    cmap = kwargs.get('cmap')
    grid = kwargs.get('grid', False)
    if cmap is None:
        cmap = plt.get_cmap('Grays')
    # cmap = sns.color_palette("viridis", as_cmap=True)
    states = kwargs.get('states', range(data.shape[1]))
    sns.heatmap(data.T, ax=ax, cbar=False, yticklabels=states, cmap=cmap)
    ax.set_title(kwargs.get('title', ''), fontsize=20, pad=20)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=14, fontweight='bold', rotation=0)

    if grid:
        ax.grid(grid, alpha=0.5, linestyle='--')


def plot_trajectory(data, **kwargs):
    ax = kwargs.get('ax')
    title = kwargs.get('title', '')
    if ax is None:
        ax = plt.subplot(111)
    sns.lineplot(data=data, ax=ax)
    ax.set_title(title, fontsize=20, pad=20)


def create_digraph(model, states, axis, thr=0.1, pos=None, fontsize=25):
    G_layout = nx.DiGraph()
    G_layout.add_nodes_from(states)
    if pos is None:
        pos = nx.spring_layout(G_layout, seed=42)

    G = nx.DiGraph()
    for i, state_from in enumerate(states):
        for j, state_to in enumerate(states):
            if model.W[i, j] > thr:
                G.add_edge(state_from, state_to, weight=model.W[i, j])
    norm = mcolors.Normalize(vmin=0., vmax=.5)
    cmap = plt.cm.Greys
    edge_labels = nx.get_edge_attributes(G, 'weight')
    edge_colors = [cmap(norm(G[u][v]['weight'])) for u, v in G.edges()]
    nx.draw(G, pos, ax=axis, with_labels=True, node_size=1000, width=3, node_color='lightgray',font_size=15,
            font_weight='bold',arrows=True, edgecolors='black', arrowsize=20, edge_color=edge_colors, connectionstyle='arc3,rad=0.2')

    for (u, v), w in edge_labels.items():
        if w > thr:
            if u != v:
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                dx, dy = x2 - x1, y2 - y1
                perp_offset = -0.3  # Increase for more distance from edge
                norm = np.sqrt(dx**2 + dy**2)
                ox, oy = -dy / norm * perp_offset, dx / norm * perp_offset
                axis.text(mx + ox - 0.12, my + oy, f"{w:.1f}", fontsize=fontsize, bbox=dict(facecolor='none', edgecolor='none')) #, alpha=0.0))
            else:  # self connection
                x, y = pos[u]
                axis.text(x, y + 0.3, f"{w:.1f}", fontsize=fontsize, va='center', ha='center', bbox=dict(facecolor='none', edgecolor='none')) #, alpha=0.6))
    for node in states:
        if G.has_edge(node, node):
            x, y = pos[node]
            offset = 0.2
            axis.add_patch(FancyArrowPatch((x + offset, y + offset), (x + offset, y + offset),
                                           connectionstyle="arc3,rad=0.5",  arrowstyle='->',
                                           mutation_scale=20, color='black', lw=1.5))

    axis.axis('off')
    axis.set_title(model.model_name, fontweight='bold', fontsize=35, pad=20)

