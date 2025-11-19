import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os  # <-- added

TITLE_FONTSIZE = 24
LABEL_FONTSIZE = 15
LEGEND_FONTSIZE = 20

def build_bipartite_graph(df, col_plant="plant_name", col_poll="pollinator_name"):
    """
    Build a bipartite NetworkX graph from the strengths dataframe.
    Nodes: plants and pollinators
    Edges: plant-pollinator pairs, with attributes:
        - robustness_score  (for edge thickness)
        - PFIS_temp
        - PFIS_wind
        - PFIS_sun
    """
    plants = sorted(df[col_plant].unique())
    pollinators = sorted(df[col_poll].unique())

    G = nx.Graph()
    G.add_nodes_from(plants, bipartite="plants")
    G.add_nodes_from(pollinators, bipartite="pollinators")

    required_edge_cols = ["robustness_score", "PFIS_temp", "PFIS_wind", "PFIS_sun"]
    missing = [c for c in required_edge_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in strengths.csv: {missing}")

    for row in df.itertuples(index=False):
        plant = getattr(row, col_plant)
        poll = getattr(row, col_poll)
        robustness = float(getattr(row, "robustness_score"))
        pfis_temp = float(getattr(row, "PFIS_temp"))
        pfis_wind = float(getattr(row, "PFIS_wind"))
        pfis_sun = float(getattr(row, "PFIS_sun"))

        G.add_edge(
            plant,
            poll,
            weight=robustness,
            robustness=robustness,
            PFIS_temp=pfis_temp,
            PFIS_wind=pfis_wind,
            PFIS_sun=pfis_sun,
        )

    return G, plants, pollinators


def compute_ordered_layout(G, plants, pollinators):
    """
    Ordered bipartite layout:
    - plants on x=0, ordered by degree
    - pollinators on x=1, ordered by degree
    """
    plant_deg = dict(G.degree(plants))
    poll_deg = dict(G.degree(pollinators))

    plants_sorted = sorted(plants, key=lambda n: plant_deg.get(n, 0), reverse=True)
    polls_sorted = sorted(pollinators, key=lambda n: poll_deg.get(n, 0), reverse=True)

    pos = {}

    # Plants at x = 0
    n_plants = len(plants_sorted)
    if n_plants > 1:
        plant_y = [i / (n_plants - 1) for i in range(n_plants)]
    else:
        plant_y = [0.5]
    for y, node in zip(plant_y, plants_sorted):
        pos[node] = (0.0, y)

    # Pollinators at x = 1
    n_polls = len(polls_sorted)
    if n_polls > 1:
        poll_y = [i / (n_polls - 1) for i in range(n_polls)]
    else:
        poll_y = [0.5]
    for y, node in zip(poll_y, polls_sorted):
        pos[node] = (1.0, y)

    return pos


def build_pollinator_short_labels(pollinators):
    """
    Build a mapping {full_name -> short_label} for pollinators.

    Short label = first initial of first word + first initial of second word.
    If that pair of initials is used by more than one pollinator, append
    a number (e.g. 'Bt1', 'Bt2', ...).
    """
    base_labels = []

    for name in pollinators:
        parts = name.split()

        # Default: try to use first two words as genus + species
        if len(parts) >= 2:
            base = parts[0][0] + parts[1][0]
        elif len(parts) == 1:
            # If only one word, just use its first two letters
            base = parts[0][:2]
        else:
            base = name[:2]

        base_labels.append((name, base))

    # Count how many times each base label appears
    counts = {}
    for _, base in base_labels:
        counts[base] = counts.get(base, 0) + 1

    # Build final mapping, numbering duplicates
    label_map = {}
    used_indices = {}
    for name, base in base_labels:
        if counts[base] == 1:
            label_map[name] = base
        else:
            idx = used_indices.get(base, 0) + 1
            used_indices[base] = idx
            label_map[name] = f"{base}{idx}"

    return label_map


def edge_widths_from_robustness(G):
    """
    Compute edge widths based on robustness_score and return a dict
    mapping (u, v) -> width.
    """
    edges = list(G.edges(data=True))
    if not edges:
        return {}

    robustness_values = [d.get("robustness", 1.0) for _, _, d in edges]
    max_r = max(robustness_values) if robustness_values else 1.0
    if max_r == 0:
        max_r = 1.0

    widths = {}
    for (u, v, d), r in zip(edges, robustness_values):
        width = 0.5 + 4.5 * (r / max_r)
        widths[(u, v)] = width

    return widths


def draw_labels_with_pollinators_right(G, plants, pollinators_to_label, pos, poll_label_map=None):
    """
    Draw plant labels slightly to the LEFT of plant nodes (full names)
    and pollinator labels slightly to the RIGHT of pollinator nodes
    using short labels from poll_label_map if provided.
    """

    # --- Plants: labels to the LEFT of the node (full names) ---
    plant_offset = -0.12  # negative -> move left
    plant_labels = {n: n for n in plants}
    pos_plant_labels = {n: (pos[n][0] + plant_offset, pos[n][1]) for n in plants}

    nx.draw_networkx_labels(
        G,
        pos_plant_labels,
        labels=plant_labels,
        font_size=LABEL_FONTSIZE,
        font_weight="bold",
        verticalalignment="center",
        horizontalalignment="right",  # text extends leftwards
    )

    # --- Pollinators: labels to the RIGHT, using short codes ---
    poll_offset = 0.12  # positive -> move right
    if poll_label_map is None:
        poll_labels = {n: n for n in pollinators_to_label}
    else:
        poll_labels = {n: poll_label_map.get(n, n) for n in pollinators_to_label}

    pos_poll_labels = {n: (pos[n][0] + poll_offset, pos[n][1]) for n in pollinators_to_label}

    nx.draw_networkx_labels(
        G,
        pos_poll_labels,
        labels=poll_labels,
        font_size=LABEL_FONTSIZE,
        font_weight="bold",
        verticalalignment="center",
        horizontalalignment="left",  # text extends rightwards
    )



def set_x_limits_for_labels(pos):
    """
    Extend x-limits so that the right-hand labels are not clipped.
    """
    xs = [p[0] for p in pos.values()]
    xmin, xmax = min(xs), max(xs)
    # extra space on both sides, more on the right for labels
    plt.xlim(xmin - 0.1, xmax + 0.7)


def draw_striped_edges(pos, edgelist, widths, color1="red", color2="yellow", n_stripes=20):
    """
    Draw edges as alternating coloured segments (stripes) between the
    two colours. Intended for edges affected by both temperature and sunshine.
    """
    ax = plt.gca()
    for (u, v), w in zip(edgelist, widths):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        dx = x2 - x1
        dy = y2 - y1

        for i in range(n_stripes):
            t0 = i / n_stripes
            t1 = (i + 1) / n_stripes
            xs = [x1 + dx * t0, x1 + dx * t1]
            ys = [y1 + dy * t0, y1 + dy * t1]
            color = color1 if i % 2 == 0 else color2
            ax.plot(xs, ys, color=color, linewidth=w, solid_capstyle="round", alpha=0.9)


def plot_extinction_graph(G, plants, pollinators, pos):
    """
    Plot the graph showing all edges that disappear under any PFIS.
    Colour scheme:
        - red: temperature only
        - blue: wind only
        - orange: sunshine only
        - striped red/yellow: temperature & sunshine both <= -1
        - light grey: edges with no extinction
    Only pollinators with at least one extinction edge are labelled.
    Returns the list of extinction edges (to be removed for the second graph).
    """
    widths_dict = edge_widths_from_robustness(G)

    red_edges = []
    blue_edges = []
    orange_edges = []
    striped_edges = []
    grey_edges = []
    extinction_edges = []

    pollinators_with_extinction = set()

    for u, v, data in G.edges(data=True):
        t_flag = float(data.get("PFIS_temp", 0.0)) <= -1.0
        w_flag = float(data.get("PFIS_wind", 0.0)) <= -1.0
        s_flag = float(data.get("PFIS_sun", 0.0)) <= -1.0

        if t_flag or w_flag or s_flag:
            extinction_edges.append((u, v))

            # mark pollinator endpoint(s) for labelling
            if u in pollinators:
                pollinators_with_extinction.add(u)
            if v in pollinators:
                pollinators_with_extinction.add(v)

            # classify for colouring
            if t_flag and s_flag:
                striped_edges.append((u, v))
            elif t_flag:
                red_edges.append((u, v))
            elif w_flag:
                blue_edges.append((u, v))
            elif s_flag:
                orange_edges.append((u, v))
        else:
            grey_edges.append((u, v))

    plt.figure(figsize=(10, 16))  # taller than wide

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=plants, node_color="lightgreen", label="Plants")
    nx.draw_networkx_nodes(G, pos, nodelist=pollinators, node_color="lightblue", label="Pollinators")

    # Draw non-extinction (grey) edges
    if grey_edges:
        widths_grey = [widths_dict.get((u, v), 1.0) for (u, v) in grey_edges]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=grey_edges,
            width=widths_grey,
            edge_color="lightgrey",
            alpha=0.4,
        )

    # Draw simple coloured extinction edges
    if red_edges:
        widths_red = [widths_dict.get((u, v), 1.0) for (u, v) in red_edges]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=red_edges,
            width=widths_red,
            edge_color="red",
            alpha=0.9,
        )

    if orange_edges:
        widths_orange = [widths_dict.get((u, v), 1.0) for (u, v) in orange_edges]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=orange_edges,
            width=widths_orange,
            edge_color="green",
            alpha=0.9,
        )

    # Draw striped edges for temperature + sunshine
    if striped_edges:
        widths_striped = [widths_dict.get((u, v), 1.0) for (u, v) in striped_edges]
        draw_striped_edges(pos, striped_edges, widths_striped, color1="red", color2="green")

    # Labels: all plants, plus pollinators with at least one extinction edge
    draw_labels_with_pollinators_right(G, plants, pollinators_with_extinction, pos)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", linestyle="None",
               color="lightgreen", label="Plants"),
        Line2D([0], [0], marker="o", linestyle="None",
               color="lightblue", label="Pollinators"),
        Line2D([0], [0], color="lightgrey", lw=3,
               label="No extinction"),
        Line2D([0], [0], color="red", lw=3,
               label="Extinction - Temperature"),
        Line2D([0], [0], color="green", lw=3,
               label="Extinction - Sunshine"),
    ]
    plt.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        prop={"size": LEGEND_FONTSIZE},
    )

    plt.title(
        "Plant-Pollinator Bipartite Network\n"
        "Edges that disappear under PFIS (coloured by driver)",
        fontsize=TITLE_FONTSIZE,
    )

    set_x_limits_for_labels(pos)
    plt.axis("off")
    plt.tight_layout()

    # --- added block: save figure ---
    os.makedirs("MDM2/outputs", exist_ok=True)
    plt.savefig("MDM2/outputs/image_3.png", dpi=300, bbox_inches="tight")
    # -------------------------------

    plt.show()

    return extinction_edges


def plot_pruned_graph(G_pruned, plants, pollinators, pos, removed_count, fraction_removed):
    """
    Plot the bipartite graph after removing all extinction edges.
    Only pollinators that still have at least one edge are labelled.
    """
    widths_dict = edge_widths_from_robustness(G_pruned)

    # Pollinators with remaining edges
    pollinators_with_edges = {p for p in pollinators if G_pruned.degree(p) > 0}

    plt.figure(figsize=(10, 16))  # taller than wide

    nx.draw_networkx_nodes(G_pruned, pos, nodelist=plants, node_color="lightgreen", label="Plants")
    nx.draw_networkx_nodes(G_pruned, pos, nodelist=pollinators, node_color="lightblue", label="Pollinators")

    if G_pruned.number_of_edges() > 0:
        edgelist = list(G_pruned.edges())
        widths = [widths_dict.get((u, v), 1.0) for (u, v) in edgelist]
        nx.draw_networkx_edges(
            G_pruned,
            pos,
            edgelist=edgelist,
            width=widths,
            edge_color="lightgrey",
            alpha=0.8,
        )

    # Labels: all plants, plus only pollinators that still have edges
    draw_labels_with_pollinators_right(G_pruned, plants, pollinators_with_edges, pos)

    plt.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="None",
                   color="lightgreen", label="Plants"),
            Line2D([0], [0], marker="o", linestyle="None",
                   color="lightblue", label="Pollinators"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        prop={"size": LEGEND_FONTSIZE},
    )

    plt.title(
        "Plant-Pollinator Bipartite Network\n"
        "After removing all extinction edges\n"
        f"Removed ({fraction_removed:.2%} of edges)",
        fontsize=TITLE_FONTSIZE,
    )

    set_x_limits_for_labels(pos)
    plt.axis("off")
    plt.tight_layout()

    # --- added block: save figure ---
    os.makedirs("MDM2/outputs", exist_ok=True)
    plt.savefig("MDM2/outputs/image_4.png", dpi=300, bbox_inches="tight")
    # -------------------------------

    plt.show()


def plot_feature_PFIS_coloured_graph(
    G,
    plants,
    pollinators,
    pos,
    feature_name: str,
    title_feature_label: str,
    output_path: str,
):
    """
    Plot a bipartite graph where edge colours reflect PFIS values for a single feature.

    Colour scheme (based on the feature_name attribute on edges):
        - red   : -1.0 <= PFIS < -0.8
        - orange: -0.8 <= PFIS < -0.5
        - black : all other edges
    """
    widths_dict = edge_widths_from_robustness(G)

    red_edges = []
    orange_edges = []
    black_edges = []

    for u, v, data in G.edges(data=True):
        val = float(data.get(feature_name, float("nan")))
        if -1.0 <= val < -0.8:
            red_edges.append((u, v))
        elif -0.8 <= val < -0.5:
            orange_edges.append((u, v))
        else:
            black_edges.append((u, v))

    plt.figure(figsize=(200, 5))  

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=plants, node_color="lightgreen", label="Plants")
    nx.draw_networkx_nodes(G, pos, nodelist=pollinators, node_color="lightblue", label="Pollinators")

    # Draw edges by category
    if black_edges:
        widths_black = [widths_dict.get((u, v), 1.0) for (u, v) in black_edges]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=black_edges,
            width=widths_black,
            edge_color="black",
            alpha=0.6,
        )

    if orange_edges:
        widths_orange = [widths_dict.get((u, v), 1.0) for (u, v) in orange_edges]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=orange_edges,
            width=widths_orange,
            edge_color="orange",
            alpha=0.9,
        )

    if red_edges:
        widths_red = [widths_dict.get((u, v), 1.0) for (u, v) in red_edges]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=red_edges,
            width=widths_red,
            edge_color="red",
            alpha=0.9,
        )

    # Labels: all plants and all pollinators
    draw_labels_with_pollinators_right(G, plants, pollinators, pos)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", linestyle="None",
               color="lightgreen", label="Plants"),
        Line2D([0], [0], marker="o", linestyle="None",
               color="lightblue", label="Pollinators"),
        Line2D([0], [0], color="red", lw=3,
               label="-1 ≤ PFIS < -0.8"),
        Line2D([0], [0], color="orange", lw=3,
               label="-0.8 ≤ PFIS < -0.5"),
        Line2D([0], [0], color="black", lw=3,
               label="Other edges"),
    ]
    plt.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        prop={"size": LEGEND_FONTSIZE},
    )

    plt.title(
        "Plant–Pollinator Bipartite Network\n"
        f"PFIS coloured by {title_feature_label}",
        fontsize=TITLE_FONTSIZE,
    )
    set_x_limits_for_labels(pos)
    plt.axis("off")
    plt.tight_layout()

    # Ensure output directory exists and save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def classify_edges_by_PFIS(G):
    """
    Classify edges according to PFIS across ALL features (temp, wind, sun).

    Returns
    -------
    unaffected_edges : list[(u, v)]
        Edges with PFIS >= -0.5 for ALL features.
    moderate_edges : list[(u, v)]
        Edges that have -0.8 <= PFIS < -0.5 for AT LEAST one feature,
        and no feature with PFIS < -0.8.
    strong_edges : list[(u, v)]
        Edges that have PFIS < -0.8 for AT LEAST one feature.
    """
    unaffected_edges = []
    moderate_edges = []
    strong_edges = []

    for u, v, data in G.edges(data=True):
        vals = [
            float(data.get("PFIS_temp", 0.0)),
            float(data.get("PFIS_wind", 0.0)),
            float(data.get("PFIS_sun", 0.0)),
        ]

        has_strong = any(vv < -0.8 for vv in vals)
        has_moderate = any(-0.8 <= vv < -0.5 for vv in vals)
        all_not_affected = all(vv >= -0.5 for vv in vals)

        if has_strong:
            strong_edges.append((u, v))
        elif has_moderate:
            moderate_edges.append((u, v))
        elif all_not_affected:
            unaffected_edges.append((u, v))
        else:
            # This should not happen given the above logic, but just in case
            unaffected_edges.append((u, v))

    return unaffected_edges, moderate_edges, strong_edges


def plot_combined_PFIS_coloured_graph(
    G, plants, pollinators, pos, output_path: str, poll_label_map
):
    """
    Horizontal bipartite graph with:
      - plants on the top row (y = 1)
      - pollinators on the bottom row (y = 0)
    Edge colours reflect the worst PFIS across temp, wind, sun.
    Labels for each row follow an up–down–up pattern in y to reduce overlap.
    """

    # ---------- build horizontal layout with shared x-scale ----------
    plants_sorted = sorted(plants, key=lambda n: pos[n][1])
    polls_sorted = sorted(pollinators, key=lambda n: pos[n][1])

    n_plants = len(plants_sorted)
    n_polls = len(polls_sorted)

    pos_h = {}

    # Spread plants evenly between x=0 and x=1
    for i, n in enumerate(plants_sorted):
        x = i / (n_plants - 1) if n_plants > 1 else 0.5
        pos_h[n] = (x, 1.0)   # top row

    # Spread pollinators evenly between x=0 and x=1 (same horizontal span)
    for j, n in enumerate(polls_sorted):
        x = j / (n_polls - 1) if n_polls > 1 else 0.5
        pos_h[n] = (x, 0.0)   # bottom row

    # ---------- classify edges by worst PFIS across all drivers ----------
    widths_dict = edge_widths_from_robustness(G)

    red_edges = []
    orange_edges = []
    black_edges = []

    for u, v, data in G.edges(data=True):
        vals = [
            float(data.get("PFIS_temp", 0.0)),
            float(data.get("PFIS_wind", 0.0)),
            float(data.get("PFIS_sun", 0.0)),
        ]
        min_val = min(vals)

        if min_val < -0.8:
            red_edges.append((u, v))
        elif -0.8 <= min_val < -0.5:
            orange_edges.append((u, v))
        else:
            black_edges.append((u, v))

    # ---------- draw figure ----------
    plt.figure(figsize=(120, 10))

    # Nodes – plants larger than pollinators
    nx.draw_networkx_nodes(
        G, pos_h, nodelist=plants_sorted,
        node_color="lightgreen",
        label="Plants",
        node_size=800,
    )
    nx.draw_networkx_nodes(
        G, pos_h, nodelist=polls_sorted,
        node_color="lightblue",
        label="Pollinators",
        node_size=400,
    )

    # Black edges (unaffected by any feature)
    if black_edges:
        widths_black = [widths_dict.get((u, v), 1.0) for (u, v) in black_edges]
        nx.draw_networkx_edges(
            G,
            pos_h,
            edgelist=black_edges,
            width=widths_black,
            edge_color="black",
            alpha=0.6,
        )

    # Orange edges – twice the usual width
    if orange_edges:
        widths_orange = [2.0 * widths_dict.get((u, v), 1.0) for (u, v) in orange_edges]
        nx.draw_networkx_edges(
            G,
            pos_h,
            edgelist=orange_edges,
            width=widths_orange,
            edge_color="orange",
            alpha=0.9,
        )

    # Red edges – twice the usual width
    if red_edges:
        widths_red = [2.0 * widths_dict.get((u, v), 1.0) for (u, v) in red_edges]
        nx.draw_networkx_edges(
            G,
            pos_h,
            edgelist=red_edges,
            width=widths_red,
            edge_color="red",
            alpha=0.9,
        )

    # ---------- labels: plants above, pollinators below, zig-zag in y ----------

    # Plants: alternate label heights above the top row
    plant_label_pos = {}
    for i, n in enumerate(plants_sorted):
        x, y = pos_h[n]
        # base offset above node
        base = 0.06
        # even indices a bit higher, odd a bit lower -> up–down–up pattern
        extra = 0.03 if i % 2 == 0 else -0.03
        plant_label_pos[n] = (x, y + base + extra)

    nx.draw_networkx_labels(
        G,
        plant_label_pos,
        labels={n: n for n in plants_sorted},
        font_size=LABEL_FONTSIZE,
        font_weight="bold",
        verticalalignment="bottom",
        horizontalalignment="center",
    )

    # Pollinators: alternate label depths below the bottom row
    poll_label_pos = {}
    for j, n in enumerate(polls_sorted):
        x, y = pos_h[n]
        base = -0.06
        extra = -0.03 if j % 2 == 0 else 0.03  # down–up–down pattern
        poll_label_pos[n] = (x, y + base + extra)

    nx.draw_networkx_labels(
        G,
        poll_label_pos,
        labels={n: poll_label_map.get(n, n) for n in polls_sorted},
        font_size=LABEL_FONTSIZE,
        font_weight="bold",
        verticalalignment="top",
        horizontalalignment="center",
    )

    # ---------- legend, title, final tweaks ----------
    legend_elements = [
        Line2D([0], [0], marker="o", linestyle="None",
               color="lightgreen", label="Plants"),
        Line2D([0], [0], marker="o", linestyle="None",
               color="lightblue", label="Pollinators"),
        Line2D([0], [0], color="red", lw=3,
               label="PFIS < -0.8 (any feature)"),
        Line2D([0], [0], color="orange", lw=3,
               label="-0.8 ≤ PFIS < -0.5 (any feature)"),
        Line2D([0], [0], color="black", lw=3,
               label="All PFIS ≥ -0.5"),
    ]
    plt.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        prop={"size": LEGEND_FONTSIZE},
    )

    plt.title(
        "Plant-Pollinator Bipartite Network\n"
        "PFIS coloured by worst driver (Temperature, Sunshine)",
        fontsize=TITLE_FONTSIZE + 1,
    )

    plt.axis("off")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()



def plot_unaffected_and_moderate(
    G,
    plants,
    pollinators,
    pos,
    unaffected_edges,
    moderate_edges,
    total_edges,
    strong_edges_count,
    affected_edges_count,
    output_path: str,
    poll_label_map,
):
    """
    Plot a graph containing:
      * black edges  : not impacted by any feature (PFIS >= -0.5 for all)
      * light grey   : PFIS in [-0.8, -0.5) for ANY feature
    All edges with PFIS < -0.8 for ANY feature are removed from this plot.

    Light-grey (moderately affected) edges are drawn with twice the width
    of unaffected edges.
    """
    widths_dict = edge_widths_from_robustness(G)

    plt.figure(figsize=(20, 16))  # taller than wide

    # Nodes – plants bigger
    nx.draw_networkx_nodes(
        G, pos, nodelist=plants,
        node_color="lightgreen",
        label="Plants",
        node_size=800,
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=pollinators,
        node_color="lightblue",
        label="Pollinators",
        node_size=400,
    )

    # Unaffected (black) edges
    if unaffected_edges:
        widths_black = [widths_dict.get((u, v), 1.0) for (u, v) in unaffected_edges]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=unaffected_edges,
            width=widths_black,
            edge_color="black",
            alpha=0.8,
        )

    # Moderately affected (light grey) edges – twice the width
    if moderate_edges:
        widths_grey = [2.0 * widths_dict.get((u, v), 1.0) for (u, v) in moderate_edges]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=moderate_edges,
            width=widths_grey,
            edge_color="orange",
            alpha=0.9,
        )

    # Labels: all plants + pollinators that still have edges
    pollinators_in_plot = {
        p for p in pollinators if any((p == u or p == v) for (u, v) in (unaffected_edges + moderate_edges))
    }
    draw_labels_with_pollinators_right(G, plants, pollinators_in_plot, pos, poll_label_map)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", linestyle="None", color="lightgreen", label="Plants"),
        Line2D([0], [0], marker="o", linestyle="None", color="lightblue", label="Pollinators"),
        Line2D([0], [0], color="black", lw=3, label="Not impacted (all PFIS ≥ -0.5)"),
        Line2D([0], [0], color="orange", lw=3, label="-0.8 ≤ PFIS < -0.5 (any feature)"),
    ]
    plt.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        prop={"size": LEGEND_FONTSIZE},
    )

    removed_edges_count = strong_edges_count
    removed_pct = (removed_edges_count / total_edges) * 100 if total_edges > 0 else 0.0
    affected_pct = (affected_edges_count / total_edges) * 100 if total_edges > 0 else 0.0

    plt.title(
        "Remaining relationships with stress conditions applied\n",
        fontsize=TITLE_FONTSIZE,
    )

    set_x_limits_for_labels(pos)
    plt.axis("off")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def write_pollinator_label_table(poll_label_map, filepath="MDM2/outputs/pollinator_label_key.csv"):
    """
    Write a table translating each shortened pollinator label to its full name.

    Parameters
    ----------
    poll_label_map : dict
        Mapping {full_name -> short_label}

    filepath : str
        Output file path. Supports CSV, TXT, or MD (Markdown).
    """

    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Build rows sorted by short label for readability
    rows = [(short, full) for full, short in poll_label_map.items()]
    rows.sort(key=lambda x: x[0])  # sort by short label

    # Decide format based on file extension
    if filepath.endswith(".csv"):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("Short Label,Full Name\n")
            for short, full in rows:
                f.write(f"{short},{full}\n")

    elif filepath.endswith(".md"):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("| Short Label | Full Name |\n")
            f.write("|-------------|-----------|\n")
            for short, full in rows:
                f.write(f"| {short} | {full} |\n")

    else:  # default: plain text
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("Pollinator Label Key\n")
            f.write("--------------------\n")
            for short, full in rows:
                f.write(f"{short}  ->  {full}\n")

    print(f"Pollinator label translation table saved to: {filepath}")


def main():
    # 1) Load data
    df = pd.read_csv("MDM2/outputs/strengths_10.csv")

    # 2) Build graph and layout
    G, plants, pollinators = build_bipartite_graph(df)
    pos = compute_ordered_layout(G, plants, pollinators)

    total_edges = G.number_of_edges()

    # 3) Build short labels for pollinators (used consistently across all plots)
    poll_label_map = build_pollinator_short_labels(pollinators)

    # 4) Save a translation table from short labels to full pollinator names
    write_pollinator_label_table(
        poll_label_map,
        filepath="MDM2/outputs/pollinator_label_key.csv",
    )

    # 5) Combined PFIS graph (all features: temp, wind, sun)
    plot_combined_PFIS_coloured_graph(
        G,
        plants,
        pollinators,
        pos,
        output_path="MDM2/outputs/graph_PFIS_all_drivers.png",
        poll_label_map=poll_label_map,
    )

    # 6) Classification across all features, then plot the “not strongly impacted” graph
    unaffected_edges, moderate_edges, strong_edges = classify_edges_by_PFIS(G)
    removed_count = len(strong_edges)
    moderate_count = len(moderate_edges)
    affected_count = moderate_count + removed_count  # PFIS < -0.5 for any feature

    plot_unaffected_and_moderate(
        G,
        plants,
        pollinators,
        pos,
        unaffected_edges,
        moderate_edges,
        total_edges,
        strong_edges_count=removed_count,
        affected_edges_count=affected_count,
        output_path="MDM2/outputs/graph_unaffected_and_moderate.png",
        poll_label_map=poll_label_map,
    )

    # --- 7) Extra summaries in the console ---

    # Helper to avoid zero-division
    def pct(count):
        return (count / total_edges * 100) if total_edges > 0 else 0.0

    # Removed edges by feature (PFIS < -0.8)
    temp_removed = (df["PFIS_temp"] < -0.8).sum()
    sun_removed = (df["PFIS_sun"] < -0.8).sum()
    both_temp_sun = ((df["PFIS_temp"] < -0.8) & (df["PFIS_sun"] < -0.8)).sum()

    print(f"\n Total edges in original graph: {total_edges}")

    # Edges removed under stress conditions
    print(
        f"\nEdges removed under stress (PFIS < -0.8 for any feature): "
        f"{removed_count} ({pct(removed_count):.2f}%)"
    )
    print(
        f"\n  - Removed by temperature (PFIS_temp < -0.8): "
        f"{temp_removed} ({pct(temp_removed):.2f}%)"
    )
    print(
        f"\n  - Removed by sunshine (PFIS_sun < -0.8): "
        f"{sun_removed} ({pct(sun_removed):.2f}%)"
    )
    print(
        f"\n  - Removed by BOTH temperature and sunshine: "
        f"{both_temp_sun} ({pct(both_temp_sun):.2f}%)"
    )

    # Degraded but not removed: PFIS between -0.8 and -0.5 for any feature
    print(
        f"\nEdges degraded but not removed (-0.8 ≤ PFIS < -0.5 for any feature): "
        f"{moderate_count} ({pct(moderate_count):.2f}%)"
    )

    # Total negatively affected (full range PFIS < -0.5, i.e. degraded + removed)
    print(
        f"\nTotal edges negatively affected (PFIS < -0.5 for any feature): "
        f"{affected_count} ({pct(affected_count):.2f}%)\n"
    )


if __name__ == "__main__":
    main()
