"""
Pure-Python UML Class Diagram → SVG (Portrait DAG Layout)

Features:
- Portrait DAG layout with no overlaps.
- Parents centered only over unique children (DAG-aware).
- Per-line styled text for attributes/methods.
- Straight-edge labels:
    - vertically between the two boxes,
    - pushed to the right until they don't overlap the edge or boxes.
- Multiplicity labels placed locally near their endpoints.
- Grey edges & labels with hover highlighting.
"""

from dataclasses import dataclass, field
from typing import List, Dict
import html


# ======================================================
# Data model
# ======================================================

@dataclass
class UMLClass:
    name: str
    attributes: List[object] = field(default_factory=list)  # str or (str, dict) or [str, dict]
    methods: List[object] = field(default_factory=list)     # str or (str, dict) or [str, dict]
    style: Dict[str, str] = field(default_factory=dict)     # fill, stroke, etc.


@dataclass
class UMLRelation:
    source: str
    target: str
    kind: str = 'association'
    label: str = ''
    source_multiplicity: str = ''
    target_multiplicity: str = ''


# ======================================================
# Text helpers
# ======================================================

def _parse_text_entry(entry):
    """
    Accepts:
        'text'
        ('text', {style})
        ['text', {style}]
    Returns (text, style_dict).
    """
    if (
        isinstance(entry, (tuple, list))
        and len(entry) == 2
        and isinstance(entry[0], str)
        and isinstance(entry[1], dict)
    ):
        return entry[0], entry[1]
    return str(entry), {}


# ======================================================
# Layout helpers
# ======================================================

def _compute_box_size(cls, font_size, char_width, line_height):
    lines = [cls.name]

    for a in cls.attributes:
        text, _ = _parse_text_entry(a)
        lines.append(text)
    for m in cls.methods:
        text, _ = _parse_text_entry(m)
        lines.append(text)

    max_chars = max(len(line) for line in lines) if lines else 1
    width = int(max(140, min(300, max_chars * char_width + 24)))

    h = 10 + line_height
    if cls.attributes or cls.methods:
        h += 8
    h += len(cls.attributes) * line_height
    if cls.attributes and cls.methods:
        h += 6
    h += len(cls.methods) * line_height
    h += 10

    return {
        'width': width,
        'height': h,
        'attr_lines': len(cls.attributes),
        'method_lines': len(cls.methods),
    }


def _build_graph(classes, relations):
    children = {c.name: [] for c in classes}
    parents  = {c.name: [] for c in classes}

    for r in relations:
        if r.source in children and r.target in parents:
            children[r.source].append(r.target)
            parents[r.target].append(r.source)

    return children, parents


def _find_roots(classes, parents):
    roots = [c.name for c in classes if len(parents[c.name]) == 0]
    return roots if roots else [c.name for c in classes]


def _compute_depths(roots, children, names):
    depths = {n: None for n in names}
    q = [(r, 0) for r in roots]

    for r in roots:
        depths[r] = 0

    while q:
        node, d = q.pop(0)
        for ch in children[node]:
            if depths[ch] is None or depths[ch] < d + 1:
                depths[ch] = d + 1
                q.append((ch, d + 1))

    for n in depths:
        if depths[n] is None:
            depths[n] = 0

    return depths


def _layout_tree(
    classes,
    relations,
    font_size,
    vertical_spacing,
    horizontal_spacing,
    margin,
    line_height,
    *,
    horizontal_gaps=None,
    vertical_gaps=None,
):
    """
    DAG-aware portrait layout with label-aware horizontal and vertical padding.

    NEW:
    - horizontal_gaps: {depth → extra horizontal spacing}
    - vertical_gaps:   {depth → extra vertical spacing}

    Both are optional. If omitted, behavior matches the original.
    """

    name_to_class = {c.name: c for c in classes}
    char_width = font_size * 0.60

    # --------------------------------------------------
    # Precompute sizes
    # --------------------------------------------------
    sizes = {
        name: _compute_box_size(cls, font_size, char_width, line_height)
        for name, cls in name_to_class.items()
    }

    # --------------------------------------------------
    # Build DAG + find roots
    # --------------------------------------------------
    children, parents = _build_graph(classes, relations)
    roots = _find_roots(classes, parents)
    depths = _compute_depths(roots, children, list(name_to_class.keys()))

    # --------------------------------------------------
    # Initialize layout dictionary
    # --------------------------------------------------
    layout = {
        name: {
            **sizes[name],
            'x': None,
            'y': None,
            'depth': depths[name],
        }
        for name in name_to_class
    }

    # --------------------------------------------------
    # Extract DAG-aware spanning tree
    # --------------------------------------------------
    tree_children = {n: [] for n in name_to_class}
    tree_parents = {n: None for n in name_to_class}

    for p, chs in children.items():
        for ch in chs:
            if len(parents[ch]) == 1:  # true tree edges
                tree_children[p].append(ch)
                tree_parents[ch] = p

    tree_roots = [n for n in name_to_class if tree_parents[n] is None]
    if not tree_roots:
        tree_roots = list(name_to_class.keys())

    visited = set()
    next_x = margin

    # Helper: get depth-based horizontal gap
    def depth_gap(d):
        base = horizontal_spacing
        if horizontal_gaps:
            return max(base, horizontal_gaps.get(d, 0))
        return base

    # --------------------------------------------------
    # DFS to place each subtree compactly
    # --------------------------------------------------
    def layout_subtree(root, start_x):
        nonlocal next_x
        local_next_x = 0.0
        nodes = set()

        def dfs(node):
            nonlocal local_next_x

            if node in nodes:
                return
            nodes.add(node)

            chs = tree_children[node]
            for c in chs:
                dfs(c)

            if chs:
                # Center parent above median of children
                centers = [layout[c]['x'] + layout[c]['width'] / 2 for c in chs]
                centers.sort()
                median = centers[len(centers) // 2]
                layout[node]['x'] = median - layout[node]['width'] / 2
            else:
                d = layout[node]['depth']
                gap = depth_gap(d)
                layout[node]['x'] = local_next_x
                local_next_x += layout[node]['width'] + gap

        dfs(root)

        # Normalize subtree so left edge = start_x
        min_x = min(layout[n]['x'] for n in nodes)
        shift = start_x - min_x
        for n in nodes:
            layout[n]['x'] += shift

        # Update next_x
        rightmost = max(layout[n]['x'] + layout[n]['width'] for n in nodes)
        visited.update(nodes)
        next_x = rightmost + horizontal_spacing

    # --------------------------------------------------
    # Layout tree roots
    # --------------------------------------------------
    for r in tree_roots:
        if r not in visited:
            layout_subtree(r, next_x)

    # --------------------------------------------------
    # Layout remaining nodes (multi-parent or disconnected)
    # --------------------------------------------------
    for n in name_to_class:
        if n not in visited:
            d = layout[n]['depth']
            gap = depth_gap(d)
            layout[n]['x'] = next_x
            next_x += layout[n]['width'] + gap

    # --------------------------------------------------
    # Enforce left margin
    # --------------------------------------------------
    min_x = min(info['x'] for info in layout.values())
    if min_x < margin:
        shift = margin - min_x
        for info in layout.values():
            info['x'] += shift

    # --------------------------------------------------
    # Vertical placement with depth-dependent gaps
    # --------------------------------------------------
    depth_heights = {}
    for name, info in layout.items():
        d = info['depth']
        depth_heights[d] = max(depth_heights.get(d, 0), info['height'])

    # Existing label gaps (vertical)
    builtin_vertical_gaps = _compute_label_vertical_gaps(relations, depths, font_size)

    # Prepare combined gaps
    def vgap(d):
        base = vertical_spacing
        extra1 = builtin_vertical_gaps.get(d, 0)
        extra2 = 0
        if vertical_gaps:
            extra2 = vertical_gaps.get(d, 0)
        return base + max(extra1, extra2)

    max_depth = max(depths.values()) if depths else 0

    depth_tops = {}
    current_y = margin
    for d in range(max_depth + 1):
        depth_tops[d] = current_y
        row_h = depth_heights.get(d, 0)
        current_y += row_h + vgap(d)

    # Assign y positions
    for name, info in layout.items():
        d = info['depth']
        info['y'] = depth_tops[d]

    return layout, char_width


def _connected_components(classes, relations):
    """
    Returns connected components as sets of class names.
    """
    adj = {c.name: set() for c in classes}
    for r in relations:
        if r.source in adj and r.target in adj:
            adj[r.source].add(r.target)
            adj[r.target].add(r.source)

    visited = set()
    comps = []

    for node in adj:
        if node in visited:
            continue
        stack = [node]
        comp = {node}
        visited.add(node)
        while stack:
            cur = stack.pop()
            for nxt in adj[cur]:
                if nxt not in visited:
                    visited.add(nxt)
                    comp.add(nxt)
                    stack.append(nxt)
        comps.append(comp)
    return comps


# ======================================================
# Geometry & collision helpers
# ======================================================

def _bezier_vertical(x1, y1, x2, y2, curve=40):
    mid_y = (y1 + y2) / 2
    return f'M{x1},{y1} Q{(x1+x2)/2},{mid_y-curve} {x2},{y2}'


def _boxes_collide(box, boxes, pad=2.0, eps=0.5):
    """
    Robust AABB collision check (touching counts as collision).
    Used for edge-label vs. node/label collision.
    """
    x1, y1, x2, y2 = box
    x1 -= pad
    y1 -= pad
    x2 += pad
    y2 += pad

    for ax1, ay1, ax2, ay2 in boxes:
        ax1 -= pad
        ay1 -= pad
        ax2 += pad
        ay2 += pad
        if (x1 <= ax2 + eps and x2 >= ax1 - eps and
            y1 <= ay2 + eps and y2 >= ay1 - eps):
            return True
    return False


def _line_intersects_box(x1, y1, x2, y2, box):
    """
    Check if a line segment intersects or passes through a box.
    Used to keep straight-edge labels off the edge line.
    """
    bx1, by1, bx2, by2 = box

    # Quick reject if line is entirely to one side
    if max(x1, x2) < bx1 or min(x1, x2) > bx2:
        return False
    if max(y1, y2) < by1 or min(y1, y2) > by2:
        return False

    def seg_intersects(ax, ay, bx, by, cx, cy, dx, dy):
        def ccw(p, q, r):
            return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])
        A, B = (ax, ay), (bx, by)
        C, D = (cx, cy), (dx, dy)
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    return (
        seg_intersects(x1, y1, x2, y2, bx1, by1, bx2, by1) or  # top
        seg_intersects(x1, y1, x2, y2, bx1, by2, bx2, by2) or  # bottom
        seg_intersects(x1, y1, x2, y2, bx1, by1, bx1, by2) or  # left
        seg_intersects(x1, y1, x2, y2, bx2, by1, bx2, by2)     # right
    )


def _place_straight_edge_label(
    x1, y1, x2, y2,
    source_box, target_box,
    lines,
    char_width,
    font_size,
    node_boxes,
    placed_label_boxes,
):
    """
    Place label for a straight edge:
    - vertically between the source and target boxes,
    - horizontally to the right of the edge,
    - shifted right until it doesn't overlap the edge or nodes/labels.
    """
    sx, sy, sw, sh = source_box
    tx, ty, tw, th = target_box

    # Vertical midpoint between boxes
    source_bottom = sy + sh
    target_top = ty
    cy = (source_bottom + target_top) / 2

    # Edge direction & right-hand perpendicular
    vx = x2 - x1
    vy = y2 - y1
    L = (vx * vx + vy * vy) ** 0.5 or 1.0
    ex = vx / L
    ey = vy / L
    px = ey
    py = -ex

    # Text dimensions
    fs = font_size - 2
    lh = fs
    max_chars = max(len(line) for line in lines) if lines else 0
    lw = max_chars * char_width
    lh_total = lh * max(1, len(lines))

    # Initial candidate: to the right of the edge's midpoint
    base_cx = (x1 + x2) / 2
    offset = 20.0

    for _ in range(12):
        cx = base_cx + px * offset
        box = (
            cx - lw / 2,
            cy - lh_total / 2,
            cx + lw / 2,
            cy + lh_total / 2,
        )
        if (not _line_intersects_box(x1, y1, x2, y2, box)
            and not _boxes_collide(box, node_boxes.values())
            and not _boxes_collide(box, placed_label_boxes)):
            placed_label_boxes.append(box)
            return cx, cy
        offset += 6.0

    # Fallback: use last attempt
    cx = base_cx + px * offset
    box = (
        cx - lw / 2,
        cy - lh_total / 2,
        cx + lw / 2,
        cy + lh_total / 2,
    )
    placed_label_boxes.append(box)
    return cx, cy


def _place_curved_edge_label(
    x1, y1, x2, y2,
    curve_amount,
    lines,
    char_width,
    font_size,
    node_boxes,
    placed_label_boxes,
):
    """
    Place label near a quadratic Bezier curve:
        P0 = (x1, y1)
        C  = ((x1+x2)/2, mid_y - curve_amount)
        P2 = (x2, y2)

    - Evaluate point & tangent at t=0.5
    - Offset along right-hand normal
    - Iteratively push outward until no collisions with nodes/labels.
    """
    mid_y = (y1 + y2) / 2
    cx_ctrl = (x1 + x2) / 2
    cy_ctrl = mid_y - curve_amount

    # Point on curve at t=0.5
    bx = 0.25 * x1 + 0.5 * cx_ctrl + 0.25 * x2
    by = 0.25 * y1 + 0.5 * cy_ctrl + 0.25 * y2

    # Tangent B'(0.5)
    tx = (cx_ctrl - x1) + (x2 - cx_ctrl)
    ty = (cy_ctrl - y1) + (y2 - cy_ctrl)
    L = (tx * tx + ty * ty) ** 0.5 or 1.0
    ex = tx / L
    ey = ty / L

    # Right-hand normal
    px = ey
    py = -ex

    # Label size
    fs = font_size - 2
    lh = fs
    max_chars = max(len(line) for line in lines) if lines else 0
    lw = max_chars * char_width
    lh_total = lh * max(1, len(lines))

    offset = 20.0

    for _ in range(12):
        cx = bx + px * offset
        cy = by + py * offset

        box = (
            cx - lw / 2,
            cy - lh_total / 2,
            cx + lw / 2,
            cy + lh_total / 2,
        )

        if (
            not _boxes_collide(box, node_boxes.values())
            and not _boxes_collide(box, placed_label_boxes)
        ):
            placed_label_boxes.append(box)
            return cx, cy

        offset += 6.0

    # Fallback
    cx = bx + px * offset
    cy = by + py * offset
    placed_label_boxes.append(box)
    return cx, cy


def _compute_label_horizontal_gaps(
    relations,
    depths,
    layout,
    font_size,
    char_width
):
    """
    Compute additional *horizontal spacing* needed at each depth level to
    accommodate labels to the right of edges.

    Returns: { depth : required_extra_horizontal_spacing }
    """
    fs = font_size - 2  # label font size
    gaps = {}

    for r in relations:
        if not r.label:
            continue

        s_depth = depths.get(r.source)
        t_depth = depths.get(r.target)
        if s_depth is None or t_depth is None:
            continue

        # We only worry about downward edges (parent → child)
        depth = min(s_depth, t_depth)

        # ---- Label width ----
        lines = r.label.split("\n")
        max_chars = max(len(line) for line in lines)
        label_width = max_chars * char_width + 20  # some padding

        # ---- Approximate label position ----
        s_info = layout[r.source]
        t_info = layout[r.target]

        # Approx edge midpoint
        x_mid = (
            (s_info['x'] + s_info['width'] / 2)
            + (t_info['x'] + t_info['width'] / 2)
        ) / 2

        # All labels are placed to the "right-hand side"
        # But we don't know px yet — so approximate the horizontal demand
        # as requiring label_width space *right* of x_mid.
        needed_extra = label_width + 30  # 30px margin

        # Accumulate the maximum padding needed at this depth
        gaps[depth] = max(gaps.get(depth, 0), needed_extra)

    return gaps


def _compute_label_vertical_gaps(relations, depths, font_size):
    """
    Compute extra vertical gap needed between depth d and d+1
    based on label heights of edges crossing that boundary.
    """
    fs = font_size - 2  # label font size
    gaps = {}           # depth -> extra gap

    for r in relations:
        if not r.label:
            continue

        s_depth = depths.get(r.source)
        t_depth = depths.get(r.target)
        if s_depth is None or t_depth is None:
            continue

        # Only consider edges going "downward" in the DAG
        if t_depth <= s_depth:
            continue

        # We treat this label as living between the shallower and next level
        d = s_depth  # usual case: child at s_depth+1

        lines = r.label.split('\n')
        label_height = fs * max(1, len(lines))

        # Some padding so label isn't cramped vertically
        needed_gap = label_height + 12

        if d in gaps:
            gaps[d] = max(gaps[d], needed_gap)
        else:
            gaps[d] = needed_gap

    return gaps


# ======================================================
# Main renderer
# ======================================================

def render_svg_string(
    classes: List[UMLClass],
    relations: List[UMLRelation],
    *,
    font_size=14,
    font_family='DejaVu Sans, Arial, sans-serif',
    line_height=None,
    vertical_spacing=80,
    horizontal_spacing=60,
    margin=40,
) -> str:

    if line_height is None:
        line_height = int(font_size * 1.4)

    # --------------------------------------------------
    # PASS 1: Initial layout (no label-aware gaps)
    # --------------------------------------------------
    layout1, char_width = _layout_tree(
        classes, relations,
        font_size,
        vertical_spacing,
        horizontal_spacing,
        margin,
        line_height,
    )

    # Extract node depths from the first pass
    depths = {name: info['depth'] for name, info in layout1.items()}

    # --------------------------------------------------
    # COMPUTE HORIZONTAL & VERTICAL GAPS FOR PASS 2
    # --------------------------------------------------

    # Vertical gaps (built-in)
    builtin_vertical_gaps = _compute_label_vertical_gaps(
        relations, depths, font_size
    )

    # Horizontal gaps (new)
    horizontal_gaps = _compute_label_horizontal_gaps(
        relations,
        depths,
        layout1,
        font_size,
        char_width,
    )

    # --------------------------------------------------
    # PASS 2: Final label-aware layout
    # --------------------------------------------------
    layout, char_width = _layout_tree(
        classes,
        relations,
        font_size,
        vertical_spacing,
        horizontal_spacing,
        margin,
        line_height,
        horizontal_gaps=horizontal_gaps,
        vertical_gaps=builtin_vertical_gaps,
    )

    # --------------------------------------------------
    # Connected-component highlighting
    # --------------------------------------------------
    components = _connected_components(classes, relations)
    main = max(components, key=len) if components else set()
    is_disconnected = {c.name: (c.name not in main) for c in classes}

    # --------------------------------------------------
    # SVG canvas size
    # --------------------------------------------------
    width  = max(info['x'] + info['width']  + margin for info in layout.values())
    height = max(info['y'] + info['height'] + margin for info in layout.values())

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'font-family="{html.escape(font_family)}" font-size="{font_size}">',

        """
        <defs>
          <marker id="arrow" markerWidth="10" markerHeight="10"
                  refX="9" refY="5" orient="auto">
            <polygon points="0,0 10,5 0,10" fill="black" />
          </marker>
        </defs>

        <style>
          .edge-line { stroke:#888; stroke-width:1; transition:stroke 0.15s, stroke-width 0.15s; }
          .edge-label { fill:#888; transition:fill 0.15s, font-weight 0.15s; }

          .edge-group:hover .edge-label {
            fill:#000;
            font-weight:bold;
          }

          .edge-group:hover .edge-line {
            stroke:#444;
            stroke-width:2;
          }
        </style>
        """
    ]

    # --------------------------------------------------
    # Prepare node boxes for collision detection
    # --------------------------------------------------
    node_boxes = {
        c.name: (
            layout[c.name]['x'],
            layout[c.name]['y'],
            layout[c.name]['x'] + layout[c.name]['width'],
            layout[c.name]['y'] + layout[c.name]['height'],
        )
        for c in classes
    }
    placed_label_boxes = []

    # --------------------------------------------------
    # Build full children map for determining straight edges
    # --------------------------------------------------
    children, _ = _build_graph(classes, relations)

    # --------------------------------------------------
    # RENDER EDGES
    # --------------------------------------------------
    for r in relations:
        if r.source not in layout or r.target not in layout:
            continue

        s_info = layout[r.source]
        t_info = layout[r.target]

        sx, sy, sw, sh = s_info['x'], s_info['y'], s_info['width'], s_info['height']
        tx, ty, tw, th = t_info['x'], t_info['y'], t_info['width'], t_info['height']

        s_center = sx + sw / 2
        t_center = tx + tw / 2
        dx = t_center - s_center
        side_limit = sw * 0.2

        # Source exit
        if dx > side_limit:
            x1 = sx + sw
            y1 = sy + sh / 2
        elif dx < -side_limit:
            x1 = sx
            y1 = sy + sh / 2
        else:
            x1 = s_center
            y1 = sy + sh

        # Target entry (top center)
        x2 = tx + tw / 2
        y2 = ty

        # Edge direction → right-hand normal
        vx = x2 - x1
        vy = y2 - y1
        L = (vx * vx + vy * vy) ** 0.5 or 1.0
        ex = vx / L
        ey = vy / L
        px = ey
        py = -ex

        parts.append(f'<g class="edge-group edge-r-{r.source}-{r.target}">')

        # Straight vs. Bezier
        chs = children[r.source]
        if len(chs) == 3:
            # Only middle child gets straight edge
            mid = sorted(
                ((layout[ch]['x'] + layout[ch]['width'] / 2, ch) for ch in chs),
                key=lambda p: p[0]
            )[1][1]
            is_straight = (r.target == mid)
        else:
            is_straight = (len(chs) == 1)

        if is_straight:
            parts.append(
                f'<line class="edge-line" x1="{x1}" y1="{y1}" '
                f'x2="{x2}" y2="{y2}" marker-end="url(#arrow)" />'
            )
        else:
            curve_amount = 40
            path = _bezier_vertical(x1, y1, x2, y2, curve_amount)
            parts.append(
                f'<path class="edge-line" d="{path}" fill="none" '
                f'marker-end="url(#arrow)" />'
            )

        # ----------------------------
        # Edge Labels
        # ----------------------------
        if r.label:
            lines = r.label.split('\n')
            if is_straight:
                cx, cy = _place_straight_edge_label(
                    x1, y1, x2, y2,
                    (sx, sy, sw, sh),
                    (tx, ty, tw, th),
                    lines,
                    char_width,
                    font_size,
                    node_boxes,
                    placed_label_boxes,
                )
            else:
                curve_amount = 40
                cx, cy = _place_curved_edge_label(
                    x1, y1, x2, y2,
                    curve_amount,
                    lines,
                    char_width,
                    font_size,
                    node_boxes,
                    placed_label_boxes,
                )

            fs = font_size - 2
            lh = fs
            parts.append(
                f'<text class="edge-label" x="{cx}" y="{cy}" '
                f'text-anchor="middle" font-size="{fs}">'
            )
            parts.append(html.escape(lines[0]))
            for line in lines[1:]:
                parts.append(f'<tspan x="{cx}" dy="{lh}">{html.escape(line)}</tspan>')
            parts.append('</text>')

        # Multiplicities
        along = 10
        perp = 12

        if r.source_multiplicity:
            sxm = x1 + ex * along
            sym = y1 + ey * along
            mx = sxm + px * perp
            my = sym + py * perp
            parts.append(
                f'<text class="edge-label" x="{mx}" y="{my}" '
                f'text-anchor="middle" font-size="{font_size - 2}">'
                f'{html.escape(r.source_multiplicity)}</text>'
            )

        if r.target_multiplicity:
            txm = x2 - ex * along
            tym = y2 - ey * along
            mx = txm + px * perp
            my = tym + py * perp
            parts.append(
                f'<text class="edge-label" x="{mx}" y="{my}" '
                f'text-anchor="middle" font-size="{font_size - 2}">'
                f'{html.escape(r.target_multiplicity)}</text>'
            )

        parts.append('</g>')

    # --------------------------------------------------
    # RENDER NODES
    # --------------------------------------------------
    for cls in classes:
        info = layout[cls.name]
        x, y = info['x'], info['y']
        w, h = info['width'], info['height']

        fill = cls.style.get('fill', '#f5f5f5')
        base_color = cls.style.get('text', '#000')

        if is_disconnected[cls.name]:
            stroke = cls.style.get('stroke', 'red')
            sw = cls.style.get('stroke_width', '2')
        else:
            stroke = cls.style.get('stroke', '#000')
            sw = cls.style.get('stroke_width', '1')

        parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'rx="4" ry="4" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" />'
        )

        # Class name
        cx = x + w / 2
        cy = y + 2 + line_height
        parts.append(
            f'<text x="{cx}" y="{cy}" text-anchor="middle" '
            f'font-weight="bold" fill="{base_color}">{html.escape(cls.name)}</text>'
        )

        # Header divider
        divider = y + 10 + line_height + 3
        if info['attr_lines'] or info['method_lines']:
            parts.append(
                f'<line x1="{x}" y1="{divider}" x2="{x+w}" y2="{divider}" stroke="{stroke}" />'
            )

        cy = divider + line_height

        # Attributes
        for entry in cls.attributes:
            text, sty = _parse_text_entry(entry)
            weight = sty.get('weight', 'normal')
            style  = sty.get('style', 'normal')
            color  = sty.get('color', base_color)
            size   = sty.get('size')
            fam    = sty.get('family')
            anchor = sty.get('anchor', 'start')

            bits = []
            if size:
                bits.append(f'font-size="{size}"')
            if fam:
                bits.append(f'font-family="{html.escape(fam)}"')

            parts.append(
                f'<text x="{x+10}" y="{cy}" {" ".join(bits)} '
                f'font-weight="{weight}" font-style="{style}" '
                f'text-anchor="{anchor}" fill="{color}">{html.escape(text)}</text>'
            )
            cy += line_height

        # Divider between attributes and methods
        if info['attr_lines'] and info['method_lines']:
            mid = cy - line_height / 2
            parts.append(
                f"<line x1='{x}' y1='{mid}' x2='{x+w}' y2='{mid}' stroke='{stroke}' />"
            )
            cy = mid + line_height

        # Methods
        for entry in cls.methods:
            text, sty = _parse_text_entry(entry)
            weight = sty.get('weight', 'normal')
            style  = sty.get('style', 'normal')
            color  = sty.get('color', base_color)
            size   = sty.get('size')
            fam    = sty.get('family')
            anchor = sty.get('anchor', 'start')

            bits = []
            if size:
                bits.append(f"font-size='{size}'")
            if fam:
                bits.append(f"font-family='{html.escape(fam)}'")

            parts.append(
                f'<text x="{x+10}" y="{cy}" {" ".join(bits)} '
                f'font-weight="{weight}" font-style="{style}" '
                f'text-anchor="{anchor}" fill="{color}">{html.escape(text)}</text>'
            )
            cy += line_height

    parts.append('</svg>')
    return '\n'.join(parts)


# ======================================================
# File writer
# ======================================================

def render_svg(classes, relations, filename, **kwargs):
    """
    Convenience wrapper: write SVG to a file.
    """
    svg = render_svg_string(classes, relations, **kwargs)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(svg)

    print(f'[pyuml2svg] Saved UML diagram to {filename}')
