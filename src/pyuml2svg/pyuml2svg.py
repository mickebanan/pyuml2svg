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
from importlib.resources import files
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


def _load_asset(filename):
    return (files('pyuml2svg.assets') / filename).read_text(encoding='utf-8')


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

        # key fix: children must start placing at start_x
        local_next_x = start_x
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
                # Leaf: assign new horizontal slot
                d = layout[node]['depth']
                gap = depth_gap(d)
                layout[node]['x'] = local_next_x
                local_next_x += layout[node]['width'] + gap

        dfs(root)

        # Normalize subtree so that the leftmost node is exactly at start_x
        min_x = min(layout[n]['x'] for n in nodes)
        shift = start_x - min_x
        for n in nodes:
            layout[n]['x'] += shift

        rightmost = max(layout[n]['x'] + layout[n]['width'] for n in nodes)
        next_x = rightmost + horizontal_spacing
        visited.update(nodes)

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


def _compute_exit_point(sx, sy, sw, sh, idx, N):
    """
    Compute the exit point for child idx out of N
    using the full exit region:
    - lower half of left side
    - entire bottom
    - lower half of right side
    """
    # Coordinates
    left_x  = sx
    right_x = sx + sw
    mid_y   = sy + sh * 0.5
    bot_y   = sy + sh

    # Parameter t in [0,1]
    t = (idx + 0.5) / N

    # 4 segments of length 0.25:
    # [0.00,0.25]: left side (mid_y → bot_y)
    # [0.25,0.50]: bottom-left → bottom-center
    # [0.50,0.75]: bottom-center → bottom-right
    # [0.75,1.00]: right side (bot_y → mid_y)
    if t < 0.25:
        # left vertical segment
        local = t / 0.25
        x = left_x
        y = mid_y + (bot_y - mid_y) * local
    elif t < 0.50:
        # bottom-left → bottom-center
        local = (t - 0.25) / 0.25
        x = left_x + (right_x - left_x) * 0.5 * local
        y = bot_y
    elif t < 0.75:
        # bottom-center → bottom-right
        local = (t - 0.50) / 0.25
        x = left_x + sw * 0.5 + sw * 0.5 * local
        y = bot_y
    else:
        # right vertical segment
        local = (t - 0.75) / 0.25
        x = right_x
        y = bot_y - (bot_y - mid_y) * local

    return x, y


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


def _orthogonal_path_lanes(
    x1, y1, x2, y2,
    sx, sy, sw, sh,
    idx, total,
    *,
    gap=10,
    lane_spacing=10,
):
    """
    Simple, mostly overlap-free orthogonal routing.

    - Bottom exits: short 'down → horizontal → down'.
      Each sibling gets a slightly different horizontal band (y_mid).

    - Side exits: short 'side → down → side'.
      Each sibling gets a slightly different outward x_clear.

    Goal: keep paths short and monotonic; reduce coincident segments.
    """

    bottom_y = sy + sh
    left_x   = sx
    right_x  = sx + sw

    # Which face are we closest to?
    dist_bottom = abs(y1 - bottom_y)
    dist_left   = abs(x1 - left_x)
    dist_right  = abs(x1 - right_x)

    is_bottom_exit = dist_bottom <= min(dist_left, dist_right) + 1e-3

    # Small per-sibling offset: centered around 0
    if total > 1:
        mid = (total - 1) / 2.0
        sib_offset = (idx - mid) * lane_spacing
    else:
        sib_offset = 0.0

    # ------------------------------------------------------------
    # CASE 1: Bottom exit → down → horizontal → down
    # ------------------------------------------------------------
    if is_bottom_exit:
        # Single horizontal band between parent and child, slightly shifted per sibling
        base_mid = (y1 + y2) / 2.0
        y_mid = base_mid + sib_offset  # tiny vertical separation for siblings

        # Keep y_mid between y1 and y2 so we don't bend upwards or overshoot
        y_min, y_max = sorted((y1, y2))
        y_mid = max(y_min + gap, min(y_max - gap, y_mid))

        return (
            f"M{x1},{y1} "
            f"L{x1},{y_mid} "
            f"L{x2},{y_mid} "
            f"L{x2},{y2}"
        )

    # ------------------------------------------------------------
    # CASE 2: Side exit → side → down → side
    # ------------------------------------------------------------

    # Determine side: left or right
    if dist_left < dist_right:
        side_dir = -1  # exiting from left side
    else:
        side_dir = +1  # exiting from right side

    # Step 1: a short horizontal move *away* from the box.
    # Use sib_offset to spread siblings side-by-side.
    x_clear = x1 + side_dir * (gap + abs(sib_offset))

    # Step 2: vertical mid-band between y1 and y2 (no loops)
    base_mid = (y1 + y2) / 2.0
    y_mid = base_mid  # we could add tiny per-sibling tweak if needed

    # Ensure we don't bend upwards past start/target
    y_min, y_max = sorted((y1, y2))
    y_mid = max(y_min + gap, min(y_max - gap, y_mid))

    return (
        f"M{x1},{y1} "
        f"L{x_clear},{y1} "
        f"L{x_clear},{y_mid} "
        f"L{x2},{y_mid} "
        f"L{x2},{y2}"
    )


def _place_straight_edge_label(
    x1, y1, x2, y2,
    source_box, target_box,
    lines,
    char_width,
    font_size,
    node_boxes,
    placed_label_boxes,
    margin,
    forced_cy=None
):
    """
    Place label for a straight edge:
    - vertically between the source and target boxes,
    - horizontally to the right of the edge,
    - shifted right until it doesn't overlap the edge or nodes/labels,
    - and not allowed to spill left of `margin`.
    """
    sx, sy, sw, sh = source_box
    tx, ty, tw, th = target_box

    # Vertical midpoint between boxes
    source_bottom = sy + sh
    target_top = ty
    if forced_cy is not None:
        cy = forced_cy
    else:
        # current midpoint logic
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

    base_cx = (x1 + x2) / 2
    offset = 20.0

    for _ in range(12):
        cx = base_cx + px * offset

        # clamp so label doesn't spill left of margin
        if lw > 0 and cx - lw / 2 < margin:
            cx = margin + lw / 2

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

    # Fallback: last attempt, clamp if needed
    cx = base_cx + px * offset
    if lw > 0 and cx - lw / 2 < margin:
        cx = margin + lw / 2
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
    margin,
):
    """
    Improved curved-edge label placement:
    - samples multiple t positions along the curve
    - chooses left or right normal based on available space
    - pushes outward until a free spot is found
    - clamps so the label does not spill left of `margin`
    """

    # --- curve geometry (must match _bezier_vertical) ---
    mid_y = (y1 + y2) / 2
    cx_ctrl = (x1 + x2) / 2
    cy_ctrl = mid_y - curve_amount

    def bezier_point(t: float):
        """Quadratic Bezier point at parameter t."""
        u = 1.0 - t
        bx = u * u * x1 + 2 * u * t * cx_ctrl + t * t * x2
        by = u * u * y1 + 2 * u * t * cy_ctrl + t * t * y2
        return bx, by

    def bezier_tangent(t: float):
        """Unit tangent of quadratic Bezier at parameter t."""
        # derivative of quadratic Bezier:
        tx = 2 * (1 - t) * (cx_ctrl - x1) + 2 * t * (x2 - cx_ctrl)
        ty = 2 * (1 - t) * (cy_ctrl - y1) + 2 * t * (y2 - cy_ctrl)
        L = (tx * tx + ty * ty) ** 0.5 or 1.0
        return tx / L, ty / L

    # --- label size ---
    fs = font_size - 2
    lh = fs
    max_chars = max(len(line) for line in lines) if lines else 0
    lw = max_chars * char_width
    lh_total = lh * max(1, len(lines))

    # --- candidate t positions along the curve ---
    t_candidates = [0.35, 0.5, 0.65]

    best = None
    best_penalty = float("inf")

    for t in t_candidates:
        bx, by = bezier_point(t)
        txn, tyn = bezier_tangent(t)

        # left & right normals for a unit tangent
        # (rotate 90° CCW / CW)
        left_normal  = (-tyn, txn)
        right_normal = ( tyn, -txn)

        for nx, ny in (left_normal, right_normal):
            offset = 20.0
            for _ in range(30):
                cx = bx + nx * offset
                cy = by + ny * offset

                # clamp so label doesn't spill left of margin
                if lw > 0 and cx - lw / 2 < margin:
                    cx = margin + lw / 2

                label_box = (
                    cx - lw / 2,
                    cy - lh_total / 2,
                    cx + lw / 2,
                    cy + lh_total / 2,
                )

                if (
                    not _boxes_collide(label_box, node_boxes.values())
                    and not _boxes_collide(label_box, placed_label_boxes)
                ):
                    # simple penalty: prefer smaller offset and t near 0.5
                    penalty = offset + abs(t - 0.5) * 20.0
                    if penalty < best_penalty:
                        best_penalty = penalty
                        best = (cx, cy, label_box)
                    break

                offset += 8.0

    # Fallback: sit on the curve midpoint if nothing worked
    if best is None:
        t = 0.5
        bx, by = bezier_point(t)
        cx, cy = bx, by
        if lw > 0 and cx - lw / 2 < margin:
            cx = margin + lw / 2
        label_box = (
            cx - lw / 2,
            cy - lh_total / 2,
            cx + lw / 2,
            cy + lh_total / 2,
        )
        best = (cx, cy, label_box)

    cx, cy, box = best
    placed_label_boxes.append(box)
    return cx, cy


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
    edge_style="auto",
) -> str:

    if line_height is None:
        line_height = int(font_size * 1.4)

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------
    layout, char_width = _layout_tree(
        classes, relations, font_size,
        vertical_spacing, horizontal_spacing,
        margin, line_height
    )

    components = _connected_components(classes, relations)
    main = max(components, key=len) if components else set()
    is_disconnected = {c.name: (c.name not in main) for c in classes}

    height = max(info['y'] + info['height'] + margin for info in layout.values())

    parts = [
        """
    <!-- ================================================================ -->
    <!--                   UML Arrowhead Definitions                      -->
    <!-- ================================================================ -->
    <defs>
      <!-- Generalization (inheritance): hollow triangle -->
      <marker id="inheritance" markerWidth="12" markerHeight="12"
              refX="10" refY="6" orient="auto">
        <path d="M0,0 L12,6 L0,12 Z" fill="white" stroke="black" />
      </marker>

      <!-- Realization: hollow triangle, dashed line -->
      <marker id="realization" markerWidth="12" markerHeight="12"
              refX="10" refY="6" orient="auto">
        <path d="M0,0 L12,6 L0,12 Z" fill="white" stroke="black" />
      </marker>

      <!-- Composition: filled diamond -->
      <marker id="composition" markerWidth="12" markerHeight="12"
              refX="12" refY="6" orient="auto">
        <path d="M12,6 L6,0 L0,6 L6,12 Z" fill="black" stroke="black" />
      </marker>

      <!-- Aggregation: hollow diamond -->
      <marker id="aggregation" markerWidth="12" markerHeight="12"
              refX="12" refY="6" orient="auto">
        <path d="M12,6 L6,0 L0,6 L6,12 Z" fill="white" stroke="black" />
      </marker>

      <!-- Filled arrowhead: directed association -->
      <marker id="association" markerWidth="10" markerHeight="10"
              refX="9" refY="5" orient="auto">
        <polygon points="0,0 10,5 0,10" fill="black" />
      </marker>

      <!-- Dependency (open arrow) -->
      <marker id="dependency" markerWidth="10" markerHeight="10"
              refX="9" refY="5" orient="auto">
        <polygon points="0,0 10,5 0,10" fill="white" stroke="black" />
      </marker>
    </defs>

    <style>%(style)s</style>
    """ % {'style': _load_asset('style.css')}
    ]

    # Build child mapping
    children, _ = _build_graph(classes, relations)

    # Boxes for label collision
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
    # Nodes
    # --------------------------------------------------
    for cls in classes:

        info = layout[cls.name]
        x, y = info['x'], info['y']
        w, h = info['width'], info['height']

        fill = cls.style.get('fill', '#f5f5f5')
        base_color = cls.style.get('text', '#000')

        if is_disconnected[cls.name]:
            # Highlight disconnected nodes
            stroke = cls.style.get('stroke', 'red')
            swidth = cls.style.get('stroke_width', '2')
        else:
            stroke = cls.style.get('stroke', '#000')
            swidth = cls.style.get('stroke_width', '1')

        parts.append(f'<g id="class-{cls.name}" class="uml-class collapsible-visible">')

        # box
        parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'rx="4" ry="4" fill="{fill}" stroke="{stroke}" stroke-width="{swidth}" '
            f'onclick="toggleNode(\'{cls.name}\')" />'
        )

        # class name
        cx = x + w/2
        cy = y + 2 + line_height
        parts.append(
            f'<text x="{cx}" y="{cy}" text-anchor="middle" '
            f'font-weight="bold" fill="{base_color}">{html.escape(cls.name)}</text>'
        )

        # toggle marker (only if class has children)
        if children.get(cls.name):
            mx = x + w - 12
            my = y + 14
            parts.append(
                f'<text class="toggle-marker" id="toggle-{cls.name}" '
                f'x="{mx}" y="{my}" '
                f'onclick="toggleNode(\'{cls.name}\'); event.stopPropagation();">▼</text>'
            )

        # divider
        divider = y + 10 + line_height + 3
        if info['attr_lines'] or info['method_lines']:
            parts.append(
                f'<line x1="{x}" y1="{divider}" x2="{x+w}" y2="{divider}" stroke="{stroke}"/>'
            )

        cy = divider + line_height

        # attributes
        for entry in cls.attributes:
            text, sty = _parse_text_entry(entry)
            weight = sty.get('weight', 'normal')
            style  = sty.get('style', 'normal')
            color  = sty.get('color', base_color)
            size   = sty.get('size')
            fam    = sty.get('family')
            anchor = sty.get('anchor', 'start')

            bits = []
            if size: bits.append(f'font-size="{size}"')
            if fam:  bits.append(f'font-family="{html.escape(fam)}"')

            parts.append(
                f'<text x="{x+10}" y="{cy}" {" ".join(bits)} '
                f'font-weight="{weight}" font-style="{style}" '
                f'text-anchor="{anchor}" fill="{color}">{html.escape(text)}</text>'
            )
            cy += line_height

        # divider between attributes and methods
        if info['attr_lines'] and info['method_lines']:
            mid = cy - line_height/2
            parts.append(
                f"<line x1='{x}' y1='{mid}' x2='{x+w}' y2='{mid}' stroke='{stroke}' />"
            )
            cy = mid + line_height

        # methods
        for entry in cls.methods:
            text, sty = _parse_text_entry(entry)
            weight = sty.get('weight', 'normal')
            style  = sty.get('style', 'normal')
            color  = sty.get('color', base_color)
            size   = sty.get('size')
            fam    = sty.get('family')
            anchor = sty.get('anchor', 'start')

            bits = []
            if size: bits.append(f"font-size='{size}'")
            if fam:  bits.append(f"font-family='{html.escape(fam)}'")

            parts.append(
                f'<text x="{x+10}" y="{cy}" {" ".join(bits)} '
                f'font-weight="{weight}" font-style="{style}" '
                f'text-anchor="{anchor}" fill="{color}">{html.escape(text)}</text>'
            )
            cy += line_height

        parts.append("</g>")  # end class group
    # --------------------------------------------------
    # Edges
    # --------------------------------------------------
    for r in relations:
        if r.source not in layout or r.target not in layout:
            continue

        s = layout[r.source]
        t = layout[r.target]

        sx, sy, sw, sh = s['x'], s['y'], s['width'], s['height']
        tx, ty, tw, th = t['x'], t['y'], t['width'], t['height']

        # Sibling ordering
        siblings = children[r.source]
        sibs = sorted(
            siblings,
            key=lambda name: layout[name]['x'] + layout[name]['width'] / 2
        )
        idx = sibs.index(r.target)
        num = len(sibs)

        # NEW: label vertical offset between children (unchanged)
        if num > 1:
            source_bottom = sy + sh
            children_top = min(layout[ch]['y'] for ch in siblings)
            span = children_top - source_bottom
            label_y_offset = source_bottom + (span / (num + 1)) * (idx + 1)
        else:
            label_y_offset = None

        # --------------------------------------------------
        # Compute exit point
        # --------------------------------------------------
        x1, y1 = _compute_exit_point(sx, sy, sw, sh, idx, num)

        # Preserve original point for multiplicities
        orig_x1, orig_y1 = x1, y1

        # --------------------------------------------------
        # UML arrowhead logic (including diamond handling)
        # --------------------------------------------------
        kind = (r.kind or "association").lower()
        marker_start = None
        marker_end   = None

        if kind == "inheritance":
            marker_end = "inheritance"

        elif kind == "realization":
            marker_end = "realization"

        elif kind == "composition":
            marker_start = "composition"

        elif kind == "aggregation":
            marker_start = "aggregation"

        elif kind == "dependency":
            marker_end = "dependency"

        elif kind == "directed-association":
            marker_end = "association"

        # everything else: no markers

        # --------------------------------------------------
        # Nudge start point outward for start markers (diamonds)
        # --------------------------------------------------
        if marker_start:
            # vector from box center to exit point
            cx = sx + sw / 2
            cy = sy + sh / 2
            vx = x1 - cx
            vy = y1 - cy
            L = (vx * vx + vy * vy)**0.5 or 1.0

            OUTSET = 12.0  # diamond offset distance

            x1 = x1 + (vx / L) * OUTSET
            y1 = y1 + (vy / L) * OUTSET

        # --------------------------------------------------
        # Compute entry point into target (x2 is clamped to box)
        # --------------------------------------------------
        ideal_x2 = x1
        x2 = max(tx + 4, min(tx + tw - 4, ideal_x2))
        y2 = ty

        # --------------------------------------------------
        # Open edge-group
        # --------------------------------------------------
        edge_kind_class = f"edge-kind-{kind}"
        parts.append(
            f'<g class="edge-group {edge_kind_class} collapsible-visible" '
            f'data-source="{r.source}" data-target="{r.target}">'
        )

        # Marker attributes
        ms = f' marker-start="url(#{marker_start})"' if marker_start else ""
        me = f' marker-end="url(#{marker_end})"'     if marker_end   else ""

        # Straight vs orthogonal vs bezier
        use_ortho = edge_style == "orthogonal"
        is_straight = abs(x1 - (sx + sw / 2)) < 1.0

        if is_straight:
            # vertical straight edge
            parts.append(
                f'<line class="edge-line" x1="{x1}" y1="{y1}" '
                f'x2="{x2}" y2="{y2}"{ms}{me} />'
            )

        elif use_ortho:
            d = _orthogonal_path_lanes(
                x1, y1, x2, y2,
                sx, sy, sw, sh,
                idx, num,
            )
            parts.append(
                f'<path class="edge-line" d="{d}" fill="none"{ms}{me} />'
            )

        else:
            d = _bezier_vertical(x1, y1, x2, y2, 40)
            parts.append(
                f'<path class="edge-line" d="{d}" fill="none"{ms}{me} />'
            )

        # --------------------------------------------------
        # Edge label placement (unchanged)
        # --------------------------------------------------
        if r.label:
            lines = r.label.split("\n")
            cy_override = label_y_offset if label_y_offset else None

            cx, cy = _place_straight_edge_label(
                x1, y1, x2, y2,
                (sx, sy, sw, sh),
                (tx, ty, tw, th),
                lines,
                char_width,
                font_size,
                node_boxes,
                placed_label_boxes,
                margin,
                forced_cy=cy_override,
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

        # --------------------------------------------------
        # Multiplicities (must use original start point!)
        # --------------------------------------------------
        if r.source_multiplicity:
            eps = 1e-3
            msy = sy + sh/2
            sl, sr = sx, sx+sw

            if abs(orig_x1 - sl) < eps and abs(orig_y1 - msy) < eps:
                mx = orig_x1 - 12; my = orig_y1
            elif abs(orig_x1 - sr) < eps and abs(orig_y1 - msy) < eps:
                mx = orig_x1 + 12; my = orig_y1
            else:
                mx = orig_x1 + 5; my = orig_y1 + 12

            parts.append(
                f'<text class="edge-label" x="{mx}" y="{my}" '
                f'font-size="{font_size-2}">{html.escape(r.source_multiplicity)}</text>'
            )

        if r.target_multiplicity:
            mx = x2 + 6
            my = ty - 6
            parts.append(
                f'<text class="edge-label" x="{mx}" y="{my}" '
                f'font-size="{font_size-2}">{html.escape(r.target_multiplicity)}</text>'
            )

        parts.append("</g>")  # close edge-group
    # --------------------------------------------------
    # Recompute width now that all labels are placed
    # --------------------------------------------------

    node_right = max(info['x'] + info['width'] for info in layout.values())
    label_right = max((box[2] for box in placed_label_boxes), default=0)
    width = max(node_right, label_right) + margin
    parts[0] %= {'width': width}

    svg_open = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'font-family="{html.escape(font_family)}" font-size="{font_size}" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink">'
    )
    parts.insert(0, svg_open)
    parts.append('<script type="text/ecmascript"><![CDATA[%(js)s]]></script>' % {'js': _load_asset('script.js')})
    parts.append("</svg>")
    return "\n".join(parts)


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
