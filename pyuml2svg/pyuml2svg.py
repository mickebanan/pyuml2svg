"""
Pure-Python UML Class Diagram → SVG (Portrait DAG Layout)

Features:
- Portrait (top-down) DAG/tree layout with no overlaps.
- Parent centered only over unique children (DAG-safe).
- Per-line text styling via tuples:
      ('+ id: UUID', {'color': '#880000', 'weight': 'bold'})
      ["+ id: UUID", {"color": "#880000", "weight": "bold"}]  # JSON-friendly
- Grey edges + grey labels with hover effect (label only).
- Multiplicity labels.
- Default box fill = light grey (#f5f5f5) unless overridden.
- Edge labels can contain '\n' (multiline) and, for straight edges,
  are placed outside the edge using the label's bounding box, preferring
  to be to the right of the edge when possible.
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
# Utility for per-line styled text
# ======================================================

def _parse_text_entry(entry):
    """
    Accepts:
        'text'
        ('text', {style})
        ["text", {style}]  <-- JSON-compatible form
    Ensures the returned text is always a string.
    """
    if (
        isinstance(entry, (tuple, list))
        and len(entry) == 2
        and isinstance(entry[1], dict)
        and isinstance(entry[0], str)
    ):
        return entry[0], entry[1]

    return str(entry), {}


# ======================================================
# Layout helpers
# ======================================================

def _compute_box_size(cls: UMLClass, font_size: int, char_width: float, line_height: int):
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
        'name_lines': 1,
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
    queue = [(r, 0) for r in roots]

    for r in roots:
        depths[r] = 0

    while queue:
        node, d = queue.pop(0)
        for ch in children[node]:
            if depths[ch] is None or depths[ch] < d + 1:
                depths[ch] = d + 1
                queue.append((ch, d + 1))

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
):
    name_to_class = {c.name: c for c in classes}
    char_width = font_size * 0.60

    sizes = {
        name: _compute_box_size(cls, font_size, char_width, line_height)
        for name, cls in name_to_class.items()
    }

    children, parents = _build_graph(classes, relations)
    roots = _find_roots(classes, parents)
    depths = _compute_depths(roots, children, list(name_to_class.keys()))

    layout = {
        name: {
            **sizes[name],
            'x': None,
            'y': None,
            'depth': depths[name],
        }
        for name in name_to_class
    }

    next_x = margin
    visited = set()

    def layout_node(name: str):
        nonlocal next_x
        if name in visited:
            return
        visited.add(name)

        chs = children[name]
        for c in chs:
            layout_node(c)

        unique_children = [c for c in chs if len(parents[c]) == 1]

        if unique_children:
            centers = [
                layout[c]['x'] + layout[c]['width'] / 2
                for c in unique_children
            ]
            centers.sort()
            median_center = centers[len(centers) // 2]  # integer index, middle element
            layout[name]['x'] = median_center - layout[name]['width'] / 2
        else:
            if layout[name]['x'] is None:
                layout[name]['x'] = next_x
                next_x += layout[name]['width'] + horizontal_spacing

    for r in roots:
        layout_node(r)

    for name in name_to_class:
        if name not in visited:
            layout_node(name)

    min_x = min(info['x'] for info in layout.values())
    if min_x < margin:
        shift = margin - min_x
        for info in layout.values():
            info['x'] += shift

    max_h = max(info['height'] for info in layout.values())
    for name, info in layout.items():
        info['y'] = margin + info['depth'] * (max_h + vertical_spacing)

    return layout, char_width


def _connected_components(classes, relations):
    """
    Returns a list of connected components, each a set of class names.
    """
    # Build undirected adjacency
    adj = {c.name: set() for c in classes}

    for r in relations:
        if r.source in adj and r.target in adj:
            adj[r.source].add(r.target)
            adj[r.target].add(r.source)

    visited = set()
    components = []

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
        components.append(comp)
    return components


# ======================================================
# SVG helpers
# ======================================================

def _bezier_vertical(x1, y1, x2, y2, curve_amount=40):
    mid_y = (y1 + y2) / 2
    ctrl_x = (x1 + x2) / 2
    ctrl_y = mid_y - curve_amount
    return f'M{x1},{y1} Q{ctrl_x},{ctrl_y} {x2},{y2}'


# --------- label placement helper for straight edges ---------

def _boxes_collide(box, boxes):
    """Axis-aligned bbox collision checker."""
    x1, y1, x2, y2 = box
    for ax1, ay1, ax2, ay2 in boxes:
        if x1 < ax2 and x2 > ax1 and y1 < ay2 and y2 > ay1:
            return True
    return False


def _place_label_straight_edge(
    x1, y1, x2, y2,
    label_width, label_height,
    node_boxes,       # dict of name -> (x1,y1,x2,y2)
    existing_labels,  # list of (x1,y1,x2,y2)
):
    """
    Place a label for a straight edge based on the label's bbox.

    Preference order:
    1) Right of the edge
    2) Above the edge
    3) Below the edge
    4) Left of the edge

    Avoids overlap with node boxes and previously placed labels.
    """

    dx = x2 - x1
    dy = y2 - y1
    length = (dx * dx + dy * dy) ** 0.5 or 1.0

    # Unit vector along the edge
    ex = dx / length
    ey = dy / length
    # Unit perpendicular (right-hand normal)
    # Correct "right of edge" for screen coordinates (y increases downward)
    px = dy / length
    py = -dx / length

    # Edge midpoint
    mx = (x1 + x2) / 2.0
    my = (y1 + y2) / 2.0

    # Helper to build bbox from center coordinates
    def bbox_from_center(cx, cy):
        return (
            cx - label_width / 2.0,
            cy - label_height / 2.0,
            cx + label_width / 2.0,
            cy + label_height / 2.0,
        )

    # Candidate centers (not top-left), in priority order
    # Right-of-edge: move a bit along the edge and to the right side
    right_center = (
        mx + ex * 20 + px * (label_width / 2.0 + 12),
        my + ey * 20 + py * (label_width / 2.0 + 12),
    )

    # Above edge: move perpendicular in +normal direction
    above_center = (
        mx + px * (label_height + 10),
        my + py * (label_height + 10),
    )

    # Below edge: move perpendicular in -normal direction
    below_center = (
        mx - px * (label_height + 10),
        my - py * (label_height + 10),
    )

    # Left-of-edge: opposite of right-of-edge
    left_center = (
        mx - ex * 20 - px * (label_width / 2.0 + 12),
        my - ey * 20 - py * (label_width / 2.0 + 12),
    )

    candidates = [
        right_center,
        above_center,
        below_center,
        left_center,
    ]

    all_node_boxes = list(node_boxes.values())

    for cx, cy in candidates:
        box = bbox_from_center(cx, cy)
        if _boxes_collide(box, all_node_boxes):
            continue
        if _boxes_collide(box, existing_labels):
            continue
        existing_labels.append(box)
        # Return top-left coordinates for text anchor "middle"/"start" handling
        # We'll use the center's y as baseline for first line, so convert:
        x_left, y_top, _, _ = box
        # For the SVG <text>, x is center, y is baseline. We'll pass center.x, baseline.y
        return cx, cy - label_height / 2.0 + (label_height / max(1, label_height)) * 0.0

    # Fallback: use right-of-edge even if overlapping
    cx, cy = right_center
    box = bbox_from_center(cx, cy)
    existing_labels.append(box)
    return cx, cy - label_height / 2.0


# ======================================================
# Main renderers
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
    """
    Returns the SVG described by the classes and relations as a string.

    Use inside web applications or other Python code.
    """
    if line_height is None:
        line_height = int(font_size * 1.4)

    layout, char_width = _layout_tree(
        classes, relations, font_size,
        vertical_spacing, horizontal_spacing,
        margin, line_height,
    )

    # Identify disconnected components
    components = _connected_components(classes, relations)
    main_component = max(components, key=len)

    is_disconnected = {
        cls.name: (cls.name not in main_component)
        for cls in classes
    }

    width  = max(info['x'] + info['width'] + margin for info in layout.values())
    height = max(info['y'] + info['height'] + margin for info in layout.values())

    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'font-family="{html.escape(font_family)}" '
        f'font-size="{font_size}">'
    )

    parts.append('''
    <defs>
        <marker id='arrow' markerWidth='10' markerHeight='10'
                refX='9' refY='5' orient='auto'>
          <polygon points='0,0 10,5 0,10' fill='black' />
        </marker>
    </defs>

    <style>
        .edge-line {
            stroke: #888;
            stroke-width: 1;
            transition: stroke 0.15s, stroke-width 0.15s;
        }
    
        .edge-label {
            fill: #888;
            transition: fill 0.15s, font-weight 0.15s;
        }
    
        /* Hover effect for BOTH label and line */
        .edge-group:hover .edge-label {
            fill: #000;
            font-weight: bold;
        }
    
        .edge-group:hover .edge-line {
            stroke: #444;        /* slightly darker */
            stroke-width: 2;     /* slightly thicker */
        }
    </style>
    ''')

    children, _parents = _build_graph(classes, relations)
    child_counts = {n: len(chs) for n, chs in children.items()}

    # Precompute node bounding boxes for label-placement collision checks
    node_boxes = {}
    for cls in classes:
        info = layout[cls.name]
        x1 = info['x']
        y1 = info['y']
        x2 = x1 + info['width']
        y2 = y1 + info['height']
        node_boxes[cls.name] = (x1, y1, x2, y2)

    placed_label_boxes = []  # bboxes of labels (straight-edge case only for now)

    # -----------------------
    # Edges
    # -----------------------
    for r in relations:
        if r.source not in layout or r.target not in layout:
            continue

        s = layout[r.source]
        t = layout[r.target]

        # Determine better exit point based on child horizontal position
        px = s['x']
        py = s['y']
        pw = s['width']
        ph = s['height']

        cx = t['x']
        cy = t['y']
        cw = t['width']
        ch = t['height']

        pcenter = px + pw / 2
        ccenter = cx + cw / 2
        dx = ccenter - pcenter

        side_threshold = pw * 0.2  # tweakable

        if dx > side_threshold:
            # exit right
            x1 = px + pw
            y1 = py + ph / 2
        elif dx < -side_threshold:
            # exit left
            x1 = px
            y1 = py + ph / 2
        else:
            # exit bottom center
            x1 = pcenter
            y1 = py + ph
        x2 = t['x'] + t['width'] / 2
        y2 = t['y']

        group_class = f'edge-group edge-r-{r.source}-{r.target}'
        parts.append(f'<g class="{group_class}">')

        # Determine whether to draw straight or Bezier
        child_list = children[r.source]
        count = len(child_list)

        # Default: same as before (straight only if one child)
        is_straight = (count == 1)

        if count == 3:
            # Order children by horizontal position
            child_positions = [
                (layout[ch]['x'] + layout[ch]['width'] / 2, ch)
                for ch in child_list
            ]
            child_positions.sort(key=lambda x: x[0])  # left → right
            ordered_children = [name for (_, name) in child_positions]

            left, middle, right = ordered_children

            # Middle child gets straight line
            if r.target == middle:
                is_straight = True
            else:
                is_straight = False

        if is_straight:
            parts.append(
                f'<line class="edge-line" '
                f'x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                f'marker-end="url(#arrow)" />'
            )
        else:
            d = _bezier_vertical(x1, y1, x2, y2)
            parts.append(
                f'<path class="edge-line" d="{d}" fill="none" '
                f'marker-end="url(#arrow)" />'
            )

        # Edge labels
        if r.label:
            lines = str(r.label).split("\n")
            label_font_size = font_size - 2
            label_line_height = label_font_size  # same as dy we use below

            # Compute label bbox size (approx) for collision detection
            max_chars = max(len(line) for line in lines) if lines else 0
            label_width = max_chars * char_width
            label_height = max(1, len(lines)) * label_line_height

            if is_straight:
                # Use bbox-based placement for straight edges
                cx, cy = _place_label_straight_edge(
                    x1, y1, x2, y2,
                    label_width, label_height,
                    node_boxes,
                    placed_label_boxes,
                )
                lx = cx  # text anchor will be "middle"
                ly = cy
            else:
                # Original behavior for curved edges: offset from midpoint
                mx = (x1 + x2) / 2
                my = (y1 + y2) / 2
                dx = x2 - x1
                dy = y2 - y1
                length = (dx * dx + dy * dy) ** 0.5 or 1
                px = -dy / length
                py = dx / length
                offset = 10
                lx = mx + px * offset
                ly = my + py * offset

            parts.append(
                f'<text class="edge-label" x="{lx}" y="{ly}" '
                f'text-anchor="middle" font-size="{label_font_size}">'
            )
            # First line
            if lines:
                parts.append(html.escape(lines[0]))
                # Additional lines
                for line in lines[1:]:
                    parts.append(
                        f'<tspan x="{lx}" dy="{label_line_height}">'
                        f'{html.escape(line)}</tspan>'
                    )
            parts.append('</text>')

        if r.source_multiplicity:
            parts.append(
                f'<text class="edge-label" x="{x1}" y="{y1 + 15}" '
                f'text-anchor="middle" font-size="{font_size - 2}">'
                f'{html.escape(r.source_multiplicity)}</text>'
            )
        if r.target_multiplicity:
            parts.append(
                f'<text class="edge-label" x="{x2}" y="{y2 - 5}" '
                f'text-anchor="middle" font-size="{font_size - 2}">'
                f'{html.escape(r.target_multiplicity)}</text>'
            )
        parts.append('</g>')

    # -----------------------
    # Nodes
    # -----------------------
    for cls in classes:
        info = layout[cls.name]
        x, y = info['x'], info['y']
        w, h = info['width'], info['height']

        fill = cls.style.get('fill', '#f5f5f5')
        base_color = cls.style.get('text', '#000')
        if is_disconnected[cls.name]:
            stroke = cls.style.get('stroke', 'red')
            stroke_width = cls.style.get('stroke_width', '2')
        else:
            stroke = cls.style.get('stroke', '#000')
            stroke_width = cls.style.get('stroke_width', '1')

        parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'rx="4" ry="4" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{stroke_width}" />'
        )

        cx = x + w/2
        cy = y + 2 + line_height

        parts.append(
            f'<text x="{cx}" y="{cy}" text-anchor="middle" '
            f'font-weight="bold" fill="{base_color}">'
            f'{html.escape(cls.name)}</text>'
        )

        divider_y = y + 10 + line_height + 3
        if info['attr_lines'] or info['method_lines']:
            parts.append(
                f'<line x1="{x}" y1="{divider_y}" x2="{x+w}" y2="{divider_y}" '
                f'stroke="{stroke}" />'
            )

        cy = divider_y + line_height

        for entry in cls.attributes:
            text, sty = _parse_text_entry(entry)
            weight = sty.get('weight', 'normal')
            style = sty.get('style', 'normal')
            color = sty.get('color', base_color)
            size  = sty.get('size')
            fam   = sty.get('family')
            anchor= sty.get('anchor', 'start')

            stylebits = []
            if size:  stylebits.append(f'font-size="{size}"')
            if fam:   stylebits.append(f'font-family="{html.escape(fam)}"')

            parts.append(
                f'<text x="{x + 10}" y="{cy}" '
                f'{" ".join(stylebits)} '
                f'font-weight="{weight}" font-style="{style}" '
                f'text-anchor="{anchor}" fill="{color}">'
                f'{html.escape(text)}</text>'
            )
            cy += line_height

        if info['attr_lines'] and info['method_lines']:
            div2 = cy - line_height/2
            parts.append(
                f"<line x1='{x}' y1='{div2}' x2='{x + w}' y2='{div2}' "
                f"stroke='{stroke}' />"
            )
            cy = div2 + line_height

        for entry in cls.methods:
            text, sty = _parse_text_entry(entry)
            weight = sty.get('weight', 'normal')
            style = sty.get('style', 'normal')
            color = sty.get('color', base_color)
            size  = sty.get('size')
            fam   = sty.get('family')
            anchor= sty.get('anchor', 'start')

            stylebits = []
            if size:  stylebits.append(f"font-size='{size}'")
            if fam:   stylebits.append(f"font-family='{html.escape(fam)}'")

            parts.append(
                f'<text x="{x + 10}" y="{cy}" '
                f'{" ".join(stylebits)} '
                f'font-weight="{weight}" font-style="{style}" '
                f'text-anchor="{anchor}" fill="{color}">'
                f'{html.escape(text)}</text>'
            )
            cy += line_height

    parts.append('</svg>')
    return '\n'.join(parts)


# ======================================================
# File-writing wrapper for CLI use
# ======================================================

def render_svg(
    classes: List[UMLClass],
    relations: List[UMLRelation],
    filename: str,
    **kwargs,
):
    """
    Writes SVG to a file. Wrapper around render_svg_string().
    Kept for script/CLI compatibility.
    """
    svg = render_svg_string(classes, relations, **kwargs)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(svg)

    print(f'[uml_svg] Saved UML diagram to {filename}')
