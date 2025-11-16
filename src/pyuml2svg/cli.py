# cli.py
"""
Command-line interface for pyuml2svg.

Usage examples:
    pyuml2svg diagram.json -o diagram.svg
    pyuml2svg diagram.json > out.svg
"""

import argparse
import json
import sys
from pathlib import Path

from .pyuml2svg import (
    UMLClass,
    UMLRelation,
    render_svg_string,
)


# ------------------------------------------------------
# Input loader (JSON only for now)
# ------------------------------------------------------
def load_spec(path: Path):
    text = path.read_text(encoding='utf-8')
    try:
        data = json.loads(text)
    except Exception as ex:
        raise SystemExit(f'[pyuml2svg] JSON parse error: {ex}')

    classes = []
    for c in data.get('classes', []):
        classes.append(UMLClass(
            name=c['name'],
            attributes=c.get('attributes', []),
            methods=c.get('methods', []),
            style=c.get('style', {})
        ))

    relations = []
    for r in data.get('relations', []):
        relations.append(UMLRelation(
            source=r['source'],
            target=r['target'],
            kind=r.get('kind', 'association'),
            label=r.get('label', ''),
            source_multiplicity=r.get('source_multiplicity', ''),
            target_multiplicity=r.get('target_multiplicity', '')
        ))

    return classes, relations


# ------------------------------------------------------
# Main CLI entry point
# ------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Generate an SVG UML class diagram from a JSON definition.')
    parser.add_argument('input', help='Input JSON file describing UML classes and relations.')
    parser.add_argument('-o', '--output', help='Output SVG file (default: stdout).')
    parser.add_argument('--font-size', type=int, default=14, help='Base font size used in the diagram.')
    parser.add_argument('--vertical-spacing', type=int, default=80,
                        help='Vertical spacing between classes.')
    parser.add_argument('--horizontal-spacing', type=int, default=60,
                        help='Horizontal spacing between classes.')
    parser.add_argument('--margin', type=int, default=40, help='Outer margin around the diagram.')
    args = parser.parse_args()

    # Load UML spec
    classes, relations = load_spec(Path(args.input))

    # Produce SVG text
    svg = render_svg_string(
        classes,
        relations,
        font_size=args.font_size,
        vertical_spacing=args.vertical_spacing,
        horizontal_spacing=args.horizontal_spacing,
        margin=args.margin,
    )

    # Output SVG
    if args.output:
        Path(args.output).write_text(svg, encoding='utf-8')
        print(f'[pyuml2svg] Saved UML diagram to {args.output}')
    else:
        sys.stdout.write(svg)


if __name__ == '__main__':
    main()
