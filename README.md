# pyuml2svg

**pyuml2svg** is a pure-Python library that generates **UML class diagrams as SVG** with no external dependencies.  
It performs its own layout, places labels, draws canonical UML relationship arrows, and includes interactive
collapse/expand controls directly in the SVG.

---

## Features

- Zero dependencies — pure Python + SVG
- Automatic portrait DAG layout
- Collision-aware edge labels
- Canonical UML markers (inheritance, composition, aggregation, etc.)
- Interactive collapse/expand of class hierarchies
- Disconnected-class highlighting
- Packaged CSS and JavaScript assets (no external files required)

---

## Installation

```bash
pip install pyuml2svg
```

## Quick example
```python
from pyuml2svg import UMLClass, UMLRelation, render_svg_string

classes = [
    UMLClass("Animal"),
    UMLClass("Dog"),
    UMLClass("Cat"),
    UMLClass("Spider"),
]

relations = [
    UMLRelation("Animal", "Dog", kind="inheritance"),
    UMLRelation("Animal", "Cat", kind="directed-association"),
    UMLRelation("Animal", "Spider"),
]

svg = render_svg_string(classes, relations)
with open("diagram.svg", "w", encoding="utf-8") as f:
    f.write(svg)

```
Open diagram.svg in any browser.

## UML Relationship types

Supported kind values:
- inheritance
- realization
- composition
- aggregation
- dependency
- directed-association
- association
- link (plain line)

NB. Some of these are still WIP.
