# pyuml2svg
pyuml2svg is a pure Python UML Class Diagram renderer that outputs SVG with:

- Zero dependencies
- Clean top-down DAG layout
- Styled class boxes
- Hover interactivity (lines + labels)
- Multiplicity labels
- Custom per-attribute styling
- Automatically highlighted disconnected components
- CLI and library API
- Embedded-SVG ready for web apps

## Rationale
I had a need to create custom UML diagrams, but most of the UML generators simply work on your 
source code instead of arbitrary relations defined by the developer. The ones that do allow arbitrary
layouts mainly rely on external programs that, albeit powerful, requires extra dependencies that may be unsuitable in
certain circumstances. Hence pyuml2svg!

Disclaimer: large parts are coded with LLM assistance.