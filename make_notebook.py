"""Regenerate prelim_experiments_update.ipynb from prelim_experiments_update.py.

    python make_notebook.py

The script is what gets edited and tested; the notebook is how it actually gets run, on Colab
or JupyterHub. Keeping the copy by hand went wrong once already (the notebook was still running
the pre-TopK code), so it is generated instead.

The notebook's first few cells -- the intro, the %pip install, and the environment
configuration -- are hand-written and cannot come from a .py file, so they are carried over
from the existing notebook untouched. Everything from the CONFIG cell onward is rebuilt:
`\"\"\"## N. Title\"\"\"` statements become markdown headings, and the code between them is split
at top-level statement boundaries, never inside a def or class.
"""

import ast
import json
from pathlib import Path

HERE = Path(__file__).parent
SCRIPT = HERE / "prelim_experiments_update.py"
NOTEBOOK = HERE / "prelim_experiments_update.ipynb"

# Split a run of top-level statements once it passes this many lines. Only a readability knob:
# any grouping executes identically, since the cells run top to bottom.
TARGET_CELL_LINES = 55


# Stamped on every generated cell so the next run knows exactly where the hand-written prologue
# stops. Without it the boundary has to be guessed from the content, and guessing wrong absorbs
# a generated cell into the prologue, freezing it at the current version forever.
GENERATED = {"generated_from": SCRIPT.name}


def code_cell(source):
    return {"cell_type": "code", "execution_count": None, "metadata": dict(GENERATED),
            "outputs": [], "source": source}


def markdown_cell(source):
    return {"cell_type": "markdown", "metadata": dict(GENERATED), "source": source}


def as_source(text):
    """nbformat stores source as a list of lines, each keeping its newline except the last."""
    return text.strip("\n").splitlines(keepends=True)


def is_section_marker(node):
    """A bare string statement like \"\"\"## 4. Analyze Feature Stability\"\"\"."""
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and node.value.value.lstrip().startswith("## "))


def build_cells(lines, tree):
    """Walk the top-level statements, emitting cells and preserving every line exactly.

    Lines are attributed by span rather than reconstructed from the AST, so comments, blank
    lines and formatting survive verbatim -- a regenerated notebook that had quietly dropped
    the comments explaining the methodology would be worse than no generator at all.
    """
    cells, buffer, cursor = [], [], 0

    def flush():
        nonlocal buffer
        text = "".join(buffer)
        if text.strip():
            cells.append(code_cell(as_source(text)))
        buffer = []

    for node in tree.body:
        if is_section_marker(node):
            buffer.extend(lines[cursor:node.lineno - 1])
            flush()
            heading = node.value.value.strip()
            cells.append(markdown_cell(as_source(heading)))
            cursor = node.end_lineno
            continue

        span = lines[cursor:node.end_lineno]
        cursor = node.end_lineno
        if buffer and len(buffer) + len(span) > TARGET_CELL_LINES:
            flush()
        buffer.extend(span)

    buffer.extend(lines[cursor:])
    flush()
    return cells


def main():
    lines = SCRIPT.read_text().splitlines(keepends=True)
    tree = ast.parse("".join(lines))

    nb = json.loads(NOTEBOOK.read_text())
    tagged = [i for i, c in enumerate(nb["cells"]) if c.get("metadata", {}).get("generated_from")]
    if tagged:
        generated_start = tagged[0]
    else:
        # First run against a notebook that predates the tagging: the prologue is everything
        # before the cell defining CONFIG, which is the first thing the script contributes.
        generated_start = next(
            i for i, c in enumerate(nb["cells"]) if "CONFIG = {" in "".join(c["source"])
        )
    prologue = nb["cells"][:generated_start]
    if any(c.get("metadata", {}).get("generated_from") for c in prologue):
        raise SystemExit("A generated cell precedes the prologue; refusing to guess the split.")

    nb["cells"] = prologue + build_cells(lines, tree)
    NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")

    n_code = sum(c["cell_type"] == "code" for c in nb["cells"])
    print(f"Wrote {NOTEBOOK.name}: {len(prologue)} hand-written + "
          f"{len(nb['cells']) - len(prologue)} generated cells "
          f"({n_code} code, {len(nb['cells']) - n_code} markdown)")


if __name__ == "__main__":
    main()
