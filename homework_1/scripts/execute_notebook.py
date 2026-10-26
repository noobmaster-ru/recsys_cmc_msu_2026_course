"""Execute experiments.ipynb in a fresh kernel and preserve its outputs."""

import os
import argparse
import copy
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--section', choices=['all', 'reports', 'cf', 'ials', 'ease', 'fm', 'timesvdpp', 'final'], default='reports',
                        help='reports (default): render all committed model reports; all: also recompute EDA (download data first); individual sections preserve prior outputs and block raw test IO')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    # Prefer the kernel from the Python environment running this script.
    import sys
    kernel_path = str(Path(sys.prefix) / "share/jupyter")
    os.environ["JUPYTER_PATH"] = os.pathsep.join(
        filter(None, (kernel_path, os.environ.get("JUPYTER_PATH"))))
    path = root / "experiments.ipynb"
    notebook = nbformat.read(path, as_version=4)
    if args.section != 'all':
        tags = {'cf','ials','ease','fm','timesvdpp','final'} if args.section == 'reports' else {args.section}
        selected = [i for i,c in enumerate(notebook.cells) if tags.intersection(c.metadata.get('tags', []))]
        setup = [i for i,c in enumerate(notebook.cells) if 'shared_setup' in c.metadata.get('tags', [])]
        if not selected or len(setup) != 1:
            raise ValueError(f'Missing {args.section} section or shared setup tag')
        working = nbformat.v4.new_notebook(cells=[copy.deepcopy(notebook.cells[i]) for i in setup+selected],
                                          metadata=copy.deepcopy(notebook.metadata))
        # Enforce the user's validation-only execution constraint inside the kernel.
        protection = nbformat.v4.new_code_cell(
            'import sys\nfrom pathlib import Path\n'
            f'_protected_root = Path({str(root)!r})\n'
            '_protected_paths = {_protected_root / p for p in '
            "['data/splits/test.csv.gz', 'data/processed/interactions.csv.gz', 'data/raw/Video_Games_5.json.gz']}\n"
            'def _validation_only_io(event, arguments):\n'
            "    if event == 'open' and isinstance(arguments[0], (str, bytes)):\n"
            '        value = arguments[0].decode() if isinstance(arguments[0], bytes) else arguments[0]\n'
            '        if Path(value).resolve() in _protected_paths:\n'
            "            raise RuntimeError('Validation-only run cannot open test or full-history data')\n"
            'sys.addaudithook(_validation_only_io)\n')
        working.cells.insert(0, protection)
    else:
        working = notebook
    client = NotebookClient(working, timeout=1200, kernel_name="python3",
                            resources={"metadata": {"path": str(root)}})
    client.execute()
    if args.section != 'all':
        for source_index, executed in zip(selected, working.cells[1+len(setup):]):
            notebook.cells[source_index] = executed
    nbformat.validate(notebook)
    temporary = path.with_suffix(".ipynb.part")
    nbformat.write(notebook, temporary)
    temporary.replace(path)
    count = sum(cell.cell_type == "code" for cell in working.cells)
    print(f"Executed {count} code cells ({args.section}): {path.name}")


if __name__ == "__main__":
    main()
