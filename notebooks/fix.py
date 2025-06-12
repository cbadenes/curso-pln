import nbformat

path = "05_Transformers_GPT.ipynb"

nb = nbformat.read(path, as_version=4)

if "widgets" in nb.metadata:
    del nb.metadata["widgets"]

nbformat.write(nb, path)

