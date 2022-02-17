from __future__ import annotations

import os
import json
from pathlib import Path
import typing as tp
from copy import deepcopy
from enum import Enum
from itertools import count


class Cell(tp.TypedDict, total=False):
    cell_type: str
    execution_count: int | None
    id: str
    metadata: dict[str, tp.Any]
    outputs: list[str]
    source: list[str]


class CellType(str, Enum):
    MARKDOWN = "markdown"
    CODE = "code"


class SlideType(str, Enum):
    SLIDE = "slide"
    SUBSLIDE = "subslide"
    FRAGMENT = "fragment"


class Notebook:
    BASE_NOTEBOOK = {
        "cells": [],
        "metadata": {
            "celltoolbar": "Slideshow",
            "kernelspec": {
                "display_name": "Python 3.8.12 64-bit ('.env-eigeneko': venv)",
                "language": "python",
                "name": "python3812jvsc74a57bd04eee3b84c16cd653c9409d307a1e0cdaeaa31ed4117853a4ee7c842a85681cd7",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    def __init__(self) -> None:
        self._dict = deepcopy(self.BASE_NOTEBOOK)
        self._id_gen = (str(x) for x in count())

    @property
    def cells(self) -> list[Cell]:
        return self._dict["cells"]

    def add_cell_markdown(
        self, source: list[str], *, slide_type: SlideType = SlideType.FRAGMENT
    ):
        cell = Cell(
            cell_type=CellType.MARKDOWN.value,
            id=next(self._id_gen),
            metadata={"slideshow": {"slide_type": slide_type.value}},
            source=source,
        )
        self.cells.append(cell)

    def add_cell_code(
        self, source: list[str], *, slide_type: SlideType = SlideType.FRAGMENT
    ):
        cell = Cell(
            cell_type=CellType.CODE.value,
            execution_count=None,
            id=next(self._id_gen),
            metadata={"slideshow": {"slide_type": slide_type.value}},
            outputs=[],
            source=source,
        )

        self.cells.append(cell)

    def save(self, fp: str) -> str:
        with open(fp, "w") as f:
            json.dump(self._dict, f, indent=2)
        return fp

    def add_lines(self, lines: tp.List[str]) -> None:
        code = False
        code_block: tp.List[str] = []
        for line in lines:
            line = line.strip("\n")
            flag = line[: line.find(" ")] if " " in line else line

            if flag == "CODE":
                code = True
                continue

            if flag == "END":
                self.add_cell_code(source=code_block)
                code_block = []
                code = False
                continue

            if code:
                code_block.append(f"{line}\n")
                continue

            if flag == "_OUTPUTS":
                path = Path("images", line[len("_OUTPUTS ") :])

                self.add_cell_markdown(f"# {path.stem}", slide_type=SlideType.SLIDE)
                for fn in [
                    "sample_processed_images.png",
                    "eigenfaces.png",
                    "Principle Components.png",
                    "iterative_components.gif",
                ]:
                    full_path = path / fn
                    self.add_cell_markdown(
                        [
                            f"## {path.stem}: {full_path.stem}\n",
                            f"""<center><img src="{full_path}"></center>\n""",
                        ],
                        slide_type=SlideType.SUBSLIDE,
                    )
                continue
            if flag == "NOTEBOOK_SUBSLIDE":
                path = Path(line[len("NOTEBOOK_SUBSLIDE ") :])
                self.add_jupyter_cells(path, slide_type=SlideType.SUBSLIDE)
                continue
            if flag == "NOTEBOOK":
                path = Path(line[len("NOTEBOOK ") :])
                self.add_jupyter_cells(path)
                continue
            if flag == "#":
                slide_type = SlideType.SLIDE
            elif flag == "##":
                slide_type = SlideType.SUBSLIDE
            else:
                slide_type = SlideType.FRAGMENT

            self.add_cell_markdown(source=[line], slide_type=slide_type)

    def add_markdown_cells(self, fp) -> None:
        with open(fp) as f:
            lines = f.readlines()

        self.add_lines(lines)

    def add_jupyter_cells(self, fp, slide_type=None) -> None:
        with open(fp) as f:
            nb_json = json.load(f)

        cells = nb_json["cells"]
        if slide_type:
            for c in nb_json["cells"]:
                if "slideshow" not in c["metadata"]:
                    c["metadata"]["slideshow"] = {}
                c["metadata"]["slideshow"]["slide_type"] = slide_type

        self.cells.extend(cells)

    def postprocess_outline(self, outline_index=1):
        # MUTATES SELF
        nums = count(1)
        outline = ["# Outline\n"]
        for cell in self.cells[1:]:
            if "slideshow" not in cell["metadata"]:
                print(f"WARNING: No slideshow in {cell}")
                continue
            if cell["metadata"]["slideshow"]["slide_type"] == SlideType.SLIDE:
                if not isinstance(cell["source"], list):
                    cell["source"] = [cell["source"]]
                title = cell["source"][0][2:]
                outline_num = next(nums)
                new_title = f"{outline_num}. {title}"
                outline.append(f"{outline_num}. {title}\n")
                cell["source"][0] = f"# {new_title}"
        outline_cell = Cell(
            cell_type=CellType.MARKDOWN.value,
            id=next(self._id_gen),
            metadata={"slideshow": {"slide_type": SlideType.SLIDE.value}},
            source=outline,
        )
        self.cells.insert(outline_index, outline_cell)


if __name__ == "__main__":
    outpath = Path("notebooks/final_notebook.ipynb")
    nb = Notebook()
    nb.add_markdown_cells("sketch/mccloskey/presentation.md")
    # nb.postprocess_outline(3)
    print(nb.save(outpath))
    print(
        f"jupyter nbconvert {outpath} --to slides; firefox {outpath.with_suffix('.slides.html')}"
    )
