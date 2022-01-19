from __future__ import annotations

from copy import deepcopy
from itertools import count
import json
import typing as tp
from enum import Enum


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


class Notebook:
    BASE_NOTEBOOK = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "overall-anatomy",
                "metadata": {},
                "source": ["my markdown 1"],
            },
            {
                "cell_type": "markdown",
                "id": "russian-confusion",
                "metadata": {},
                "source": ["my markdown 2"],
            },
        ],
        "metadata": {
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

    def add_cell_markdown(self, *, source: list[str]):
        cell = Cell(
            cell_type=CellType.MARKDOWN.value,
            id=next(self._id_gen),
            metadata={},
            source=source,
        )

    def add_cell_code(self, *, source: list[str]):
        cell = Cell(
            cell_type=CellType.CODE.value,
            execution_count=None,
            id=next(self._id_gen),
            metadata={},
            outputs=[],
            source=source,
        )

        self.cells.append(cell)

    def save(self, fp: str) -> str:
        with open(fp, "w") as f:
            json.dump(self._dict, f, indent=2)
        return fp


if __name__ == "__main__":
    nb = Notebook()
    nb.add_cell_markdown(source=["My cell has", "two lines"])
    nb.add_cell_code(source=["""print("hello world")"""])
    print(nb.save("test_nb.ipynb"))
