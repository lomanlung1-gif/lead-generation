"""GraphBuilder — parses an Excel workbook into a ``nx.MultiDiGraph``.

Expected Excel layout
---------------------
Sheet **Nodes** (required columns): ``id``, ``type``
  All additional columns are stored as node attributes.

Sheet **Edges** (required columns): ``source``, ``target``, ``type``
  All additional columns are stored as edge attributes.

If the workbook does not contain these sheets the builder falls back to a
best-effort heuristic (first sheet → nodes, second sheet → edges).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Build a ``nx.MultiDiGraph`` from an Excel workbook."""

    NODE_SHEET = "Nodes"
    EDGE_SHEET = "Edges"

    @classmethod
    def from_excel(cls, path: Union[str, Path]) -> nx.MultiDiGraph:
        """Parse *path* and return a populated ``nx.MultiDiGraph``.

        Parameters
        ----------
        path:
            Path to an ``.xlsx`` workbook.

        Returns
        -------
        nx.MultiDiGraph
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Excel file not found: {path}")

        xl = pd.ExcelFile(path)
        sheet_names = xl.sheet_names

        node_df = cls._load_sheet(xl, cls.NODE_SHEET, sheet_names, index=0)
        edge_df = cls._load_sheet(xl, cls.EDGE_SHEET, sheet_names, index=1)

        G: nx.MultiDiGraph = nx.MultiDiGraph()

        # ── nodes ──────────────────────────────────────────────────────────
        id_col = cls._required_col(node_df, ["id", "ID", "name", "Name"], "Nodes")
        type_col = cls._optional_col(node_df, ["type", "Type", "category"])

        for _, row in node_df.iterrows():
            node_id = str(row[id_col])
            attrs: dict = {k: v for k, v in row.items() if k != id_col}
            if type_col:
                attrs.setdefault("type", str(row[type_col]))
            G.add_node(node_id, **attrs)

        logger.info("Loaded %d nodes from '%s'", G.number_of_nodes(), path.name)

        # ── edges ──────────────────────────────────────────────────────────
        src_col = cls._required_col(
            edge_df, ["source", "Source", "from", "From"], "Edges"
        )
        tgt_col = cls._required_col(
            edge_df, ["target", "Target", "to", "To"], "Edges"
        )
        etype_col = cls._optional_col(edge_df, ["type", "Type", "relation"])

        for _, row in edge_df.iterrows():
            src = str(row[src_col])
            tgt = str(row[tgt_col])
            attrs = {k: v for k, v in row.items() if k not in (src_col, tgt_col)}
            if etype_col:
                attrs.setdefault("type", str(row[etype_col]))

            # Auto-create nodes referenced only in edges
            for n in (src, tgt):
                if n not in G:
                    G.add_node(n)

            G.add_edge(src, tgt, **attrs)

        logger.info("Loaded %d edges from '%s'", G.number_of_edges(), path.name)
        return G

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _load_sheet(
        xl: pd.ExcelFile,
        preferred_name: str,
        available: list[str],
        index: int,
    ) -> pd.DataFrame:
        if preferred_name in available:
            df = xl.parse(preferred_name)
        elif index < len(available):
            df = xl.parse(available[index])
            logger.warning(
                "Sheet '%s' not found; using sheet '%s' instead.",
                preferred_name,
                available[index],
            )
        else:
            df = pd.DataFrame()
            logger.warning("No suitable sheet found for index %d.", index)
        return df.fillna("")

    @staticmethod
    def _required_col(df: pd.DataFrame, candidates: list[str], sheet: str) -> str:
        for c in candidates:
            if c in df.columns:
                return c
        raise ValueError(
            f"Sheet '{sheet}' must contain one of {candidates}; "
            f"found columns: {list(df.columns)}"
        )

    @staticmethod
    def _optional_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None
