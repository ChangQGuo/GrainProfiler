"""QAbstractTableModel wrapping a pandas DataFrame."""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt


class PandasModel(QAbstractTableModel):
    """Qt table model adapter for a pandas DataFrame."""

    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self._df = df

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._df)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._df.columns)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None

        col = self._df.columns[index.column()]
        val = self._df.iat[index.row(), index.column()]

        if role == Qt.DisplayRole:
            if pd.isna(val):
                return ""
            # Format specific columns
            if col in ("length_mm", "max_width_mm", "width_25pct_mm", "width_50pct_mm", "width_75pct_mm"):
                return f"{float(val):.2f}"
            if col in ("area_mm2",):
                return f"{float(val):.2f}"
            if col in ("circularity", "eccentricity", "length_width_ratio"):
                return f"{float(val):.4f}"
            return str(val)

        if role == Qt.TextAlignmentRole:
            if col == "kernel_name" or col == "kernel_id":
                return int(Qt.AlignLeft | Qt.AlignVCenter)
            return int(Qt.AlignRight | Qt.AlignVCenter)

        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            cols = list(self._df.columns)
            if section < len(cols):
                return str(cols[section])
        return str(section + 1)

    def get_row_data(self, row: int) -> pd.Series | None:
        if 0 <= row < len(self._df):
            return self._df.iloc[row]
        return None

    def row_for_value(self, column: str, value) -> int:
        """Find the first row index where *column* matches *value*."""
        try:
            matches = self._df[self._df[column] == value]
            if not matches.empty:
                return int(matches.index[0])
        except (KeyError, IndexError):
            pass
        return -1
