"""Sort/filter proxy model for QTableView."""

from PySide6.QtCore import QSortFilterProxyModel


class SortFilterProxy(QSortFilterProxyModel):
    """Enables sorting and optional text filtering on a PandasModel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._filter_text: str = ""

    def set_filter(self, text: str) -> None:
        self._filter_text = text.strip().lower()
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent) -> bool:
        if not self._filter_text:
            return True
        model = self.sourceModel()
        if model is None:
            return True
        # Check the first column (kernel_name) for the filter text
        idx = model.index(source_row, 0, source_parent)
        data = model.data(idx)
        return self._filter_text in str(data).lower()
