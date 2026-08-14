"""CSV export utility."""

import pandas as pd


def export_dataframe(df: pd.DataFrame, path: str) -> bool:
    """Save a DataFrame to CSV. Returns True on success."""
    try:
        df.to_csv(path, index=False)
        return True
    except OSError:
        return False
