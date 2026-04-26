"""CSV loading utilities for FlexWorks exports."""

from __future__ import annotations

from pathlib import Path
from typing import IO, Union
import warnings

import pandas as pd


CsvSource = Union[str, Path, IO[str], IO[bytes]]


class DataLoadError(ValueError):
    """Raised when a CSV cannot be loaded into a usable dataframe."""


def load_csv(source: CsvSource) -> pd.DataFrame:
    """Load a CSV from a file path or uploaded file-like object.

    The function intentionally performs only parsing and basic emptiness checks.
    Schema normalization and cleaning happen in the dedicated validation and
    cleaning modules so the data pipeline remains transparent.
    """

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.ParserWarning)
            dataframe = pd.read_csv(source, index_col=False, skipinitialspace=True)
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError("The uploaded CSV is empty.") from exc
    except pd.errors.ParserError as exc:
        raise DataLoadError(
            "The CSV could not be parsed. Check for broken rows, unescaped commas, or inconsistent columns."
        ) from exc
    except pd.errors.ParserWarning as exc:
        raise DataLoadError(
            "The CSV structure looks inconsistent. Check for unescaped commas in numeric values or rows with extra columns."
        ) from exc
    except UnicodeDecodeError as exc:
        raise DataLoadError("The CSV encoding could not be read. Please upload a UTF-8 compatible CSV.") from exc
    except (OSError, ValueError) as exc:
        raise DataLoadError(f"The CSV could not be loaded: {exc}") from exc

    if dataframe.empty:
        raise DataLoadError("The uploaded CSV does not contain any data rows.")

    return dataframe
