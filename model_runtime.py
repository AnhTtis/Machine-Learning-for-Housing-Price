import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

TARGET_COL = "Giá bán"
AREA_COL = "Diện tích"
UNIT_PRICE_COL = "Đơn giá (tr/m2)"
UNIT_PRICE_SCALE = 1_000_000.0
UNIT_PRICE_OUTLIER_IQR_MULTIPLIER = 1.5
PROPERTY_TYPE_COL = "Loại hình"

ADDRESS_TEXT_COLUMNS = [
    "Đường",
    "Số nhà",
    "Phường/Xã cũ",
    "Phường mới",
    "Huyện/Quận cũ",
    "Tỉnh/Thành phố cũ",
    "Tỉnh/Thành phố mới",
]

ICON_PATTERN = re.compile(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+", re.UNICODE)
LEADING_TRAILING_DECORATION_PATTERN = re.compile(r"^[\s,.;:|/\-]+|[\s,.;:|/\-]+$", re.UNICODE)


def normalize_text_value(value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return np.nan

    text = ICON_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = LEADING_TRAILING_DECORATION_PATTERN.sub("", text).strip()
    return text if text else np.nan


def sanitize_location_columns(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    cleaned_df = df.copy()
    for column in columns or ADDRESS_TEXT_COLUMNS:
        if column in cleaned_df.columns:
            cleaned_df[column] = cleaned_df[column].apply(normalize_text_value)
    return cleaned_df


def parse_numeric_series(series, keep_decimal=True, area_mode=False, remove_suffix_pattern=None):
    values = series.astype("string").fillna("").str.strip()
    if remove_suffix_pattern:
        values = values.str.replace(remove_suffix_pattern, "", regex=True)

    if area_mode:
        values = (
            values.str.replace(r"(?i)^none\s*m2?$", "", regex=True)
            .str.replace(",", "", regex=False)
            .str.extract(r"^(\d+(?:\.\d+)?)", expand=False)
        )
    else:
        values = values.str.replace(",", "", regex=False)
        pattern = r"[^0-9.]" if keep_decimal else r"[^0-9]"
        values = values.str.replace(pattern, "", regex=True)

    return pd.to_numeric(values, errors="coerce")


def remove_unit_price_outliers(
    df: pd.DataFrame,
    unit_price_col: str = UNIT_PRICE_COL,
    iqr_multiplier: float = UNIT_PRICE_OUTLIER_IQR_MULTIPLIER,
) -> pd.DataFrame:
    if unit_price_col not in df.columns:
        return df

    unit_price = pd.to_numeric(df[unit_price_col], errors="coerce")
    valid_unit_price = unit_price[unit_price.gt(0)]
    if valid_unit_price.empty:
        return df

    q1 = valid_unit_price.quantile(0.25)
    q3 = valid_unit_price.quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr) or iqr <= 0:
        keep_mask = unit_price.isna() | unit_price.gt(0)
        return df.loc[keep_mask].copy()

    lower_bound = max(0.0, q1 - iqr_multiplier * iqr)
    upper_bound = q3 + iqr_multiplier * iqr
    keep_mask = unit_price.isna() | unit_price.between(lower_bound, upper_bound)
    return df.loc[keep_mask].copy()


def preprocess_raw_data(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    for column in ADDRESS_TEXT_COLUMNS:
        if column in df.columns:
            df[column] = df[column].apply(normalize_text_value)

    numeric_columns = ["Số tầng", "Kinh độ", "Vĩ độ"]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "Ngày đăng" in df.columns:
        df["Ngày đăng"] = pd.to_datetime(df["Ngày đăng"], errors="coerce")

    boolean_columns = ["Đăng bởi Đối tác", "Có hướng ra đường"]
    for column in boolean_columns:
        if column in df.columns and df[column].dtype == "object":
            df[column] = df[column].map({"True": True, "False": False, "1": True, "0": False})

    decimal_numeric_columns = [UNIT_PRICE_COL]
    integer_like_columns = [TARGET_COL]
    area_columns = ["Chiều dài", "Chiều rộng", AREA_COL]

    for column in decimal_numeric_columns:
        if column in df.columns and df[column].dtype == "object":
            df[column] = parse_numeric_series(df[column], keep_decimal=True)

    for column in integer_like_columns:
        if column in df.columns and df[column].dtype == "object":
            df[column] = parse_numeric_series(df[column], keep_decimal=False)

    for column in area_columns:
        if column in df.columns and df[column].dtype == "object":
            df[column] = parse_numeric_series(df[column], area_mode=True)

    return remove_unit_price_outliers(df)


def fill_area(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    length_col = "Chiều dài"
    width_col = "Chiều rộng"
    district_col = "Huyện/Quận cũ"
    ward_old_col = "Phường/Xã cũ"
    ward_new_col = "Phường mới"

    fallback_area_scope = df.get(ward_new_col)
    if fallback_area_scope is None:
        fallback_area_scope = df.get(ward_old_col)
    elif ward_old_col in df.columns:
        fallback_area_scope = fallback_area_scope.fillna(df[ward_old_col])

    if fallback_area_scope is None:
        fallback_area_scope = df[district_col]

    df["_khu_vuc"] = fallback_area_scope.fillna(df[district_col])

    for column in [AREA_COL, length_col, width_col, UNIT_PRICE_COL, TARGET_COL]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    mask_from_dimensions = df[AREA_COL].isna() & df[length_col].gt(0) & df[width_col].gt(0)
    df.loc[mask_from_dimensions, AREA_COL] = df.loc[mask_from_dimensions, length_col] * df.loc[mask_from_dimensions, width_col]

    mask_from_price = df[AREA_COL].isna() & df[TARGET_COL].gt(0) & df[UNIT_PRICE_COL].gt(0)
    df.loc[mask_from_price, AREA_COL] = (
        df.loc[mask_from_price, TARGET_COL] / (df.loc[mask_from_price, UNIT_PRICE_COL] * UNIT_PRICE_SCALE)
    )

    area_group_mean = df.groupby([PROPERTY_TYPE_COL, district_col, "_khu_vuc"])[AREA_COL].transform("mean")
    mask_from_group_mean = df[AREA_COL].isna() & area_group_mean.notna()
    df.loc[mask_from_group_mean, AREA_COL] = area_group_mean[mask_from_group_mean]

    return df.drop(columns=["_khu_vuc"])


class MatrixToFrameTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        elif not hasattr(self, "feature_names_in_"):
            self.feature_names_in_ = None
        return self

    def transform(self, X):
        if hasattr(X, "columns") or self.feature_names_in_ is None:
            return X

        if sparse.issparse(X):
            return pd.DataFrame.sparse.from_spmatrix(X, columns=self.feature_names_in_)

        X_array = np.asarray(X)
        if X_array.ndim == 2 and X_array.shape[1] == len(self.feature_names_in_):
            return pd.DataFrame(X_array, columns=self.feature_names_in_)
        return X
