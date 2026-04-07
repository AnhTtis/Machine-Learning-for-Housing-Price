import math
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import _column_transformer as sklearn_column_transformer


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


# scikit-learn 1.8 no longer exposes this private helper, but older 1.6.x
# ColumnTransformer pickles may still reference it during unpickling.
if not hasattr(sklearn_column_transformer, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass


    sklearn_column_transformer._RemainderColsList = _RemainderColsList


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


# Allow older joblib artifacts that were serialized with the training module path
# to be loaded without importing the training script on deploy.
sys.modules.setdefault("train_combo4_xgboost_pipeline", sys.modules[__name__])


MODEL_PATH = Path("artifacts/combo4_best_unit_price_pipeline.pkl")
DATA_PATH = Path("advertisement.csv")

AREA_FIELD = "Diện tích"
CITY_FIELD = "Tỉnh/Thành phố cũ"
DISTRICT_FIELD = "Huyện/Quận cũ"
OLD_WARD_FIELD = "Phường/Xã cũ"
NEW_CITY_FIELD = "Tỉnh/Thành phố mới"
NEW_WARD_FIELD = "Phường mới"
STREET_FIELD = "Đường"

SUBTYPE_FIELDS = [
    "Loại hình nhà ở",
    "Loại hình căn hộ",
    "Loại hình đất",
    "Loại hình văn phòng",
]


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_raw_address_data():
    df = pd.read_csv(DATA_PATH, low_memory=False, encoding="utf-8-sig")
    return sanitize_location_columns(df)


def to_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clean_option_series(series):
    values = series.dropna().astype(str).str.strip()
    values = values[(values != "") & (~values.str.lower().isin(["none", "nan", "null"]))]
    return sorted(values.unique().tolist())


def select_options(series, include_blank=True):
    options = clean_option_series(series)
    return ([""] + options) if include_blank else options


def format_vnd(price):
    if price is None or (isinstance(price, float) and math.isnan(price)):
        return "Không có kết quả"
    return f"{price:,.0f} VND"


def format_bn_vnd(price_bn):
    if price_bn is None or (isinstance(price_bn, float) and math.isnan(price_bn)):
        return "Không có kết quả"
    return f"{price_bn:.2f} tỷ VND"


def build_estimate_range(prediction_vnd, mae_bn_vnd):
    mae_vnd = mae_bn_vnd * 1e9
    return max(0.0, prediction_vnd - mae_vnd), prediction_vnd + mae_vnd


def find_relevant_subtype_fields(df, property_type):
    if not property_type:
        return []

    type_df = df[df[PROPERTY_TYPE_COL] == property_type].copy()
    return [field for field in SUBTYPE_FIELDS if field in type_df.columns and clean_option_series(type_df[field])]


def get_model_for_property_type(artifact, property_type):
    models_by_type = artifact.get("models_by_type", {})
    if property_type and property_type in models_by_type:
        model = models_by_type[property_type]
        model_name = artifact.get("property_type_models", {}).get(property_type, {}).get("best_model_name")
        metrics = artifact.get("property_type_models", {}).get(property_type, {}).get("metrics", artifact["metrics"])
        return model, model_name or artifact.get("model_name", "Không rõ"), metrics

    return artifact["pipeline"], artifact.get("model_name", "Không rõ"), artifact["metrics"]


def predict_price(model, feature_list, target_mode, unit_price_scale, values):
    row = {feature: None for feature in feature_list}
    row.update(
        {
            CITY_FIELD: values.get(CITY_FIELD) or None,
            DISTRICT_FIELD: values.get(DISTRICT_FIELD) or None,
            PROPERTY_TYPE_COL: values.get(PROPERTY_TYPE_COL) or None,
            AREA_FIELD: to_float(values.get(AREA_FIELD)),
            "Số phòng ngủ": to_float(values.get("Số phòng ngủ")),
            "Số phòng vệ sinh": to_float(values.get("Số phòng vệ sinh")),
            STREET_FIELD: values.get(STREET_FIELD) or None,
            "Loại hình nhà ở": values.get("Loại hình nhà ở") or None,
            "Loại hình căn hộ": values.get("Loại hình căn hộ") or None,
            "Loại hình đất": values.get("Loại hình đất") or None,
            "Loại hình văn phòng": values.get("Loại hình văn phòng") or None,
        }
    )

    input_df = pd.DataFrame([row], columns=feature_list)
    pred_value = float(model.predict(input_df)[0])
    area_value = to_float(values.get(AREA_FIELD))

    if target_mode == "unit_price_times_area":
        pred_price = pred_value * area_value * unit_price_scale if area_value is not None else float("nan")
        result = {
            "Giá dự đoán (VND)": format_vnd(pred_price),
            "Giá dự đoán (tỷ VND)": round(pred_price / 1e9, 3) if not math.isnan(pred_price) else None,
            "Đơn giá dự đoán (triệu/m²)": round(pred_value, 4),
        }
        return result

    return {
        "Giá dự đoán (VND)": format_vnd(pred_value),
        "Giá dự đoán (tỷ VND)": round(pred_value / 1e9, 3) if not math.isnan(pred_value) else None,
    }


def resolve_legacy_address_candidates(df, new_city, new_ward, street):
    if not new_city or not new_ward or not street:
        return pd.DataFrame(), pd.DataFrame()

    filtered = df.copy()
    filtered = filtered[filtered[NEW_CITY_FIELD] == new_city]
    filtered = filtered[filtered[NEW_WARD_FIELD] == new_ward]
    filtered = filtered[filtered[STREET_FIELD] == street]

    if filtered.empty:
        return filtered, pd.DataFrame()

    candidate_summary = (
        filtered.groupby([OLD_WARD_FIELD, DISTRICT_FIELD, CITY_FIELD], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(by=["rows", OLD_WARD_FIELD, DISTRICT_FIELD, CITY_FIELD], ascending=[False, True, True, True])
        .reset_index(drop=True)
    )
    return filtered, candidate_summary


artifact = load_artifact()
raw_address_df = load_raw_address_data()
feature_list = artifact["feature_list"]
target_mode = artifact.get("target_mode", "price_direct")
unit_price_scale = float(artifact.get("unit_price_scale", 1.0))
property_type_column = artifact.get("property_type_column", PROPERTY_TYPE_COL)

st.set_page_config(page_title="Dự đoán giá bất động sản", page_icon="🏠", layout="wide")

st.title("Dự đoán giá bất động sản tại Việt Nam")
st.caption(
    "Nhập theo địa chỉ mới từ dataset gốc: Tỉnh/Thành phố mới, rồi Phường/Xã mới, rồi Đường. "
    "Ứng dụng sẽ tự map sang địa chỉ cũ để đưa vào model dự đoán."
)

new_city_options = select_options(raw_address_df[NEW_CITY_FIELD])
selected_new_city = st.selectbox("Tỉnh/Thành phố mới", new_city_options, index=0)

new_city_df = raw_address_df.copy()
if selected_new_city:
    new_city_df = new_city_df[new_city_df[NEW_CITY_FIELD] == selected_new_city]

new_ward_options = select_options(new_city_df[NEW_WARD_FIELD])
selected_new_ward = st.selectbox("Phường/Xã mới", new_ward_options, index=0)

new_ward_df = new_city_df.copy()
if selected_new_ward:
    new_ward_df = new_ward_df[new_ward_df[NEW_WARD_FIELD] == selected_new_ward]

street_options = select_options(new_ward_df[STREET_FIELD])
selected_street = st.selectbox("Đường", street_options, index=0)

_, legacy_candidates = resolve_legacy_address_candidates(
    raw_address_df,
    selected_new_city,
    selected_new_ward,
    selected_street,
)

legacy_old_ward_options = clean_option_series(legacy_candidates[OLD_WARD_FIELD]) if not legacy_candidates.empty else []
selected_old_ward = legacy_old_ward_options[0] if legacy_old_ward_options else ""

if len(legacy_old_ward_options) > 1:
    selected_old_ward = st.selectbox(
        "Phường/Xã cũ",
        legacy_old_ward_options,
        index=0,
        help="Nếu có nhiều khả năng khớp, app mặc định lấy option đầu tiên nhưng bạn có thể chọn lại.",
    )

resolved_candidates = legacy_candidates.copy()
if selected_old_ward:
    resolved_candidates = resolved_candidates[resolved_candidates[OLD_WARD_FIELD] == selected_old_ward]

resolved_old_city = ""
resolved_old_district = ""
resolved_old_ward = selected_old_ward
if not resolved_candidates.empty:
    top_match = (
        resolved_candidates.groupby([OLD_WARD_FIELD, DISTRICT_FIELD, CITY_FIELD], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(by=["rows", OLD_WARD_FIELD, DISTRICT_FIELD, CITY_FIELD], ascending=[False, True, True, True])
        .iloc[0]
    )
    resolved_old_ward = top_match[OLD_WARD_FIELD]
    resolved_old_district = top_match[DISTRICT_FIELD]
    resolved_old_city = top_match[CITY_FIELD]

st.markdown("**Địa chỉ cũ**")
legacy_col_1, legacy_col_2, legacy_col_3 = st.columns(3)
legacy_col_1.text_input("Phường/Xã cũ", value=resolved_old_ward or "", disabled=True)
legacy_col_2.text_input("Huyện/Quận cũ", value=resolved_old_district or "", disabled=True)
legacy_col_3.text_input("Tỉnh/Thành phố cũ", value=resolved_old_city or "", disabled=True)

input_scope_df = raw_address_df.copy()
if resolved_old_city:
    input_scope_df = input_scope_df[input_scope_df[CITY_FIELD] == resolved_old_city]
if resolved_old_district:
    input_scope_df = input_scope_df[input_scope_df[DISTRICT_FIELD] == resolved_old_district]
if selected_street:
    input_scope_df = input_scope_df[input_scope_df[STREET_FIELD] == selected_street]

property_type_options = select_options(input_scope_df[property_type_column])
selected_property_type = st.selectbox("Loại hình", property_type_options, index=0)

type_df = input_scope_df.copy()
if selected_property_type:
    type_df = type_df[type_df[property_type_column] == selected_property_type]

relevant_subtype_fields = find_relevant_subtype_fields(input_scope_df, selected_property_type)

left_col, right_col = st.columns(2)
with left_col:
    area_value = st.number_input("Diện tích", min_value=0.0, step=1.0)
    bedroom_options = select_options(type_df["Số phòng ngủ"])
    bathroom_options = select_options(type_df["Số phòng vệ sinh"])
    selected_bedrooms = st.selectbox("Số phòng ngủ", bedroom_options, index=0)
    selected_bathrooms = st.selectbox("Số phòng vệ sinh", bathroom_options, index=0)

subtype_values = {}
with right_col:
    for field in relevant_subtype_fields:
        subtype_values[field] = st.selectbox(field, select_options(type_df[field]), index=0)

submitted = st.button("Dự đoán", use_container_width=True)

if submitted:
    if area_value <= 0:
        st.error("Vui lòng nhập diện tích lớn hơn 0.")
    elif not selected_new_city or not selected_new_ward or not selected_street or not selected_property_type:
        st.error("Vui lòng chọn đầy đủ Tỉnh/Thành phố mới, Phường/Xã mới, Đường và Loại hình.")
    elif not resolved_old_city or not resolved_old_district:
        st.error("Không map được địa chỉ cũ từ dữ liệu hiện tại nên chưa thể dự đoán.")
    else:
        model, model_name, model_metrics = get_model_for_property_type(artifact, selected_property_type)
        values = {
            CITY_FIELD: resolved_old_city,
            DISTRICT_FIELD: resolved_old_district,
            STREET_FIELD: selected_street,
            PROPERTY_TYPE_COL: selected_property_type,
            AREA_FIELD: str(area_value),
            "Số phòng ngủ": selected_bedrooms,
            "Số phòng vệ sinh": selected_bathrooms,
            "Loại hình nhà ở": subtype_values.get("Loại hình nhà ở", ""),
            "Loại hình căn hộ": subtype_values.get("Loại hình căn hộ", ""),
            "Loại hình đất": subtype_values.get("Loại hình đất", ""),
            "Loại hình văn phòng": subtype_values.get("Loại hình văn phòng", ""),
        }
        prediction = predict_price(model, feature_list, target_mode, unit_price_scale, values)

        st.subheader("Kết quả dự đoán")
        metric_col_1, metric_col_2 = st.columns(2)
        metric_col_1.metric("Giá dự đoán", prediction["Giá dự đoán (VND)"])
        if target_mode == "unit_price_times_area":
            metric_col_2.metric(
                "Đơn giá dự đoán",
                f"{prediction['Đơn giá dự đoán (triệu/m²)']:.2f} triệu/m²",
            )

        predicted_price_bn = prediction["Giá dự đoán (tỷ VND)"]
        predicted_price_vnd = predicted_price_bn * 1e9 if predicted_price_bn is not None else float("nan")
        lower_bound, upper_bound = build_estimate_range(predicted_price_vnd, model_metrics["MAE (bn VND)"])

        st.info(
            f"Model đang dùng: {model_name}. Sai lệch trung bình tham khảo cho nhóm này khoảng "
            f"{format_bn_vnd(model_metrics['MAE (bn VND)'])}. "
            f"Mức giá tham khảo có thể hiểu trong khoảng {format_vnd(lower_bound)} đến {format_vnd(upper_bound)}."
        )

if artifact.get("property_type_models"):
    summary_rows = []
    for property_type, info in artifact["property_type_models"].items():
        summary_rows.append(
            {
                "Loại hình": property_type,
                "Model tốt nhất": info["best_model_name"],
                "MAE (tỷ VND)": round(info["metrics"]["MAE (bn VND)"], 3),
                "RMSLE": round(info["metrics"]["RMSLE"], 3),
                "Số mẫu": info["rows"],
            }
        )
    st.subheader("Model Tốt Nhất Theo Loại Hình")
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
