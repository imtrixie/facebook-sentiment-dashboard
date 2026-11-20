"""
Facebook Sentiment Analysis Dashboard
Ứng dụng phân tích cảm xúc bài đăng Facebook xây dựng bằng Python & Streamlit.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# ==============================================================================
# CẤU HÌNH TRANG & HẰNG SỐ MÀU SẮC
# ==============================================================================
st.set_page_config(
    page_title="Facebook Sentiment Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE_PATH = "Facebook_data_txt.csv"

SENTIMENT_ORDER = ["positive", "neutral", "negative"]
SENTIMENT_LABELS_VI = {
    "positive": "Positive (Tích cực)",
    "neutral": "Neutral (Trung lập)",
    "negative": "Negative (Tiêu cực)",
}
SENTIMENT_SHORT_VI = {
    "positive": "Tích cực",
    "neutral": "Trung lập",
    "negative": "Tiêu cực",
}
SENTIMENT_COLORS = {
    "positive": "#16A34A",  # Xanh lá
    "neutral": "#F59E0B",   # Vàng / Cam
    "negative": "#DC2626",  # Đỏ
}
SENTIMENT_DISPLAY_COLORS = {
    SENTIMENT_LABELS_VI["positive"]: "#16A34A",
    SENTIMENT_LABELS_VI["neutral"]: "#F59E0B",
    SENTIMENT_LABELS_VI["negative"]: "#DC2626",
}


# ==============================================================================
# GIAO DIỆN & CSS (FACEBOOK NAVY + ICE-BLUE + VIOLET)
# ==============================================================================
def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

            html, body, [class*="css"] {
                font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            }
            .stApp {
                background-color: #EEF4FB;
            }
            header[data-testid="stHeader"] {
                background: rgba(238, 244, 251, 0.88);
                backdrop-filter: blur(8px);
                border-bottom: 1px solid #CFE0F5;
            }
            section[data-testid="stSidebar"] {
                background-color: #FFFFFF;
                border-right: 1px solid #CFE0F5;
            }
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 3rem;
                max-width: 1460px;
            }
            /* Header Banner (Navy -> Royal Blue -> Violet gradient) */
            .dashboard-header {
                background: linear-gradient(115deg, #07327A 0%, #0D47A1 58%, #4F26A6 100%);
                border-radius: 14px;
                padding: 1.25rem 1.6rem;
                margin-bottom: 1.25rem;
                display: flex;
                flex-wrap: wrap;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
                box-shadow: 0 6px 20px rgba(7, 50, 122, 0.16);
            }
            .dashboard-title {
                font-size: 1.45rem;
                font-weight: 700;
                color: #FFFFFF;
                letter-spacing: -0.02em;
                margin: 0;
                line-height: 1.25;
            }
            .dashboard-subtitle {
                font-size: 0.875rem;
                color: rgba(255, 255, 255, 0.84);
                margin: 0.25rem 0 0 0;
            }
            .dashboard-meta {
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.78rem;
                color: #FFFFFF;
                background: rgba(255, 255, 255, 0.14);
                border: 1px solid rgba(255, 255, 255, 0.28);
                border-radius: 8px;
                padding: 0.5rem 0.9rem;
                font-variant-numeric: tabular-nums;
            }
            /* Glacier Blue KPI Tray */
            .kpi-tray-wrapper {
                background-color: #D8EAF9;
                border: 1px solid #C2DCF5;
                border-radius: 14px;
                padding: 1rem 1rem 0.5rem 1rem;
                margin-bottom: 1.25rem;
            }
            /* KPI Cards */
            div[data-testid="stMetric"] {
                background-color: #FFFFFF;
                border: 1.5px solid #C7DBF4;
                border-radius: 12px;
                padding: 0.95rem 1.15rem;
                box-shadow: 0 2px 6px rgba(20, 20, 58, 0.04);
                transition: transform 0.15s ease, box-shadow 0.15s ease;
            }
            div[data-testid="stMetric"]:hover {
                box-shadow: 0 4px 12px rgba(13, 71, 161, 0.08);
            }
            div[data-testid="stMetric"] label {
                font-size: 0.82rem;
                font-weight: 600;
                color: #475569;
            }
            div[data-testid="stMetricValue"] {
                font-size: 1.65rem;
                font-weight: 700;
                color: #14143A;
                font-variant-numeric: tabular-nums;
            }
            div[data-testid="stMetricDelta"] {
                font-size: 0.78rem;
                font-variant-numeric: tabular-nums;
            }
            /* Colored crisp outlines matching reference dashboard */
            div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(1) div[data-testid="stMetric"] {
                border: 1.5px solid #A855F7;
            }
            div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2) div[data-testid="stMetric"] {
                border: 1.5px solid #22C55E;
            }
            div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3) div[data-testid="stMetric"] {
                border: 1.5px solid #F59E0B;
            }
            div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(4) div[data-testid="stMetric"] {
                border: 1.5px solid #F43F5E;
            }
            /* Tabs styling */
            div[data-testid="stTabs"] button[role="tab"] {
                font-weight: 600;
                font-size: 0.9rem;
                padding: 0.65rem 1.15rem;
                color: #475569;
            }
            div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
                color: #0D47A1;
            }
            /* Section titles */
            .section-heading {
                font-size: 1.08rem;
                font-weight: 700;
                color: #14143A;
                margin-bottom: 0.15rem;
            }
            .section-caption {
                font-size: 0.84rem;
                color: #51617A;
                margin-bottom: 0.9rem;
            }
            .pred-card {
                border: 1px solid #CFE0F5;
                border-radius: 10px;
                padding: 1.1rem 1.25rem;
                background-color: #F4F8FD;
                margin-bottom: 0.75rem;
            }
            /* Bordered containers refinement */
            div[data-testid="stVerticalBlockBorderWrapper"] {
                background-color: #FFFFFF;
                border-color: #CFE0F5 !important;
                border-radius: 12px !important;
                box-shadow: 0 2px 6px rgba(20, 20, 58, 0.03);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_plotly_theme(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#14143A", size=12),
        title_font=dict(color="#14143A", size=15),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor="#0B2E6B",
            bordercolor="#0B2E6B",
            font=dict(family="Plus Jakarta Sans, sans-serif", color="#FFFFFF", size=12),
        ),
        height=height,
    )
    return fig


# ==============================================================================
# TẢI & TIỀN XỬ LÝ DỮ LIỆU
# ==============================================================================
@st.cache_data(show_spinner="Đang tải và xử lý dữ liệu từ Facebook_data_txt.csv...")
def load_data(file_path: str = DATA_FILE_PATH) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu '{file_path}'. "
            "Vui lòng đặt file Facebook_data_txt.csv cùng thư mục với app.py."
        )

    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"File dữ liệu '{file_path}' không có bản ghi nào.")

    required_cols = [
        "post_id",
        "timestamp",
        "text",
        "category",
        "sentiment",
        "language",
        "has_link",
        "likes",
        "comments",
        "shares",
        "is_spam",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"File CSV thiếu các cột bắt buộc: {', '.join(missing_cols)}"
        )

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df["post_id"] = df["post_id"].astype(str).str.strip()
    df["text"] = df["text"].fillna("").astype(str)
    df["category"] = df["category"].fillna("unknown").astype(str).str.strip()
    df["language"] = df["language"].fillna("unknown").astype(str).str.strip()

    # Giữ nguyên nhãn cảm xúc gốc, chỉ chuẩn hóa khoảng trắng/chữ thường để khớp ánh xạ
    df["sentiment_raw"] = df["sentiment"].astype(str)
    df["sentiment"] = (
        df["sentiment"].fillna("neutral").astype(str).str.strip().str.lower()
    )
    df = df[df["sentiment"].isin(SENTIMENT_ORDER)].copy()

    for col in ["likes", "comments", "shares", "has_link", "is_spam"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["total_engagement"] = df["likes"] + df["comments"] + df["shares"]
    df = df.reset_index(drop=True)
    return df


# ==============================================================================
# HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH MACHINE LEARNING
# ==============================================================================
def _build_vectorizer(vectorizer_type: str) -> TfidfVectorizer | CountVectorizer:
    if vectorizer_type == "Bag of Words":
        return CountVectorizer(
            max_features=4500,
            ngram_range=(1, 2),
            lowercase=True,
        )
    return TfidfVectorizer(
        max_features=4500,
        ngram_range=(1, 2),
        sublinear_tf=True,
        lowercase=True,
    )


def _build_classifier(model_name: str) -> MultinomialNB | DecisionTreeClassifier:
    if model_name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=24,
            min_samples_split=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        )
    return MultinomialNB(alpha=0.5)


@st.cache_resource(show_spinner="Đang huấn luyện và đánh giá các mô hình Machine Learning...")
def train_ml_models(
    texts: tuple[str, ...],
    labels: tuple[str, ...],
    vectorizer_type: str = "TF-IDF",
) -> dict[str, Any]:
    """
    Huấn luyện Naive Bayes và Decision Tree với Train/Test Split (80/20)
    và tạo dự đoán ngoài mẫu (Out-of-Fold 5-Fold CV) cho toàn bộ tập dữ liệu
    để đảm bảo không bao giờ huấn luyện và đánh giá trên cùng một tập dữ liệu.
    """
    X = pd.Series(texts)
    y = pd.Series(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results: dict[str, Any] = {}

    for model_name in ["Naive Bayes", "Decision Tree"]:
        pipeline = Pipeline(
            [
                ("vectorizer", _build_vectorizer(vectorizer_type)),
                ("classifier", _build_classifier(model_name)),
            ]
        )
        pipeline.fit(X_train, y_train)

        y_test_pred = pipeline.predict(X_test)
        test_acc = float(accuracy_score(y_test, y_test_pred))
        test_f1 = float(f1_score(y_test, y_test_pred, average="macro"))
        report_dict = classification_report(
            y_test,
            y_test_pred,
            labels=SENTIMENT_ORDER,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_test, y_test_pred, labels=SENTIMENT_ORDER)

        oof_pipeline = Pipeline(
            [
                ("vectorizer", _build_vectorizer(vectorizer_type)),
                ("classifier", _build_classifier(model_name)),
            ]
        )
        oof_predictions = cross_val_predict(oof_pipeline, X, y, cv=cv)

        results[model_name] = {
            "pipeline": pipeline,
            "test_accuracy": test_acc,
            "test_f1_macro": test_f1,
            "classification_report": report_dict,
            "confusion_matrix": cm,
            "oof_predictions": oof_predictions,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

    return results


# ==============================================================================
# THANH BỘ LỌC SIDEBAR
# ==============================================================================
def render_sidebar_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    st.sidebar.markdown("## Bảng điều khiển")
    st.sidebar.caption("Bộ lọc cập nhật đồng bộ toàn bộ KPI, biểu đồ và bảng dữ liệu.")

    with st.sidebar.expander("Cấu hình mô hình dự đoán", expanded=True):
        selected_model = st.selectbox(
            "Thuật toán phân loại",
            options=["Naive Bayes", "Decision Tree"],
            index=0,
            help="Áp dụng cho nhãn dự đoán trong bảng bài đăng và công cụ dự đoán trực tiếp.",
        )
        selected_vectorizer = st.radio(
            "Trích xuất đặc trưng",
            options=["TF-IDF", "Bag of Words"],
            index=0,
            horizontal=True,
        )

    st.sidebar.markdown("### Tiêu chí lọc dữ liệu")

    all_categories = sorted(df["category"].unique().tolist())
    select_all_cat = st.sidebar.checkbox("Chọn tất cả danh mục", value=True)
    selected_categories = st.sidebar.multiselect(
        "Danh mục bài đăng (Category)",
        options=all_categories,
        default=all_categories if select_all_cat else [],
    )

    select_all_sent = st.sidebar.checkbox("Chọn tất cả cảm xúc", value=True)
    selected_sentiments = st.sidebar.multiselect(
        "Nhóm cảm xúc (Sentiment)",
        options=SENTIMENT_ORDER,
        default=SENTIMENT_ORDER if select_all_sent else [],
        format_func=lambda s: SENTIMENT_LABELS_VI.get(s, s),
    )

    all_languages = sorted(df["language"].unique().tolist())
    select_all_lang = st.sidebar.checkbox("Chọn tất cả ngôn ngữ", value=True)
    selected_languages = st.sidebar.multiselect(
        "Ngôn ngữ (Language)",
        options=all_languages,
        default=all_languages if select_all_lang else [],
    )

    exclude_spam = st.sidebar.checkbox(
        "Loại bỏ bài đăng spam (is_spam = 1)",
        value=False,
    )

    mask = (
        df["category"].isin(selected_categories)
        & df["sentiment"].isin(selected_sentiments)
        & df["language"].isin(selected_languages)
    )
    if exclude_spam:
        mask = mask & (df["is_spam"] == 0)

    filtered_df = df[mask].copy()

    st.sidebar.markdown("---")
    ratio = len(filtered_df) / len(df) if len(df) > 0 else 0.0
    st.sidebar.progress(ratio)
    st.sidebar.caption(
        f"Đang hiển thị **{len(filtered_df):,}** / **{len(df):,}** bài đăng ({ratio*100:.1f}%)"
    )

    return filtered_df, selected_model, selected_vectorizer


# ==============================================================================
# A. THỐNG KÊ TỔNG QUAN (KPI CARDS TRONG KHAY XANH BĂNG NHẸ)
# ==============================================================================
def render_kpi_section(filtered_df: pd.DataFrame) -> None:
    total_posts = len(filtered_df)
    pos_count = int((filtered_df["sentiment"] == "positive").sum())
    neu_count = int((filtered_df["sentiment"] == "neutral").sum())
    neg_count = int((filtered_df["sentiment"] == "negative").sum())

    pos_pct = (pos_count / total_posts * 100) if total_posts > 0 else 0.0
    neu_pct = (neu_count / total_posts * 100) if total_posts > 0 else 0.0
    neg_pct = (neg_count / total_posts * 100) if total_posts > 0 else 0.0

    with st.container():
        st.markdown('<div class="kpi-tray-wrapper">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="Tổng số bài đăng",
                value=f"{total_posts:,}",
                delta="100.0% dữ liệu đang lọc",
                delta_color="off",
            )
        with col2:
            st.metric(
                label="Tích cực (Positive)",
                value=f"{pos_count:,} ({pos_pct:.1f}%)",
                delta=f"Chiếm {pos_pct:.1f}% tổng bài",
                delta_color="normal",
            )
        with col3:
            st.metric(
                label="Trung lập (Neutral)",
                value=f"{neu_count:,} ({neu_pct:.1f}%)",
                delta=f"Chiếm {neu_pct:.1f}% tổng bài",
                delta_color="off",
            )
        with col4:
            st.metric(
                label="Tiêu cực (Negative)",
                value=f"{neg_count:,} ({neg_pct:.1f}%)",
                delta=f"Chiếm {neg_pct:.1f}% tổng bài",
                delta_color="inverse",
            )
        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# B & C. BIỂU ĐỒ PHÂN BỐ & PHÂN TÍCH TƯƠNG TÁC
# ==============================================================================
def render_distribution_charts(filtered_df: pd.DataFrame) -> None:
    st.markdown('<div class="section-heading">Phân bố cảm xúc bài đăng</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Đối chiếu số lượng và tỷ trọng ba nhóm cảm xúc (Positive: Xanh lá, Neutral: Cam, Negative: Đỏ).</div>',
        unsafe_allow_html=True,
    )

    counts = (
        filtered_df["sentiment"]
        .value_counts()
        .reindex(SENTIMENT_ORDER, fill_value=0)
        .reset_index()
    )
    counts.columns = ["sentiment", "count"]
    counts["label"] = counts["sentiment"].map(SENTIMENT_LABELS_VI)
    total = int(counts["count"].sum())
    counts["percentage"] = np.where(total > 0, counts["count"] / total * 100, 0.0)

    col_bar, col_pie = st.columns(2)

    with col_bar:
        with st.container(border=True):
            fig_bar = px.bar(
                counts,
                x="label",
                y="count",
                color="label",
                text="count",
                color_discrete_map=SENTIMENT_DISPLAY_COLORS,
                title="Số lượng bài đăng theo nhóm cảm xúc",
                labels={"label": "Nhóm cảm xúc", "count": "Số lượng bài đăng"},
            )
            fig_bar.update_traces(
                texttemplate="%{text:,}",
                textposition="outside",
                cliponaxis=False,
                hovertemplate="<b>%{x}</b><br>Số lượng: %{y:,} bài<extra></extra>",
            )
            fig_bar.update_layout(
                showlegend=False,
                yaxis=dict(gridcolor="#EBF2FA", title="Số lượng bài đăng"),
                xaxis=dict(title=""),
                margin=dict(t=50, b=15, l=15, r=15),
            )
            apply_plotly_theme(fig_bar, height=370)
            st.plotly_chart(fig_bar, use_container_width=True)

    with col_pie:
        with st.container(border=True):
            fig_donut = px.pie(
                counts,
                names="label",
                values="count",
                color="label",
                hole=0.54,
                color_discrete_map=SENTIMENT_DISPLAY_COLORS,
                title="Tỷ trọng phần trăm các nhóm cảm xúc",
            )
            fig_donut.update_traces(
                textposition="inside",
                textinfo="percent",
                hovertemplate="<b>%{label}</b><br>Số lượng: %{value:,}<br>Tỷ trọng: %{percent}<extra></extra>",
            )
            fig_donut.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=-0.16, xanchor="center", x=0.5),
                margin=dict(t=50, b=40, l=15, r=15),
                annotations=[
                    dict(
                        text=f"<span style='font-size:12px;color:#51617A'>Tổng</span><br><b>{total:,}</b>",
                        x=0.5,
                        y=0.5,
                        font_size=18,
                        showarrow=False,
                    )
                ],
            )
            apply_plotly_theme(fig_donut, height=370)
            st.plotly_chart(fig_donut, use_container_width=True)


def render_engagement_analysis(filtered_df: pd.DataFrame) -> None:
    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
    head_col, toggle_col = st.columns([1.6, 1.0])
    with head_col:
        st.markdown(
            '<div class="section-heading">Phân tích mức độ tương tác theo cảm xúc</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-caption">So sánh Likes, Comments và Shares giữa ba nhóm cảm xúc.</div>',
            unsafe_allow_html=True,
        )
    with toggle_col:
        mode = st.radio(
            "Chế độ hiển thị:",
            options=["Trung bình trên mỗi bài đăng", "Tổng lượng tương tác"],
            horizontal=True,
        )

    agg_df = (
        filtered_df.groupby("sentiment", as_index=False)
        .agg(
            post_count=("post_id", "count"),
            likes_sum=("likes", "sum"),
            comments_sum=("comments", "sum"),
            shares_sum=("shares", "sum"),
            likes_mean=("likes", "mean"),
            comments_mean=("comments", "mean"),
            shares_mean=("shares", "mean"),
        )
    )
    order_df = pd.DataFrame({"sentiment": SENTIMENT_ORDER})
    agg_df = order_df.merge(agg_df, on="sentiment", how="left").fillna(0)
    agg_df["Nhóm cảm xúc"] = agg_df["sentiment"].map(SENTIMENT_LABELS_VI)

    if mode == "Trung bình trên mỗi bài đăng":
        melted = agg_df.melt(
            id_vars=["sentiment", "Nhóm cảm xúc"],
            value_vars=["likes_mean", "comments_mean", "shares_mean"],
            var_name="metric",
            value_name="value",
        )
        metric_labels = {
            "likes_mean": "Likes (TB)",
            "comments_mean": "Comments (TB)",
            "shares_mean": "Shares (TB)",
        }
        y_title = "Trung bình tương tác / bài"
        text_fmt = "%{text:.1f}"
    else:
        melted = agg_df.melt(
            id_vars=["sentiment", "Nhóm cảm xúc"],
            value_vars=["likes_sum", "comments_sum", "shares_sum"],
            var_name="metric",
            value_name="value",
        )
        metric_labels = {
            "likes_sum": "Tổng Likes",
            "comments_sum": "Tổng Comments",
            "shares_sum": "Tổng Shares",
        }
        y_title = "Tổng số lượt tương tác"
        text_fmt = "%{text:,.0f}"

    melted["Chỉ số tương tác"] = melted["metric"].map(metric_labels)

    col_chart, col_table = st.columns([1.3, 1.0])

    with col_chart:
        with st.container(border=True):
            # Bảng màu đồng bộ với ảnh tham khảo: Cobalt Blue (#005AA9), Cyan (#29C7DC), Royal Violet (#6D28D9)
            fig_grouped = px.bar(
                melted,
                x="Nhóm cảm xúc",
                y="value",
                color="Chỉ số tương tác",
                barmode="group",
                text="value",
                color_discrete_sequence=["#005AA9", "#29C7DC", "#6D28D9"],
                title=f"So sánh tương tác ({mode})",
                labels={"value": y_title},
            )
            fig_grouped.update_traces(
                texttemplate=text_fmt,
                textposition="outside",
                cliponaxis=False,
            )
            fig_grouped.update_layout(
                yaxis=dict(gridcolor="#EBF2FA", title=y_title),
                xaxis=dict(title=""),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=60, b=15, l=15, r=15),
            )
            apply_plotly_theme(fig_grouped, height=380)
            st.plotly_chart(fig_grouped, use_container_width=True)

    with col_table:
        with st.container(border=True):
            st.markdown("**Bảng thống kê tổng hợp tương tác**")
            summary_table = pd.DataFrame(
                {
                    "Nhóm cảm xúc": agg_df["sentiment"].map(SENTIMENT_SHORT_VI),
                    "Số bài": agg_df["post_count"].astype(int),
                    "Tổng Likes": agg_df["likes_sum"].astype(int),
                    "TB Likes": agg_df["likes_mean"].round(1),
                    "Tổng Cmt": agg_df["comments_sum"].astype(int),
                    "TB Cmt": agg_df["comments_mean"].round(1),
                    "Tổng Share": agg_df["shares_sum"].astype(int),
                    "TB Share": agg_df["shares_mean"].round(1),
                }
            )
            st.dataframe(
                summary_table,
                use_container_width=True,
                hide_index=True,
                height=315,
                column_config={
                    "Số bài": st.column_config.NumberColumn(format="%d"),
                    "Tổng Likes": st.column_config.NumberColumn(format="%d"),
                    "TB Likes": st.column_config.NumberColumn(format="%.1f"),
                    "Tổng Cmt": st.column_config.NumberColumn(format="%d"),
                    "TB Cmt": st.column_config.NumberColumn(format="%.1f"),
                    "Tổng Share": st.column_config.NumberColumn(format="%d"),
                    "TB Share": st.column_config.NumberColumn(format="%.1f"),
                },
            )


# ==============================================================================
# E. DANH SÁCH BÀI ĐĂNG & KẾT QUẢ DỰ ĐOÁN
# ==============================================================================
def _highlight_prediction_row(row: pd.Series) -> list[str]:
    is_correct = row["Đánh giá"] == "Đúng"
    bg_eval = (
        "background-color: #ECFDF5; color: #065F46;"
        if is_correct
        else "background-color: #FEF2F2; color: #991B1B;"
    )
    styles = []
    for col in row.index:
        if col == "Đánh giá":
            styles.append(bg_eval + " font-weight: 600;")
        elif col == "Nhãn thực tế":
            styles.append("background-color: #F4F8FD; font-weight: 600;")
        elif col == "Nhãn dự đoán":
            styles.append("background-color: #EDE9FE; color: #4C1D95; font-weight: 600;")
        else:
            styles.append("")
    return styles


def render_posts_table_section(
    filtered_df: pd.DataFrame,
    selected_model: str,
) -> None:
    st.markdown('<div class="section-heading">Danh sách bài đăng & Kết quả dự đoán</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-caption">Đối chiếu nhãn cảm xúc thực tế và nhãn dự đoán ngoài mẫu từ mô hình <b>{selected_model}</b>.</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        ctrl1, ctrl2, ctrl3 = st.columns([2.2, 1.3, 1.2])
        with ctrl1:
            keyword = st.text_input(
                "Tìm kiếm theo từ khóa (nội dung hoặc ID)",
                placeholder="Nhập từ khóa (ví dụ: health, Stark, sale, #relatable...)",
            )
        with ctrl2:
            sort_order = st.selectbox(
                "Sắp xếp theo thời gian",
                options=["Mới nhất trước", "Cũ nhất trước"],
                index=0,
            )
        with ctrl3:
            st.markdown("<div style='height: 1.75rem;'></div>", unsafe_allow_html=True)
            only_wrong = st.checkbox("Chỉ hiển thị bài dự đoán sai", value=False)

    table_df = filtered_df.copy()

    if keyword.strip():
        kw = keyword.strip().lower()
        table_df = table_df[
            table_df["text"].str.lower().str.contains(kw, regex=False)
            | table_df["post_id"].str.lower().str.contains(kw, regex=False)
        ]

    table_df["is_correct"] = table_df["sentiment"] == table_df["predicted_sentiment"]
    if only_wrong:
        table_df = table_df[~table_df["is_correct"]]

    ascending = sort_order == "Cũ nhất trước"
    table_df = table_df.sort_values(by="timestamp", ascending=ascending, na_position="last")

    if table_df.empty:
        st.warning("Không tìm thấy bài đăng nào khớp với điều kiện tìm kiếm hiện tại.")
        return

    correct_count = int(table_df["is_correct"].sum())
    wrong_count = len(table_df) - correct_count
    acc_filtered = (correct_count / len(table_df) * 100) if len(table_df) > 0 else 0.0

    info_col, dl_col = st.columns([2.5, 1.0])
    with info_col:
        st.markdown(
            f"Hiển thị **{len(table_df):,}** bài đăng · "
            f"Dự đoán đúng: **{correct_count:,}** ({acc_filtered:.1f}%) · "
            f"Dự đoán sai: **{wrong_count:,}**"
        )

    display_df = pd.DataFrame(
        {
            "ID bài đăng": table_df["post_id"],
            "Thời gian đăng": table_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("N/A"),
            "Danh mục": table_df["category"],
            "Nội dung bài đăng": table_df["text"],
            "Nhãn thực tế": table_df["sentiment"].map(SENTIMENT_SHORT_VI),
            "Nhãn dự đoán": table_df["predicted_sentiment"].map(SENTIMENT_SHORT_VI),
            "Đánh giá": np.where(table_df["is_correct"], "Đúng", "Sai"),
            "Likes": table_df["likes"],
            "Comments": table_df["comments"],
            "Shares": table_df["shares"],
        }
    )

    with dl_col:
        csv_bytes = display_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="Tải dữ liệu đã lọc (CSV)",
            data=csv_bytes,
            file_name="facebook_sentiment_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )

    preview_limit = 500
    if len(display_df) > preview_limit:
        st.caption(
            f"Bảng hiển thị {preview_limit} dòng đầu tiên để tối ưu tốc độ "
            f"(file CSV tải về chứa đầy đủ {len(display_df):,} dòng)."
        )
    styled_df = display_df.head(preview_limit).style.apply(_highlight_prediction_row, axis=1)

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        height=470,
        column_config={
            "ID bài đăng": st.column_config.TextColumn(width="small"),
            "Thời gian đăng": st.column_config.TextColumn(width="medium"),
            "Danh mục": st.column_config.TextColumn(width="medium"),
            "Nội dung bài đăng": st.column_config.TextColumn(width="large"),
            "Likes": st.column_config.NumberColumn(format="%d"),
            "Comments": st.column_config.NumberColumn(format="%d"),
            "Shares": st.column_config.NumberColumn(format="%d"),
        },
    )


# ==============================================================================
# F. DỰ ĐOÁN CẢM XÚC TRỰC TIẾP & ĐÁNH GIÁ MÔ HÌNH ML
# ==============================================================================
def render_live_prediction_section(
    ml_results: dict[str, Any],
    default_model: str,
    vectorizer_type: str,
) -> None:
    st.markdown('<div class="section-heading">Đánh giá mô hình & Dự đoán trực tiếp</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-caption">Huấn luyện bằng Train/Test Split (80%/20%) và trích xuất đặc trưng <b>{vectorizer_type}</b>. '
        "Đánh giá hiệu năng độc lập trên tập kiểm thử.</div>",
        unsafe_allow_html=True,
    )

    m1 = ml_results["Naive Bayes"]
    m2 = ml_results["Decision Tree"]

    perf_col1, perf_col2 = st.columns([1.15, 1.0])
    with perf_col1:
        with st.container(border=True):
            st.markdown("**So sánh hiệu năng trên tập kiểm thử (Test Set 20%)**")
            eval_rows = []
            for name, res in [("Naive Bayes", m1), ("Decision Tree", m2)]:
                rep = res["classification_report"]
                eval_rows.append(
                    {
                        "Mô hình": name,
                        "Accuracy": f"{res['test_accuracy']*100:.2f}%",
                        "Macro F1": f"{res['test_f1_macro']*100:.2f}%",
                        "F1 Positive": f"{rep['positive']['f1-score']*100:.1f}%",
                        "F1 Neutral": f"{rep['neutral']['f1-score']*100:.1f}%",
                        "F1 Negative": f"{rep['negative']['f1-score']*100:.1f}%",
                    }
                )
            st.dataframe(pd.DataFrame(eval_rows), use_container_width=True, hide_index=True)
            st.caption(
                f"Tập huấn luyện: {m1['train_size']:,} bài · Tập kiểm thử: {m1['test_size']:,} bài"
            )

    with perf_col2:
        with st.container(border=True):
            cm = ml_results[default_model]["confusion_matrix"]
            labels_short = [SENTIMENT_SHORT_VI[s] for s in SENTIMENT_ORDER]
            fig_cm = px.imshow(
                cm,
                x=labels_short,
                y=labels_short,
                text_auto=True,
                color_continuous_scale="PuBu",
                title=f"Ma trận nhầm lẫn ({default_model})",
                labels=dict(x="Nhãn dự đoán", y="Nhãn thực tế", color="Số lượng"),
            )
            fig_cm.update_layout(
                coloraxis_showscale=False,
                margin=dict(t=45, b=15, l=15, r=15),
            )
            apply_plotly_theme(fig_cm, height=230)
            st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    input_col, result_col = st.columns([1.15, 1.0])

    with input_col:
        with st.container(border=True):
            st.markdown("**Nhập bài đăng Facebook mới để dự đoán**")
            sample_choice = st.selectbox(
                "Chọn nhanh câu mẫu kiểm thử:",
                options=[
                    "-- Nhập nội dung tùy chỉnh --",
                    "Absolutely love this new product update! Works smoothly and saved my day",
                    "Still waiting for support to reply about my login issue, really frustrating experience.",
                    "Tech conference scheduled for next Thursday in downtown hall starting at 9 AM.",
                ],
            )
            default_text = "" if sample_choice.startswith("--") else sample_choice
            user_text = st.text_area(
                "Nội dung bài đăng:",
                value=default_text,
                height=125,
                placeholder="Ví dụ: Great service and friendly team, thank you so much!",
            )
            pred_model_name = st.radio(
                "Chọn mô hình dự đoán:",
                options=["Naive Bayes", "Decision Tree"],
                index=0 if default_model == "Naive Bayes" else 1,
                horizontal=True,
            )
            predict_clicked = st.button("Dự đoán cảm xúc", type="primary", use_container_width=True)

    with result_col:
        with st.container(border=True):
            st.markdown("**Kết quả dự đoán & Xác suất**")
            if predict_clicked:
                cleaned = user_text.strip()
                if not cleaned:
                    st.warning("Vui lòng nhập nội dung bài đăng trước khi nhấn Dự đoán.")
                else:
                    pipeline: Pipeline = ml_results[pred_model_name]["pipeline"]
                    pred_label = str(pipeline.predict([cleaned])[0])
                    proba = pipeline.predict_proba([cleaned])[0]
                    classes = list(pipeline.classes_)
                    proba_map = {cls: float(p) for cls, p in zip(classes, proba)}

                    color_hex = SENTIMENT_COLORS.get(pred_label, "#0D47A1")
                    label_vi = SENTIMENT_LABELS_VI.get(pred_label, pred_label)
                    confidence = proba_map.get(pred_label, 0.0) * 100

                    st.markdown(
                        f"""
                        <div class="pred-card" style="border-left: 4px solid {color_hex};">
                            <div style="font-size: 0.78rem; color: #51617A; font-weight: 600;">
                                {pred_model_name.upper()} · {vectorizer_type}
                            </div>
                            <div style="font-size: 1.4rem; font-weight: 700; color: {color_hex}; margin: 0.25rem 0;">
                                {label_vi}
                            </div>
                            <div style="font-size: 0.85rem; color: #14143A;">
                                Độ tin cậy: <b>{confidence:.2f}%</b>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    prob_df = pd.DataFrame(
                        {
                            "Nhóm cảm xúc": [SENTIMENT_SHORT_VI[c] for c in SENTIMENT_ORDER],
                            "Xác suất (%)": [proba_map.get(c, 0.0) * 100 for c in SENTIMENT_ORDER],
                            "Nhãn": [SENTIMENT_LABELS_VI[c] for c in SENTIMENT_ORDER],
                        }
                    )
                    fig_prob = px.bar(
                        prob_df,
                        x="Xác suất (%)",
                        y="Nhóm cảm xúc",
                        orientation="h",
                        color="Nhãn",
                        text="Xác suất (%)",
                        color_discrete_map=SENTIMENT_DISPLAY_COLORS,
                    )
                    fig_prob.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                    fig_prob.update_layout(
                        showlegend=False,
                        xaxis=dict(range=[0, 115], title="Xác suất (%)", gridcolor="#EBF2FA"),
                        yaxis=dict(title=""),
                        margin=dict(t=10, b=10, l=10, r=20),
                    )
                    apply_plotly_theme(fig_prob, height=195)
                    st.plotly_chart(fig_prob, use_container_width=True)
            else:
                st.info(
                    "Nhập nội dung bài đăng ở khung bên trái và nhấn **Dự đoán cảm xúc** "
                    "để xem nhãn dự đoán cùng xác suất từng lớp."
                )


# ==============================================================================
# ĐIỂM KHỞI CHẠY CHÍNH
# ==============================================================================
def main() -> None:
    inject_custom_css()

    try:
        raw_df = load_data(DATA_FILE_PATH)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Đã xảy ra lỗi khi đọc dữ liệu CSV: {exc}")
        st.stop()

    filtered_df, selected_model, selected_vectorizer = render_sidebar_filters(raw_df)

    ml_results = train_ml_models(
        texts=tuple(raw_df["text"].tolist()),
        labels=tuple(raw_df["sentiment"].tolist()),
        vectorizer_type=selected_vectorizer,
    )

    raw_with_pred = raw_df.copy()
    raw_with_pred["predicted_sentiment"] = ml_results[selected_model]["oof_predictions"]
    filtered_with_pred = raw_with_pred.loc[filtered_df.index].copy()

    # Header Banner đồng bộ phong cách Facebook Navy -> Indigo
    st.markdown(
        f"""
        <div class="dashboard-header">
            <div>
                <h1 class="dashboard-title">📊 Facebook Sentiment Analysis Dashboard</h1>
                <p class="dashboard-subtitle">
                    Phân tích cảm xúc bài đăng Facebook, đối chiếu mức độ tương tác và dự đoán bằng Machine Learning.
                </p>
            </div>
            <div class="dashboard-meta">
                {len(filtered_with_pred):,} / {len(raw_df):,} bài đăng · {selected_model} · {selected_vectorizer}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if filtered_with_pred.empty:
        st.warning(
            "Không có bài đăng nào phù hợp với các điều kiện lọc hiện tại. "
            "Vui lòng chọn thêm danh mục, cảm xúc, ngôn ngữ hoặc bỏ chọn loại bỏ spam ở thanh bên."
        )
        st.stop()

    # A. Thẻ KPI tổng quan trong khay xanh băng nhẹ
    render_kpi_section(filtered_with_pred)

    # Tổ chức nội dung chi tiết thành 3 Tab gọn gàng
    tab_viz, tab_data, tab_ml = st.tabs(
        [
            "Phân tích trực quan",
            f"Danh sách bài đăng ({len(filtered_with_pred):,})",
            "Dự đoán & Đánh giá ML",
        ]
    )

    with tab_viz:
        render_distribution_charts(filtered_with_pred)
        render_engagement_analysis(filtered_with_pred)

    with tab_data:
        render_posts_table_section(filtered_with_pred, selected_model)

    with tab_ml:
        render_live_prediction_section(ml_results, selected_model, selected_vectorizer)


if __name__ == "__main__":
    main()
