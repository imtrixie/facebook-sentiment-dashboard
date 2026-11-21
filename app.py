"""
Facebook Sentiment Analysis Dashboard
Ứng dụng phân tích cảm xúc bài đăng Facebook xây dựng bằng Python & Streamlit.
"""

from __future__ import annotations

import html
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
    page_icon="📊",
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
# TÙY CHỈNH GIAO DIỆN CSS
# ==============================================================================
def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

            :root {
                --ink: #0F172A;
                --muted: #64748B;
                --line: #E2E8F0;
                --bg: #F8FAFC;
                --card: #FFFFFF;
                --blue: #2563EB;
                --pos: #16A34A;
                --neu: #F59E0B;
                --neg: #DC2626;
            }
            html, body, [class*="css"], .stApp, button, input, textarea {
                font-family: 'Plus Jakarta Sans', -apple-system, 'Segoe UI', sans-serif;
            }
            [data-testid="stIconMaterial"] { font-family: "Material Symbols Rounded" !important; }
            .stApp { background: var(--bg); color: var(--ink); }
            header[data-testid="stHeader"] { background: transparent; }
            .block-container { padding-top: 3rem; padding-bottom: 3rem; max-width: 1440px; }
            .mono { font-family: 'IBM Plex Mono', ui-monospace, Menlo, monospace; }

            /* ---------- Top bar ---------- */
            .topbar {
                display: flex; align-items: center; gap: 0.85rem;
                background: var(--card); border: 1px solid var(--line);
                border-radius: 10px; padding: 0.75rem 1.1rem; margin-bottom: 1.6rem;
            }
            .topbar .logo {
                width: 38px; height: 38px; border-radius: 9px; background: var(--ink);
                display: flex; align-items: center; justify-content: center; flex: none;
            }
            .topbar .t1 { font-size: 1.05rem; font-weight: 700; line-height: 1.2; color: var(--ink); }
            .topbar .t2 { font-size: 0.78rem; color: var(--muted); }

            /* ---------- Section headings ---------- */
            .sec-title { font-size: 1.2rem; font-weight: 700; color: var(--ink); margin: 1.9rem 0 0.1rem 0; }
            .sec-cap { font-size: 0.9rem; color: var(--muted); margin-bottom: 1rem; }

            /* ---------- Cards (st.container(border=True)) ---------- */
            div[data-testid="stVerticalBlockBorderWrapper"] {
                background: var(--card);
                border: 1px solid var(--line) !important;
                border-radius: 10px;
                box-shadow: none;
            }
            .card-title {
                display: flex; justify-content: space-between; align-items: baseline;
                font-size: 0.95rem; font-weight: 700; color: var(--ink); margin-bottom: 0.4rem;
            }
            .card-title .right { font-size: 0.75rem; font-weight: 500; color: var(--muted); }

            /* ---------- KPI cards ---------- */
            .kpi {
                background: var(--card); border: 1px solid var(--line); border-radius: 10px;
                padding: 1rem 1.15rem; height: 100%;
            }
            .kpi .lbl { display: flex; align-items: center; gap: 0.45rem; font-size: 0.8rem; font-weight: 600; color: var(--muted); }
            .kpi .lbl.up { text-transform: uppercase; letter-spacing: 0.03em; font-size: 0.74rem; }
            .kpi .dot { width: 8px; height: 8px; border-radius: 2px; display: inline-block; }
            .kpi .val { font-size: 1.9rem; font-weight: 800; color: var(--ink); margin: 0.35rem 0 0.2rem 0; font-variant-numeric: tabular-nums; line-height: 1.15; }
            .kpi .val .pct { font-size: 0.95rem; font-weight: 700; margin-left: 0.35rem; }
            .kpi .sub { font-size: 0.78rem; color: var(--muted); }

            /* ---------- Legend cards next to donut ---------- */
            .leg { display: flex; align-items: center; gap: 0.6rem; border: 1px solid var(--line);
                   border-radius: 8px; padding: 0.55rem 0.75rem; margin-bottom: 0.55rem; }
            .leg .dot { width: 10px; height: 10px; border-radius: 3px; flex: none; }
            .leg .name { flex: 1; font-size: 0.85rem; font-weight: 500; color: var(--ink); }
            .leg .num { text-align: right; font-size: 0.95rem; font-weight: 700; color: var(--ink); line-height: 1.2; }
            .leg .num small { display: block; font-size: 0.75rem; font-weight: 500; color: var(--muted); }

            /* ---------- HTML tables ---------- */
            .tbl-wrap { overflow-x: auto; }
            table.tbl { width: 100%; border-collapse: collapse; font-size: 0.82rem; color: var(--ink); }
            table.tbl th { text-align: left; font-weight: 600; font-size: 0.76rem; color: #475569;
                           background: #F8FAFC; padding: 0.6rem 0.5rem; border-bottom: 1px solid var(--line); }
            table.tbl td { padding: 0.55rem 0.5rem; border-bottom: 1px solid #F1F5F9; vertical-align: middle; }
            table.tbl tr:last-child td { border-bottom: none; }
            table.tbl td:first-child { white-space: nowrap; }
            table.tbl .r { text-align: right; font-variant-numeric: tabular-nums; }
            table.tbl .id, table.tbl .ts { font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: 0.7rem; color: #475569; white-space: nowrap; }
            table.tbl .txt { min-width: 190px; }
            table.tbl small { display: block; color: var(--muted); font-size: 0.75rem; }
            table.tbl .dot { width: 8px; height: 8px; border-radius: 2px; display: inline-block; margin-right: 0.5rem; }
            .badge { display: inline-block; font-size: 0.74rem; font-weight: 600; padding: 0.12rem 0.5rem;
                     border-radius: 5px; border: 1px solid; white-space: nowrap; }
            .badge.positive { background: #F0FDF4; color: #15803D; border-color: #BBF7D0; }
            .badge.neutral  { background: #FFFBEB; color: #B45309; border-color: #FDE68A; }
            .badge.negative { background: #FEF2F2; color: #B91C1C; border-color: #FECACA; }
            .ok { color: #15803D; font-weight: 600; white-space: nowrap; }
            .bad { color: #B91C1C; font-weight: 600; white-space: nowrap; }
            .c-pos { color: #15803D; font-weight: 600; } .c-neu { color: #B45309; font-weight: 600; } .c-neg { color: #B91C1C; font-weight: 600; }
            .foot { font-size: 0.78rem; color: var(--muted); margin-top: 0.6rem; }
            .stat-line { font-size: 0.85rem; color: #475569; margin: 0.5rem 0 0.6rem 0; }
            .stat-line b { color: var(--ink); }

            /* ---------- Prediction result ---------- */
            .pred-box { border: 1px solid var(--line); border-radius: 8px; background: #F8FAFC; padding: 0.9rem 1rem; margin-bottom: 1rem; }
            .pred-box .k { font-size: 0.78rem; color: var(--muted); }
            .pred-box .v { font-size: 1.5rem; font-weight: 800; margin: 0.15rem 0; }
            .pred-box .c { font-size: 0.82rem; color: #334155; }
            .pbar { margin-bottom: 0.65rem; }
            .pbar .row { display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.2rem; color: var(--ink); }
            .pbar .row b { font-variant-numeric: tabular-nums; }
            .pbar .track { height: 9px; border-radius: 5px; background: #EEF2F6; overflow: hidden; }
            .pbar .fill { height: 100%; border-radius: 5px; }

            /* ---------- Sidebar ---------- */
            section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid var(--line); }
            section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
            .sb-title { display: flex; align-items: center; gap: 0.55rem; font-size: 1.05rem; font-weight: 700; color: var(--ink); margin: 0.2rem 0 0.9rem 0; }
            .sb-h { font-size: 0.74rem; font-weight: 700; letter-spacing: 0.05em; color: #64748B; text-transform: uppercase; padding-top: 0.5rem; }
            .sb-foot { font-size: 0.82rem; color: #475569; border-top: 1px solid var(--line); padding-top: 0.8rem; margin-top: 0.8rem; }
            section[data-testid="stSidebar"] .stButton > button {
                background: none; border: none; box-shadow: none; color: var(--blue);
                font-size: 0.75rem; font-weight: 500; padding: 0; min-height: 0; height: auto; float: right;
            }
            section[data-testid="stSidebar"] .stButton > button:hover { background: none; text-decoration: underline; }
            section[data-testid="stSidebar"] label p { font-size: 0.85rem; }

            /* ---------- Inputs & buttons ---------- */
            div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, textarea { border-radius: 8px !important; border-color: var(--line) !important; }
            div[data-baseweb="tag"] { background: #EFF6FF !important; border-radius: 6px !important; }
            div[data-baseweb="tag"] span { color: #1D4ED8 !important; font-weight: 600; }
            .stButton > button, .stDownloadButton > button { border-radius: 8px; font-weight: 600; font-size: 0.85rem; }
            .stButton > button[kind="primary"] { background: var(--blue); border: none; color: #fff; }
            .stButton > button[kind="primary"]:hover { background: #1D4ED8; }
            .stDownloadButton > button { background: var(--ink); color: #fff; border: none; }
            .stDownloadButton > button:hover { background: #1E293B; color: #fff; }
            .stDownloadButton > button p { color: #fff; }
            div[data-testid="stSegmentedControl"] button { font-size: 0.8rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# HÀM TIỆN ÍCH HIỂN THỊ
# ==============================================================================
def _h(markup: str) -> str:
    """Gộp HTML thành một dòng để Markdown không hiểu nhầm thụt lề là code block."""
    return "".join(line.strip() for line in markup.splitlines())


def fmt_int(n: float) -> str:
    return f"{int(round(n)):,}".replace(",", ".")


def fmt_num(x: float, digits: int = 1) -> str:
    return f"{x:,.{digits}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def esc(text: Any) -> str:
    return html.escape(str(text)).replace("$", "&#36;")


def section_header(title: str, caption: str) -> None:
    st.markdown(
        _h(f'<div class="sec-title">{esc(title)}</div><div class="sec-cap">{caption}</div>'),
        unsafe_allow_html=True,
    )


def card_title(title: str, right: str = "") -> None:
    st.markdown(
        _h(f'<div class="card-title"><span>{esc(title)}</span><span class="right">{right}</span></div>'),
        unsafe_allow_html=True,
    )


def style_figure(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    fig.update_layout(
        font=dict(family="Plus Jakarta Sans, sans-serif", size=12, color="#334155"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        separators=",.",
        height=height,
        margin=dict(t=20, b=10, l=10, r=10),
        hoverlabel=dict(bgcolor="#0F172A", font_color="#FFFFFF", bordercolor="#0F172A"),
    )
    return fig


def show_fig(fig: go.Figure) -> None:
    st.plotly_chart(fig, theme=None, width="stretch", config={"displayModeBar": False})


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

    # Chuẩn hóa thời gian sang datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Xử lý dữ liệu văn bản & phân loại bị thiếu (giữ nguyên nhãn sentiment gốc)
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

    # Chuyển đổi kiểu số an toàn cho các cột tương tác và cờ boolean
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

        # Đánh giá độc lập trên tập Test (20%)
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

        # Dự đoán ngoài mẫu (Out-of-Fold) cho bảng dữ liệu bài đăng
        # Mỗi bài đăng được dự đoán bởi mô hình KHÔNG nhìn thấy bài đăng đó lúc huấn luyện
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
# CÁC HÀM TRỰC QUAN HÓA & HIỂN THỊ
# ==============================================================================
SAMPLE_TEXTS = {
    "positive": "Absolutely love this new product update! Works smoothly and saved my day 🎉",
    "negative": "Still waiting for support to reply about my login issue, really frustrating experience.",
    "neutral": "Tech conference scheduled for next Thursday in downtown hall starting at 9 AM.",
}
FILTER_ICON = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0F172A" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>'
)
LOGO_ICON = (
    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M8 17v-5"/><path d="M13 17V8"/><path d="M18 17v-3"/></svg>'
)


def render_topbar(n_rows: int) -> None:
    st.markdown(
        _h(
            f"""
            <div class="topbar">
                <div class="logo">{LOGO_ICON}</div>
                <div>
                    <div class="t1">Facebook Sentiment Analysis Dashboard</div>
                    <div class="t2">Nguồn dữ liệu: <span class="mono">{esc(DATA_FILE_PATH)}</span> · {fmt_int(n_rows)} bản ghi</div>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def _toggle_all(keys: list[str], values: list[Any], select: bool) -> None:
    for key, val in zip(keys, values):
        st.session_state[key] = val if select else (False if isinstance(val, bool) else [])


def _sidebar_header_with_toggle(title: str, all_selected: bool, on_click, btn_key: str) -> None:
    c1, c2 = st.sidebar.columns([1.6, 1])
    c1.markdown(f'<div class="sb-h">{title}</div>', unsafe_allow_html=True)
    c2.button(
        "Bỏ chọn tất cả" if all_selected else "Chọn tất cả",
        key=btn_key,
        on_click=on_click,
        args=(not all_selected,),
    )


def render_sidebar_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    st.sidebar.markdown(
        _h(f'<div class="sb-title">{FILTER_ICON}<span>Bộ lọc dữ liệu</span></div>'),
        unsafe_allow_html=True,
    )

    # --- Mô hình & đặc trưng ---
    st.sidebar.markdown('<div class="sb-h">Mô hình dự đoán (ML)</div>', unsafe_allow_html=True)
    model_labels = {"Naive Bayes": "Naive Bayes (MultinomialNB)", "Decision Tree": "Decision Tree"}
    selected_model = st.sidebar.selectbox(
        "Thuật toán phân loại",
        options=list(model_labels),
        format_func=lambda m: model_labels[m],
        help="Mô hình dùng cho cột Nhãn dự đoán trong bảng và khu vực dự đoán trực tiếp.",
    )
    st.sidebar.caption("Trích xuất đặc trưng")
    selected_vectorizer = st.sidebar.segmented_control(
        "Trích xuất đặc trưng",
        options=["TF-IDF", "Bag of Words"],
        default="TF-IDF",
        label_visibility="collapsed",
        width="stretch",
    ) or "TF-IDF"

    # --- Nhóm cảm xúc (checkbox) ---
    sent_keys = [f"sent_{s}" for s in SENTIMENT_ORDER]
    for k in sent_keys:
        st.session_state.setdefault(k, True)
    all_sent = all(st.session_state[k] for k in sent_keys)
    _sidebar_header_with_toggle(
        "Nhóm cảm xúc",
        all_sent,
        lambda sel: _toggle_all(sent_keys, [True] * 3, sel),
        "btn_sent",
    )
    dots = {"positive": "🟩", "neutral": "🟧", "negative": "🟥"}
    selected_sentiments = [
        s
        for s in SENTIMENT_ORDER
        if st.sidebar.checkbox(f"{dots[s]}  {SENTIMENT_LABELS_VI[s]}", key=f"sent_{s}")
    ]

    # --- Danh mục ---
    all_categories = sorted(df["category"].unique().tolist())
    st.session_state.setdefault("flt_cat", all_categories)
    n_cat = len(st.session_state["flt_cat"])
    _sidebar_header_with_toggle(
        f"Danh mục ({n_cat}/{len(all_categories)})",
        n_cat == len(all_categories),
        lambda sel: _toggle_all(["flt_cat"], [all_categories], sel),
        "btn_cat",
    )
    selected_categories = st.sidebar.multiselect(
        "Danh mục bài đăng (Category)", options=all_categories, key="flt_cat", label_visibility="collapsed"
    )

    # --- Ngôn ngữ ---
    all_languages = sorted(df["language"].unique().tolist())
    st.session_state.setdefault("flt_lang", all_languages)
    n_lang = len(st.session_state["flt_lang"])
    _sidebar_header_with_toggle(
        f"Ngôn ngữ ({n_lang}/{len(all_languages)})",
        n_lang == len(all_languages),
        lambda sel: _toggle_all(["flt_lang"], [all_languages], sel),
        "btn_lang",
    )
    selected_languages = st.sidebar.multiselect(
        "Ngôn ngữ (Language)", options=all_languages, key="flt_lang", label_visibility="collapsed"
    )

    st.sidebar.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    exclude_spam = st.sidebar.checkbox("Loại bỏ bài đăng spam (is_spam = 1)", value=False)

    mask = (
        df["category"].isin(selected_categories)
        & df["sentiment"].isin(selected_sentiments)
        & df["language"].isin(selected_languages)
    )
    if exclude_spam:
        mask = mask & (df["is_spam"] == 0)
    filtered_df = df[mask].copy()

    st.sidebar.markdown(
        _h(f'<div class="sb-foot">Đang hiển thị <b>{fmt_int(len(filtered_df))}</b> / {fmt_int(len(df))} bài đăng</div>'),
        unsafe_allow_html=True,
    )
    return filtered_df, selected_model, selected_vectorizer


def render_kpi_section(filtered_df: pd.DataFrame) -> None:
    section_header("A. Thống kê tổng quan", "Chỉ số KPI cập nhật đồng bộ theo bộ lọc hiện tại.")

    total = len(filtered_df)
    cards = [("Tổng số bài đăng", None, total, 100.0, "#0F172A", "100.0% dữ liệu đang lọc")]
    pct_colors = {"positive": "#15803D", "neutral": "#B45309", "negative": "#B91C1C"}
    names = {"positive": "Tích cực (Positive)", "neutral": "Trung lập (Neutral)", "negative": "Tiêu cực (Negative)"}
    for s in SENTIMENT_ORDER:
        n = int((filtered_df["sentiment"] == s).sum())
        pct = n / total * 100 if total else 0.0
        cards.append((names[s], SENTIMENT_COLORS[s], n, pct, pct_colors[s], f"Chiếm {pct:.1f}% tổng bài đăng"))

    cols = st.columns(4)
    for col, (label, dot, n, pct, pct_color, sub) in zip(cols, cards):
        dot_html = f'<span class="dot" style="background:{dot}"></span>' if dot else ""
        pct_html = f'<span class="pct" style="color:{pct_color}">({pct:.1f}%)</span>' if dot else ""
        col.markdown(
            _h(
                f"""<div class="kpi"><div class="lbl">{dot_html}{label}</div>
                <div class="val">{fmt_int(n)}{pct_html}</div><div class="sub">{sub}</div></div>"""
            ),
            unsafe_allow_html=True,
        )


def render_distribution_charts(filtered_df: pd.DataFrame) -> None:
    section_header(
        "B. Biểu đồ phân bố cảm xúc",
        "So sánh số lượng và tỷ trọng giữa ba nhóm cảm xúc (Positive: Xanh lá, Neutral: Cam, Negative: Đỏ).",
    )
    counts = filtered_df["sentiment"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    total = int(counts.sum())
    pcts = [(c / total * 100 if total else 0.0) for c in counts]
    colors = [SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER]

    col_bar, col_pie = st.columns(2)

    with col_bar:
        with st.container(border=True):
            card_title("Số lượng bài đăng theo nhóm cảm xúc", f"Tổng: {fmt_int(total)} bài")
            fig = go.Figure(
                go.Bar(
                    x=[f"{SENTIMENT_SHORT_VI[s]}<br><span style='color:#94A3B8'>({p:.1f}%)</span>" for s, p in zip(SENTIMENT_ORDER, pcts)],
                    y=counts.values,
                    marker_color=colors,
                    text=counts.values,
                    texttemplate="<b>%{text:,}</b>",
                    textposition="outside",
                    cliponaxis=False,
                    width=0.5,
                    hovertemplate="<b>%{x}</b><br>Số lượng: %{y:,}<extra></extra>",
                )
            )
            fig.update_yaxes(gridcolor="#E2E8F0", griddash="dash", zeroline=False, tickformat=",d", rangemode="tozero")
            fig.update_xaxes(showgrid=False, linecolor="#E2E8F0")
            show_fig(style_figure(fig, 360))

    with col_pie:
        with st.container(border=True):
            card_title("Tỷ trọng phần trăm các nhóm cảm xúc")
            c_chart, c_leg = st.columns([1.15, 1])
            with c_chart:
                fig = go.Figure(
                    go.Pie(
                        labels=[SENTIMENT_SHORT_VI[s] for s in SENTIMENT_ORDER],
                        values=counts.values,
                        hole=0.68,
                        marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
                        textinfo="none",
                        sort=False,
                        direction="clockwise",
                        hovertemplate="<b>%{label}</b><br>Số lượng: %{value:,}<br>Tỷ trọng: %{percent}<extra></extra>",
                    )
                )
                fig.add_annotation(text="Tổng bài", showarrow=False, x=0.5, y=0.58, font=dict(size=12, color="#64748B"))
                fig.add_annotation(text=f"<b>{fmt_int(total)}</b>", showarrow=False, x=0.5, y=0.45, font=dict(size=22, color="#0F172A"))
                fig.update_layout(showlegend=False)
                show_fig(style_figure(fig, 320))
            with c_leg:
                st.markdown("<div style='height:2.2rem'></div>", unsafe_allow_html=True)
                for s, n, p in zip(SENTIMENT_ORDER, counts.values, pcts):
                    st.markdown(
                        _h(
                            f"""<div class="leg"><span class="dot" style="background:{SENTIMENT_COLORS[s]}"></span>
                            <span class="name">{SENTIMENT_LABELS_VI[s]}</span>
                            <span class="num">{p:.1f}%<small>({fmt_int(n)})</small></span></div>"""
                        ),
                        unsafe_allow_html=True,
                    )


def render_engagement_analysis(filtered_df: pd.DataFrame) -> None:
    h_left, h_right = st.columns([2, 1.25])
    with h_left:
        section_header(
            "C. Phân tích mức độ tương tác theo cảm xúc",
            "So sánh Likes, Comments và Shares giữa ba nhóm cảm xúc qua biểu đồ cột nhóm và bảng tổng hợp.",
        )
    with h_right:
        st.markdown("<div style='height:1.1rem'></div>", unsafe_allow_html=True)
        mode = st.segmented_control(
            "Chế độ phân tích tương tác",
            options=["Trung bình / bài đăng", "Tổng lượng tương tác"],
            default="Trung bình / bài đăng",
            label_visibility="collapsed",
            width="stretch",
            key="eng_mode",
        ) or "Trung bình / bài đăng"
    use_mean = mode.startswith("Trung bình")

    agg = (
        filtered_df.groupby("sentiment")
        .agg(
            n=("post_id", "count"),
            likes_sum=("likes", "sum"),
            comments_sum=("comments", "sum"),
            shares_sum=("shares", "sum"),
            likes_mean=("likes", "mean"),
            comments_mean=("comments", "mean"),
            shares_mean=("shares", "mean"),
        )
        .reindex(SENTIMENT_ORDER)
        .fillna(0)
    )
    suffix = "mean" if use_mean else "sum"
    chart_title = "Trung bình tương tác trên mỗi bài đăng" if use_mean else "Tổng lượng tương tác"

    col_chart, col_table = st.columns([1.15, 1])
    with col_chart:
        with st.container(border=True):
            card_title(chart_title)
            fig = go.Figure()
            for metric, name, color in [
                ("likes", "Likes", "#2563EB"),
                ("comments", "Comments", "#0EA5E9"),
                ("shares", "Shares", "#64748B"),
            ]:
                vals = agg[f"{metric}_{suffix}"]
                fig.add_bar(
                    x=[SENTIMENT_SHORT_VI[s] for s in SENTIMENT_ORDER],
                    y=vals,
                    name=name,
                    marker_color=color,
                    text=vals,
                    texttemplate="%{text:.1f}" if use_mean else "%{text:,.0f}",
                    textposition="outside",
                    cliponaxis=False,
                )
            fig.update_layout(
                barmode="group",
                bargap=0.3,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            fig.update_yaxes(gridcolor="#E2E8F0", griddash="dash", zeroline=False, rangemode="tozero")
            fig.update_xaxes(showgrid=False, linecolor="#E2E8F0")
            fig = style_figure(fig, 380)
            fig.update_layout(margin=dict(t=40, b=10, l=10, r=10))
            show_fig(fig)

    with col_table:
        with st.container(border=True):
            card_title("Bảng thống kê tổng hợp chỉ số tương tác")
            rows = ""
            for s in SENTIMENT_ORDER:
                r = agg.loc[s]
                cells = ""
                for metric in ["likes", "comments", "shares"]:
                    cells += (
                        f'<td class="r"><b>{fmt_num(r[metric + "_mean"])}</b>'
                        f"<small>{fmt_int(r[metric + '_sum'])}</small></td>"
                    )
                rows += (
                    f'<tr><td><span class="dot" style="background:{SENTIMENT_COLORS[s]}"></span>{SENTIMENT_SHORT_VI[s]}</td>'
                    f'<td class="r">{fmt_int(r["n"])}</td>{cells}</tr>'
                )
            st.markdown(
                _h(
                    f"""<div class="tbl-wrap"><table class="tbl"><thead><tr><th>Cảm xúc</th><th class="r">Số bài</th>
                    <th class="r">Likes (TB / Tổng)</th><th class="r">Comments (TB / Tổng)</th><th class="r">Shares (TB / Tổng)</th>
                    </tr></thead><tbody>{rows}</tbody></table></div>
                    <div class="foot">Dòng trên biểu thị giá trị trung bình mỗi bài đăng; dòng dưới biểu thị tổng lượt tương tác tích lũy.</div>"""
                ),
                unsafe_allow_html=True,
            )


def _badge(label_key: str) -> str:
    return f'<span class="badge {label_key}">{SENTIMENT_SHORT_VI[label_key]}</span>'


def render_posts_table_section(filtered_df: pd.DataFrame, selected_model: str) -> None:
    section_header(
        "E. Danh sách bài đăng và kết quả dự đoán cảm xúc",
        f"So sánh trực quan nhãn cảm xúc thực tế và nhãn dự đoán ngoài mẫu từ mô hình <b>{esc(selected_model)}</b>.",
    )
    page_size = 20

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([2.3, 1.5, 1.9, 1.1], vertical_alignment="center")
        keyword = c1.text_input(
            "Tìm kiếm",
            placeholder="🔍  Tìm theo từ khóa nội dung hoặc ID bài đăng...",
            label_visibility="collapsed",
        )
        only_wrong = c2.checkbox("Chỉ hiển thị bài dự đoán sai", value=False)
        sort_order = c3.selectbox(
            "Sắp xếp",
            options=["Thời gian: Mới nhất trước", "Thời gian: Cũ nhất trước"],
            label_visibility="collapsed",
        )
        download_slot = c4.empty()

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
    table_df = table_df.sort_values(
        by="timestamp", ascending=sort_order.endswith("Cũ nhất trước"), na_position="last"
    )

    if table_df.empty:
        st.warning("Không tìm thấy bài đăng nào khớp với điều kiện tìm kiếm hiện tại.")
        return

    # CSV xuất ra toàn bộ dữ liệu đã lọc
    export_df = pd.DataFrame(
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
    download_slot.download_button(
        label=f"📥 Tải CSV ({fmt_int(len(export_df))})",
        data=export_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="facebook_sentiment_filtered.csv",
        mime="text/csv",
        width="stretch",
    )

    total_rows = len(table_df)
    correct = int(table_df["is_correct"].sum())
    total_pages = max(1, -(-total_rows // page_size))
    sig = (keyword, only_wrong, sort_order, selected_model, total_rows)
    if st.session_state.get("posts_sig") != sig:
        st.session_state["posts_sig"] = sig
        st.session_state["posts_page"] = 1
    page = min(max(int(st.session_state.get("posts_page", 1)), 1), total_pages)
    st.session_state["posts_page"] = page

    st.markdown(
        _h(
            f"""<div class="stat-line">Hiển thị <b>{fmt_int(total_rows)}</b> bài đăng · Dự đoán đúng:
            <span class="c-pos">{fmt_int(correct)}</span> ({correct / total_rows * 100:.1f}%) · Dự đoán sai:
            <span class="c-neg">{fmt_int(total_rows - correct)}</span></div>"""
        ),
        unsafe_allow_html=True,
    )

    start = (page - 1) * page_size
    page_df = table_df.iloc[start : start + page_size]
    rows = ""
    for _, r in page_df.iterrows():
        ts = r["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if pd.notna(r["timestamp"]) else "N/A"
        verdict = '<span class="ok">✓ Đúng</span>' if r["is_correct"] else '<span class="bad">✕ Sai</span>'
        rows += (
            f'<tr><td class="id">#{esc(r["post_id"])}</td><td class="ts">{ts}</td><td>{esc(r["category"])}</td>'
            f'<td class="txt">{esc(r["text"])}</td><td>{_badge(r["sentiment"])}</td>'
            f'<td>{_badge(r["predicted_sentiment"])}</td><td>{verdict}</td>'
            f'<td class="r">{fmt_int(r["likes"])}</td><td class="r">{fmt_int(r["comments"])}</td>'
            f'<td class="r">{fmt_int(r["shares"])}</td></tr>'
        )
    with st.container(border=True):
        st.markdown(
            _h(
                f"""<div class="tbl-wrap"><table class="tbl"><thead><tr><th>ID</th><th>Thời gian đăng</th><th>Danh mục</th>
                <th>Nội dung bài đăng</th><th>Nhãn thực tế</th><th>Nhãn dự đoán</th><th>Đánh giá</th>
                <th class="r">Likes</th><th class="r">Comments</th><th class="r">Shares</th></tr></thead>
                <tbody>{rows}</tbody></table></div>"""
            ),
            unsafe_allow_html=True,
        )
        f1, f2, f3, f4 = st.columns([5, 1.6, 1, 1], vertical_alignment="center")
        f1.markdown(
            f'<div class="foot">Dòng {fmt_int(start + 1)} – {fmt_int(start + len(page_df))} trong {fmt_int(total_rows)}</div>',
            unsafe_allow_html=True,
        )
        f2.markdown(f'<div class="foot" style="text-align:right">Trang {page} / {fmt_int(total_pages)}</div>', unsafe_allow_html=True)
        f3.button("‹ Trước", disabled=page <= 1, key="pg_prev", width="stretch",
                  on_click=lambda: st.session_state.update(posts_page=page - 1))
        f4.button("Tiếp ›", disabled=page >= total_pages, key="pg_next", width="stretch",
                  on_click=lambda: st.session_state.update(posts_page=page + 1))


def _set_sample(kind: str) -> None:
    st.session_state["live_text"] = SAMPLE_TEXTS[kind]


def render_live_prediction_section(
    ml_results: dict[str, Any],
    default_model: str,
    vectorizer_type: str,
) -> None:
    m1, m2 = ml_results["Naive Bayes"], ml_results["Decision Tree"]
    section_header(
        "F. Dự đoán cảm xúc bài đăng mới (Machine Learning)",
        f"Huấn luyện bằng Train/Test Split (80%/20%) với trích xuất đặc trưng <b>{esc(vectorizer_type)}</b>. "
        f"Đánh giá độc lập trên tập kiểm thử (Test set n = {fmt_int(m1['test_size'])}).",
    )

    c1, c2, c3 = st.columns([1, 1, 1.6])
    for col, name, m in [(c1, "Naive Bayes", m1), (c2, "Decision Tree", m2)]:
        col.markdown(
            _h(
                f"""<div class="kpi"><div class="lbl up">{name} · {esc(vectorizer_type)}</div>
                <div class="val">{m['test_accuracy'] * 100:.2f}%</div>
                <div class="sub">Accuracy trên Test set · Macro F1: {m['test_f1_macro'] * 100:.2f}%</div></div>"""
            ),
            unsafe_allow_html=True,
        )
    rows = ""
    for name, m in [("Naive Bayes", m1), ("Decision Tree", m2)]:
        rep = m["classification_report"]
        rows += (
            f'<tr><td><b>{name}</b></td><td class="r">{m["test_accuracy"] * 100:.2f}%</td>'
            f'<td class="r c-pos">{rep["positive"]["f1-score"] * 100:.1f}%</td>'
            f'<td class="r c-neu">{rep["neutral"]["f1-score"] * 100:.1f}%</td>'
            f'<td class="r c-neg">{rep["negative"]["f1-score"] * 100:.1f}%</td></tr>'
        )
    c3.markdown(
        _h(
            f"""<div class="kpi" style="padding:0.6rem 0.9rem"><table class="tbl"><thead><tr><th>Mô hình</th>
            <th class="r">Accuracy</th><th class="r">F1 Positive</th><th class="r">F1 Neutral</th><th class="r">F1 Negative</th></tr></thead>
            <tbody>{rows}</tbody></table></div>"""
        ),
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.session_state.setdefault("live_text", SAMPLE_TEXTS["positive"])
    left, right = st.columns([1.5, 1])

    with left:
        with st.container(border=True):
            card_title("Nhập nội dung bài đăng Facebook mới")
            b1, b2, b3, _sp = st.columns([1, 1, 1.15, 1.3])
            b1.button("Mẫu Tích cực", on_click=_set_sample, args=("positive",), width="stretch")
            b2.button("Mẫu Tiêu cực", on_click=_set_sample, args=("negative",), width="stretch")
            b3.button("Mẫu Trung lập", on_click=_set_sample, args=("neutral",), width="stretch")
            user_text = st.text_area(
                "Nội dung bài đăng",
                key="live_text",
                height=130,
                label_visibility="collapsed",
                placeholder="Ví dụ: Great service and friendly team, thank you so much! ❤️",
            )
            r1, r2 = st.columns([2, 1], vertical_alignment="center")
            pred_model_name = r1.radio(
                "Mô hình:",
                options=["Naive Bayes", "Decision Tree"],
                index=0 if default_model == "Naive Bayes" else 1,
                horizontal=True,
            )
            r2.button("✨ Dự đoán cảm xúc", type="primary", width="stretch")

    with right:
        with st.container(border=True):
            card_title(f"Kết quả & Xác suất dự đoán ({pred_model_name})")
            cleaned = user_text.strip()
            if not cleaned:
                st.info("Nhập nội dung bài đăng ở khung bên trái để xem nhãn dự đoán và xác suất chi tiết.")
                return
            pipeline: Pipeline = ml_results[pred_model_name]["pipeline"]
            pred_label = str(pipeline.predict([cleaned])[0])
            proba_map = {str(c): float(p) for c, p in zip(pipeline.classes_, pipeline.predict_proba([cleaned])[0])}
            color = SENTIMENT_COLORS.get(pred_label, "#2563EB")
            bars = "".join(
                f"""<div class="pbar"><div class="row"><span>{SENTIMENT_LABELS_VI[c]}</span><b>{proba_map.get(c, 0) * 100:.1f}%</b></div>
                <div class="track"><div class="fill" style="width:{proba_map.get(c, 0) * 100:.1f}%;background:{SENTIMENT_COLORS[c]}"></div></div></div>"""
                for c in SENTIMENT_ORDER
            )
            st.markdown(
                _h(
                    f"""<div class="pred-box"><div class="k">Nhãn cảm xúc dự đoán</div>
                    <div class="v" style="color:{color}">{SENTIMENT_LABELS_VI.get(pred_label, pred_label)}</div>
                    <div class="c">Độ tin cậy: <b>{proba_map.get(pred_label, 0) * 100:.2f}%</b></div></div>{bars}"""
                ),
                unsafe_allow_html=True,
            )


# ==============================================================================
# ĐIỂM KHỞI CHẠY CHÍNH CỦA ỨNG DỤNG
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

    render_topbar(len(raw_df))
    filtered_df, selected_model, selected_vectorizer = render_sidebar_filters(raw_df)

    ml_results = train_ml_models(
        texts=tuple(raw_df["text"].tolist()),
        labels=tuple(raw_df["sentiment"].tolist()),
        vectorizer_type=selected_vectorizer,
    )

    raw_with_pred = raw_df.copy()
    raw_with_pred["predicted_sentiment"] = ml_results[selected_model]["oof_predictions"]
    filtered_with_pred = raw_with_pred.loc[filtered_df.index].copy()

    if filtered_with_pred.empty:
        st.warning(
            "⚠️ Không có bài đăng nào phù hợp với các điều kiện lọc hiện tại. "
            "Vui lòng chọn thêm danh mục, cảm xúc, ngôn ngữ hoặc bỏ chọn loại bỏ spam ở thanh bên."
        )
        st.stop()

    render_kpi_section(filtered_with_pred)
    render_distribution_charts(filtered_with_pred)
    render_engagement_analysis(filtered_with_pred)
    render_posts_table_section(filtered_with_pred, selected_model)
    render_live_prediction_section(ml_results, selected_model, selected_vectorizer)


if __name__ == "__main__":
    main()
