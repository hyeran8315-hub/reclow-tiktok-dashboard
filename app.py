
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Reclow TikTok Analytics",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv("reclow_tiktok_analysis.csv")
    df["createTimeISO"] = pd.to_datetime(
        df["createTimeISO"],
        errors="coerce"
    )
    return df

df = load_data()

st.title("Reclow TikTok Analytics Dashboard")
st.caption("TikTok 댓글 기반 브랜드 반응 분석")

# KPI
col1, col2, col3, col4 = st.columns(4)

col1.metric("전체 댓글 수", f"{len(df):,}")
col2.metric("평균 좋아요", f"{df['diggCount'].mean():.1f}")
col3.metric("분석 언어 수", df["language"].nunique())
col4.metric("영상 수", df["videoWebUrl"].nunique())

st.divider()

# 필터
st.sidebar.header("필터")

language_options = sorted(df["language"].dropna().unique())
sentiment_options = sorted(df["sentiment"].dropna().unique())
category_options = sorted(df["category"].dropna().unique())

selected_languages = st.sidebar.multiselect(
    "언어",
    language_options,
    default=language_options
)

selected_sentiments = st.sidebar.multiselect(
    "감성",
    sentiment_options,
    default=sentiment_options
)

selected_categories = st.sidebar.multiselect(
    "카테고리",
    category_options,
    default=category_options
)

search_text = st.sidebar.text_input("댓글 검색")

filtered_df = df[
    df["language"].isin(selected_languages)
    & df["sentiment"].isin(selected_sentiments)
    & df["category"].isin(selected_categories)
].copy()

if search_text:
    filtered_df = filtered_df[
        filtered_df["text"]
        .astype(str)
        .str.contains(search_text, case=False, na=False)
    ]

st.subheader("필터 적용 결과")
st.write(f"현재 표시 댓글 수: {len(filtered_df):,}")

# 차트
col1, col2 = st.columns(2)

with col1:
    st.subheader("감성 분포")
    sentiment_count = (
        filtered_df["sentiment"]
        .value_counts()
        .rename_axis("sentiment")
        .reset_index(name="count")
    )
    st.bar_chart(
        sentiment_count,
        x="sentiment",
        y="count"
    )

with col2:
    st.subheader("언어 분포")
    language_count = (
        filtered_df["language"]
        .value_counts()
        .rename_axis("language")
        .reset_index(name="count")
    )
    st.bar_chart(
        language_count,
        x="language",
        y="count"
    )

st.subheader("카테고리 분포")
category_count = (
    filtered_df["category"]
    .value_counts()
    .rename_axis("category")
    .reset_index(name="count")
)

st.bar_chart(
    category_count,
    x="category",
    y="count"
)

st.divider()

# 인기 댓글
st.subheader("좋아요가 많은 댓글")

top_comments = (
    filtered_df
    .sort_values("diggCount", ascending=False)
    .head(10)
)

st.dataframe(
    top_comments[
        [
            "text",
            "diggCount",
            "uniqueId",
            "language",
            "sentiment",
            "category",
            "videoWebUrl"
        ]
    ],
    use_container_width=True,
    hide_index=True
)

# 전체 댓글
st.subheader("전체 댓글 데이터")

st.dataframe(
    filtered_df[
        [
            "text",
            "diggCount",
            "uniqueId",
            "createTimeISO",
            "language",
            "sentiment",
            "category",
            "videoWebUrl"
        ]
    ],
    use_container_width=True,
    hide_index=True
)

# 다운로드
csv = filtered_df.to_csv(
    index=False,
    encoding="utf-8-sig"
).encode("utf-8-sig")

st.download_button(
    label="필터된 데이터 CSV 다운로드",
    data=csv,
    file_name="reclow_filtered_comments.csv",
    mime="text/csv"
)
