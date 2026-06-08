"""
B站热榜交互式数据看板 (Streamlit)
包含：大盘宏观追踪、高质量内容挖掘、流量密码解码、单视频时序追踪 四大模块。
"""

import re
from pathlib import Path

import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- 可选依赖 ---
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

st.set_page_config(page_title="B站热榜数据看板", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 28px; font-weight: bold; margin-bottom: 0; }
    .metric-card { background: #f8f9fa; border-radius: 8px; padding: 16px; text-align: center; }
    .metric-value { font-size: 24px; font-weight: bold; color: #00a1d6; }
    .metric-label { font-size: 13px; color: #666; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = Path(__file__).parent / "data" / "cleaned"
DEFAULT_FILE = DATA_DIR / "cleaned_bilibili_ranking.csv"

STOPWORDS = set([
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
    "什么", "怎么", "如何", "为什么", "吗", "吧", "啊", "呢", "哦", "嗯",
    "可以", "这个", "那个", "还是", "但是", "如果", "因为", "所以",
    "真的", "感觉", "觉得", "应该", "可能", "已经", "一直",
    "终于", "居然", "竟然", "全网", "第一", "整个",
    "挑战", "视频", "一个", "今天", "现在", "看到", "简直", "原来",
    "vlog", "不要", "不能", "[", "]", "!", "?", ".", ",",
    " ", "\t", "\n", "|", "/",
])


def load_data(filepath=None):
    fp = filepath or DEFAULT_FILE
    if not fp.exists():
        candidates = sorted(DATA_DIR.glob("*.csv"))
        if candidates:
            fp = candidates[-1]
        else:
            st.error(f"No data file: {DATA_DIR}")
            st.stop()
    df = pd.read_csv(fp, encoding="utf-8-sig")
    df["rank_date"] = pd.to_datetime(df["rank_date"], errors="coerce")
    return df


@st.cache_data
def compute_category_trends(df):
    trend = df.groupby(["rank_date", "main_category"], as_index=False).size()
    trend.columns = ["rank_date", "main_category", "count"]
    return trend


@st.cache_data
def generate_word_frequencies(df, top_k=80):
    titles = df["title"].dropna().astype(str).tolist()
    if not titles:
        return {}
    cleaned = []
    for t in titles:
        t = re.sub(r"[\[\]{}()【】《》""''#@&*]", "", t)
        t = re.sub(r"[0-9a-zA-Z]+", "", t)
        cleaned.append(t)
    text = " ".join(cleaned)
    if JIEBA_AVAILABLE:
        words = jieba.cut(text, cut_all=False)
    else:
        words = list(text.replace(" ", ""))
    freq = {}
    for w in words:
        w = w.strip()
        if len(w) < 2 or w in STOPWORDS:
            continue
        freq[w] = freq.get(w, 0) + 1
    return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_k])


def render_wordcloud(freq, ax=None):
    if not freq or not WORDCLOUD_AVAILABLE:
        return None
    wc = WordCloud(
        font_path="C:/Windows/Fonts/msyh.ttc",
        width=800, height=400, background_color="white",
        max_words=80, colormap="viridis", prefer_horizontal=0.7,
    )
    wc.generate_from_frequencies(freq)
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return ax


# ====== 加载数据 ======
st.markdown('<p class="main-header"> B站全站热榜 数据洞察看板</p>', unsafe_allow_html=True)
df = load_data()
cat_col = "main_category" if "main_category" in df.columns else "category"

# ====== 侧边栏筛选 ======
st.sidebar.header(" 筛选条件")
if "rank_date" in df.columns and df["rank_date"].notna().any():
    min_date = df["rank_date"].min().date()
    max_date = df["rank_date"].max().date()
    date_range = st.sidebar.date_input("日期范围", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df[(df["rank_date"] >= pd.Timestamp(start_date)) & (df["rank_date"] <= pd.Timestamp(end_date))].copy()
    else:
        df_filtered = df.copy()
else:
    df_filtered = df.copy()

categories = sorted(df_filtered[cat_col].dropna().unique().tolist())
default_cats = categories[:8] if len(categories) > 8 else categories
selected_cats = st.sidebar.multiselect("一级分区", categories, default=default_cats,
                                       help="B站官方一级分区：动画、音乐、游戏、知识、生活、娱乐、影视 等")
if selected_cats:
    df_filtered = df_filtered[df_filtered[cat_col].isin(selected_cats)]

if "category" in df_filtered.columns:
    sub_cats = sorted(df_filtered["category"].dropna().unique().tolist())
    selected_sub = st.sidebar.multiselect("二级分区（可选）", sub_cats, default=[])
    if selected_sub:
        df_filtered = df_filtered[df_filtered["category"].isin(selected_sub)]

if "duration_bucket" in df_filtered.columns:
    buckets = sorted(df_filtered["duration_bucket"].dropna().unique().tolist())
    selected_buckets = st.sidebar.multiselect("时长区间", buckets, default=buckets)
    if selected_buckets:
        df_filtered = df_filtered[df_filtered["duration_bucket"].isin(selected_buckets)]

show_anomaly = st.sidebar.checkbox("显示异常数据", value=False)
if not show_anomaly and "anomaly_flag" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["anomaly_flag"] == "normal"]

st.sidebar.caption(f"当前筛选: {len(df_filtered)} 条数据")

# v2: 模块1-3用去重后数据（每个视频只保留最新快照）
df_latest = df_filtered.sort_values("snapshot_time").groupby("bvid", as_index=False).last()

# ====== KPI 卡片 ======
st.markdown("---")
cols = st.columns(5)
metrics = [
    ("视频数", f"{len(df_latest):,}"),
    ("UP主数", f"{df_latest['author'].nunique():,}"),
    ("平均播放量", f"{df_latest['view_count'].mean():,.0f}" if len(df_latest) > 0 else "0"),
    ("三连互动率", f"{df_latest['engage_rate'].mean():.4f}" if 'engage_rate' in df_latest.columns else "N/A"),
    ("一级分区数", f"{df_latest[cat_col].nunique()}"),
]
for col, (label, value) in zip(cols, metrics):
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

# ====== 模块1：大盘宏观追踪 ======
st.markdown("---")
st.subheader(" 模块一：大盘宏观追踪 — 各一级分区每日上榜视频数量趋势")

if len(df_latest) > 0 and "rank_date" in df_latest.columns:
    trend_df = compute_category_trends(df_latest)
    tab1, tab2 = st.tabs([" 堆叠面积图", " 数据表"])
    with tab1:
        pivot = trend_df.pivot_table(index="rank_date", columns="main_category", values="count", fill_value=0)
        if not pivot.empty:
            fig_area = px.area(pivot, title="各一级分区每日上榜视频数量趋势",
                               labels={"value": "上榜数量", "variable": "一级分区", "rank_date": "日期"})
            fig_area.update_layout(height=400, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig_area, use_container_width=True)
        else:
            st.info("无趋势数据。")
    with tab2:
        st.dataframe(pivot.reset_index() if not pivot.empty else trend_df, use_container_width=True)
else:
    st.info("数据不足。")

# ====== 模块2：高质量内容挖掘 ======
st.markdown("---")
st.subheader(" 模块二：高质量内容挖掘 — 播放量 vs 互动率")

if len(df_latest) > 0 and "engage_rate" in df_latest.columns:
    col_left, col_right = st.columns([3, 1])
    with col_right:
        st.markdown("**解读**")
        st.markdown("- **右上**：叫好叫座\n- **左上**：宝藏视频\n- **右下**：大众娱乐\n- **左下**：普通内容")
        engage_median = df_latest["engage_rate"].median()
        st.metric("互动率中位数", f"{engage_median:.4f}")
    with col_left:
        fig_scatter = px.scatter(
            df_latest, x="view_count", y="engage_rate", color=cat_col,
            size="coin_count", hover_data=["title", "author", "rank_position", "duration_bucket", "category"],
            title="三连互动率 vs 播放量（气泡大小 = 投币数，颜色 = 一级分区）",
            labels={"view_count": "播放量", "engage_rate": "三连互动率", cat_col: "一级分区"}, log_x=True,
        )
        median_engage = df_latest["engage_rate"].median()
        median_view = df_latest["view_count"].median()
        fig_scatter.add_hline(y=median_engage, line_dash="dash", line_color="gray",
                              annotation_text=f"互动率中位数: {median_engage:.4f}")
        fig_scatter.add_vline(x=median_view, line_dash="dash", line_color="gray", annotation_text="播放量中位数")
        fig_scatter.update_layout(height=550)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("** 宝藏视频 Top 10**")
    low_view = df_latest["view_count"].median()
    treasure = df_latest[(df_latest["view_count"] < low_view) & (df_latest["engage_rate"].notna())].nlargest(10, "engage_rate")
    if not treasure.empty:
        show_cols = ["rank_position", "title", "author", cat_col, "category", "view_count", "engage_rate", "hardcore_index", "duration_bucket"]
        show_cols = [c for c in show_cols if c in treasure.columns]
        st.dataframe(treasure[show_cols], use_container_width=True,
                     column_config={"engage_rate": st.column_config.NumberColumn("互动率", format="%.4f"),
                                    "hardcore_index": st.column_config.NumberColumn("硬核度", format="%.4f"),
                                    "view_count": st.column_config.NumberColumn("播放量", format="%,d")})
    else:
        st.caption("无宝藏视频。")
else:
    st.info("数据不足。")

# ====== 模块3：流量密码解码 ======
st.markdown("---")
st.subheader(" 模块三：流量密码解码 — 标题词云 & 时长分析")

if len(df_latest) > 0:
    tab_wc, tab_dur = st.tabs([" 标题词云", " 时长分析"])
    with tab_wc:
        if not JIEBA_AVAILABLE:
            st.warning(" jieba ，pip install jieba")
        freq = generate_word_frequencies(df_latest, top_k=80)
        if freq:
            words_df = pd.DataFrame(list(freq.items())[:30], columns=["word", "count"]).sort_values("count", ascending=True)
            fig_bar_wc = px.bar(words_df, x="count", y="word", orientation="h", title="标题高频词 Top 30",
                                labels={"count": "出现次数", "word": "关键词"}, color="count", color_continuous_scale="viridis")
            fig_bar_wc.update_layout(height=550, yaxis=dict(dtick=1))
            st.plotly_chart(fig_bar_wc, use_container_width=True)
            if WORDCLOUD_AVAILABLE:
                st.markdown("**词云图**")
                fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
                render_wordcloud(freq, ax=ax_wc)
                st.pyplot(fig_wc)
        else:
            st.info("无文本数据。")
    with tab_dur:
        if "duration_bucket" in df_latest.columns:
            dur_stats = df_latest.groupby("duration_bucket").agg(
                video_count=("bvid", "count"), avg_views=("view_count", "mean"),
                avg_engage=("engage_rate", "mean"), avg_hardcore=("hardcore_index", "mean"),
            ).reset_index()
            col1, col2 = st.columns(2)
            with col1:
                fig_dur = px.bar(dur_stats, x="duration_bucket", y="avg_views", color="duration_bucket",
                                 title="各时长区间平均播放量", labels={"duration_bucket": "时长区间", "avg_views": "平均播放量"}, text_auto=",.0f")
                fig_dur.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_dur, use_container_width=True)
            with col2:
                fig_eng = px.bar(dur_stats, x="duration_bucket", y="avg_engage", color="duration_bucket",
                                 title="各时长区间平均互动率", labels={"duration_bucket": "时长区间", "avg_engage": "平均互动率"}, text_auto=".4f")
                fig_eng.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_eng, use_container_width=True)
            st.dataframe(dur_stats, use_container_width=True,
                         column_config={"avg_views": st.column_config.NumberColumn("平均播放量", format="%,.0f"),
                                        "avg_engage": st.column_config.NumberColumn("平均互动率", format="%.4f"),
                                        "avg_hardcore": st.column_config.NumberColumn("平均硬核度", format="%.4f")})
        else:
            st.info("无时长数据。")

# ====== 模块4：单视频时序追踪 ======
st.markdown("---")
st.subheader(" 模块四：单视频时序追踪 — 播放量爆发路径与生命周期")

has_timeseries = ("snapshot_seq" in df_filtered.columns and
                  df_filtered["snapshot_seq"].notna().any() and
                  df_filtered["snapshot_seq"].max() > 1)

if not has_timeseries:
    st.info(" 当前仅 1 期快照，暂无增量时序数据。连续采集 2 次以上后可见时序图表。")
else:
    video_list = df_filtered.groupby(["bvid", "title", "main_category"]).agg(
        total_views=("view_count", "max"),
        snapshot_count=("snapshot_seq", "max"),
    ).reset_index().sort_values("total_views", ascending=False)

    video_options = video_list["bvid"].tolist()
    selected_bvid = st.selectbox(
        " 选择目标视频进行深度下钻",
        options=video_options,
        format_func=lambda bv: next(
            (f"[{r['main_category']}] {r['title'][:50]} ({r['bvid']}) | {int(r['snapshot_count'])}期"
             for _, r in video_list.iterrows() if r["bvid"] == bv),
            bv,
        ),
        index=0,
    )

    video_df = df_filtered[df_filtered["bvid"] == selected_bvid].sort_values("snapshot_time")
    video_title = video_df["title"].iloc[0]
    video_author = video_df["author"].iloc[0]

    if len(video_df) < 2:
        st.warning(f" 「{video_title}」仅 1 期快照，暂无增量。")
    else:
        st.markdown(f"** {video_title}** — {video_author}")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        peak_v = video_df["peak_velocity"].iloc[0]
        peak_time = video_df.loc[video_df["delta_views"].idxmax(), "snapshot_time"]
        peak_time_str = pd.Timestamp(peak_time).strftime("%m-%d %H:%M") if pd.notna(peak_time) else "-"

        total_growth = video_df["view_count"].iloc[-1] - video_df["view_count"].iloc[0]
        first_engage = video_df["engage_rate"].iloc[0]
        last_engage = video_df["engage_rate"].iloc[-1]
        engage_change = last_engage - first_engage

        with kpi1:
            st.metric(" 峰值流速", f"{peak_v:,}", delta=f"@{peak_time_str}" if peak_v > 0 else None)
        with kpi2:
            st.metric(" 上榜总涨幅", f"{total_growth:,}")
        with kpi3:
            st.metric(" 初期互动率", f"{first_engage:.4f}")
        with kpi4:
            st.metric(" 互动衰减", f"{engage_change:+.4f}",
                      delta_color="inverse" if engage_change < 0 else "normal")

        # 折线图：累计播放量增长路径
        fig_line = px.line(
            video_df, x="snapshot_time", y="view_count",
            title=f"「{video_title[:40]}」播放量增长路径",
            labels={"snapshot_time": "快照时间", "view_count": "累计播放量"},
            markers=True,
        )
        fig_line.update_traces(line=dict(color="#fb7299", width=3), marker=dict(size=8))
        fig_line.update_layout(height=400, hovermode="x unified")
        st.plotly_chart(fig_line, use_container_width=True)

        with st.expander(" 查看逐期增量详情"):
            detail_cols = ["snapshot_time", "view_count", "delta_views", "delta_likes",
                           "delta_coins", "engage_rate", "engagement_decay", "rank_position"]
            detail_cols = [c for c in detail_cols if c in video_df.columns]
            st.dataframe(
                video_df[detail_cols].reset_index(drop=True),
                use_container_width=True,
                column_config={
                    "snapshot_time": st.column_config.DatetimeColumn("快照时间", format="MM-DD HH:mm"),
                    "view_count": st.column_config.NumberColumn("累计播放", format="%,d"),
                    "delta_views": st.column_config.NumberColumn("新增播放", format="%,d"),
                    "delta_likes": st.column_config.NumberColumn("新增点赞", format="%,d"),
                    "delta_coins": st.column_config.NumberColumn("新增投币", format="%,d"),
                    "engage_rate": st.column_config.NumberColumn("互动率", format="%.4f"),
                    "engagement_decay": st.column_config.NumberColumn("互动衰减", format="%.4f"),
                    "rank_position": st.column_config.NumberColumn("排名", format="d"),
                },
            )

# ====== 原始数据 ======
st.markdown("---")
with st.expander(" 查看原始数据"):
    st.dataframe(df_latest, use_container_width=True)

st.caption(f"数据来源: {DEFAULT_FILE} | {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")