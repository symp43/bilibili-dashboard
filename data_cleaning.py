"""
B站热榜数据清洗与指标计算 (ETL v2)
读取 raw/*.csv -> 清洗 -> 衍生指标计算 -> 输出 cleaned_bilibili_ranking.csv
v2: 新增时间序列面板数据处理，支持 snapshot_time、差分增量指标。
"""

import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

CST = timezone(timedelta(hours=8))
RAW_DIR = Path(__file__).parent / "data" / "raw"
CLEAN_DIR = Path(__file__).parent / "data" / "cleaned"
OUTPUT_FILE = CLEAN_DIR / "cleaned_bilibili_ranking.csv"

# tname -> B站官方一级分区映射
TNAME_TO_MAIN = {
    "动漫杂谈": "动画", "同人·手书": "动画", "MMD·3D": "动画",
    "特摄": "动画", "短片": "动画",
    "音乐现场": "音乐", "音乐综合": "音乐", "乐评盘点": "音乐", "翻唱": "音乐",
    "宅舞": "舞蹈",
    "单机游戏": "游戏", "网络游戏": "游戏", "手机游戏": "游戏", "电子竞技": "游戏",
    "社科·法律·心理": "知识", "科学科普": "知识", "人文历史": "知识",
    "财经商业": "知识", "校园学习": "知识",
    "数码": "科技",
    "竞技体育": "运动", "篮球": "运动", "运动综合": "运动", "健身": "运动",
    "汽车生活": "汽车",
    "日常": "生活", "出行": "生活", "亲子": "生活", "三农": "生活", "手工": "生活",
    "美食侦探": "美食", "美食制作": "美食", "美食测评": "美食", "美食记录": "美食",
    "鬼畜调教": "鬼畜",
    "仿妆cos": "时尚",
    "娱乐粉丝创作": "娱乐", "小剧场": "娱乐", "搞笑": "娱乐",
    "影视剪辑": "影视", "影视杂谈": "影视", "预告·资讯": "影视",
    "绘画": "绘画",
    "综合": "综合",
}


def map_main_category(tname: str) -> str:
    return TNAME_TO_MAIN.get(tname, tname)


def parse_number_string(val) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    if not isinstance(val, str) or not val.strip():
        return np.nan
    s = val.strip().replace(",", "").replace("，", "")
    multiplier = 1.0
    if "亿" in s:
        multiplier = 1e8
        s = s.replace("亿", "")
    elif "万" in s:
        multiplier = 1e4
        s = s.replace("万", "")
    try:
        return float(s) * multiplier
    except ValueError:
        return np.nan


def parse_duration_to_seconds(val) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    if not isinstance(val, str) or not val.strip():
        return np.nan
    parts = [int(x) for x in val.strip().split(":") if x.isdigit()]
    if not parts:
        return np.nan
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        return parts[0] * 60 + parts[1]
    else:
        return float(parts[0])


def load_raw_csvs(raw_dir: Path) -> pd.DataFrame:
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"raw 目录下无 CSV 文件: {raw_dir}")
    dfs = []
    for fp in csv_files:
        try:
            df = pd.read_csv(fp, encoding="utf-8-sig")
            dfs.append(df)
        except Exception as e:
            print(f"  跳过 {fp.name}: {e}")
    if not dfs:
        raise ValueError("未能加载任何有效 CSV")
    df = pd.concat(dfs, ignore_index=True)

    # v2: 兼容旧 CSV 无 snapshot_time 列
    if "snapshot_time" not in df.columns:
        print("  旧 CSV 兼容: 从 rank_date 生成 snapshot_time")
        df["snapshot_time"] = df["rank_date"].astype(str) + " 00:00:00"

    # v2: 按 [bvid, snapshot_time] 去重（保留每次快照）
    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"])
    df = df.drop_duplicates(subset=["bvid", "snapshot_time"], keep="last")
    return df


def clean_and_enrich(df: pd.DataFrame) -> pd.DataFrame:
    print(f"  原始行数: {len(df)}")

    # --- 1. 列名标准化 ---
    df.columns = [c.strip().lower() for c in df.columns]

    # --- 2. 缺失值处理 ---
    required = ["bvid", "title", "view_count", "rank_position"]
    df = df.dropna(subset=required).copy()

    df["category"] = df["category"].fillna("未知")
    df["author"] = df["author"].fillna("未知")
    df["title"] = df["title"].fillna("")

    if "main_category" not in df.columns:
        df["main_category"] = df["category"].apply(map_main_category)
    else:
        df["main_category"] = df["main_category"].fillna(df["category"].apply(map_main_category))

    for col in ["danmaku_count", "reply_count", "like_count", "coin_count", "favorite_count", "share_count", "duration"]:
        if col not in df.columns:
            df[col] = 0

    # --- 3. 类型统一 ---
    numeric_cols = ["view_count", "danmaku_count", "reply_count", "like_count",
                    "coin_count", "favorite_count", "share_count"]
    for col in numeric_cols:
        if df[col].dtype == object:
            df[col] = df[col].apply(parse_number_string)
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")

    if df["duration"].dtype == object:
        df["duration"] = df["duration"].apply(parse_duration_to_seconds)
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0).astype("int64")

    df["rank_position"] = pd.to_numeric(df["rank_position"], errors="coerce").fillna(0).astype("int64")

    # --- 4. snapshot_time 类型 ---
    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"])

    # --- 5. 时序排序（v2: 全局按 bvid + snapshot_time 升序）---
    df = df.sort_values(["bvid", "snapshot_time"]).reset_index(drop=True)

    # --- 6. 异常值标记 ---
    df["anomaly_flag"] = "normal"
    df.loc[df["view_count"] <= 0, "anomaly_flag"] = "zero_views"
    q01 = df["view_count"].quantile(0.01)
    df.loc[(df["view_count"] < q01) & (df["rank_position"] <= 30), "anomaly_flag"] = "low_views_high_rank"
    anomaly_count = (df["anomaly_flag"] != "normal").sum()
    if anomaly_count > 0:
        print(f"  标记异常数据: {anomaly_count} 条")

    # --- 7. 衍生业务指标 ---
    # 三连互动率
    df["engage_rate"] = np.where(
        df["view_count"] > 0,
        (df["like_count"] + df["coin_count"] + df["favorite_count"]) / df["view_count"],
        np.nan,
    )
    # 硬核度
    df["hardcore_index"] = np.where(
        df["like_count"] > 0,
        df["coin_count"] / df["like_count"],
        np.nan,
    )
    # 讨论热度
    df["discussion_heat"] = np.where(
        df["view_count"] > 0,
        (df["danmaku_count"] + df["reply_count"]) / df["view_count"],
        np.nan,
    )

    # --- 8. v2: 时序差分指标（按 bvid 分组计算）---
    # 标记每个 bvid 组内的序号
    df["snapshot_seq"] = df.groupby("bvid").cumcount() + 1

    # 单期增量：负增长截断为 0
    for col, delta_col in [("view_count", "delta_views"),
                            ("like_count", "delta_likes"),
                            ("coin_count", "delta_coins")]:
        df[delta_col] = df.groupby("bvid")[col].diff().fillna(0).clip(lower=0).astype("int64")

    # 峰值流速：每个 bvid 的最大单期新增播放量
    df["peak_velocity"] = df.groupby("bvid")["delta_views"].transform("max")

    # 互动衰减偏离度：当前期 engage_rate - 首期 engage_rate
    first_engage = df.groupby("bvid")["engage_rate"].transform("first")
    df["engagement_decay"] = df["engage_rate"] - first_engage

    # --- 9. 时长区间 ---
    def duration_label(sec):
        if sec <= 0:
            return "未知"
        elif sec <= 180:
            return "0-3分钟"
        elif sec <= 600:
            return "3-10分钟"
        else:
            return "10分钟以上"

    df["duration_bucket"] = df["duration"].apply(duration_label)

    # 峰值流速发生时间
    peak_idx = df.groupby("bvid")["delta_views"].idxmax()
    peak_times = df.loc[peak_idx, ["bvid", "snapshot_time"]].set_index("bvid")
    peak_times.columns = ["peak_velocity_time"]
    df = df.join(peak_times, on="bvid")

    print(f"  清洗后行数: {len(df)}")
    print(f"  独立视频数: {df['bvid'].nunique()}")
    print(f"  快照时间跨度: {df['snapshot_time'].min()} ~ {df['snapshot_time'].max()}")
    print(f"  单期 max delta_views: {df['delta_views'].max():,}")
    if "main_category" in df.columns:
        print(f"  一级分区数: {df['main_category'].nunique()}")
        print(f"  二级分区数: {df['category'].nunique()}")

    return df


def main():
    print(f"[{datetime.now(CST).strftime('%Y-%m-%d %H:%M:%S')}] ETL v2 start")

    df_raw = load_raw_csvs(RAW_DIR)
    print(f"  合并后总行数: {len(df_raw)}")

    df_clean = clean_and_enrich(df_raw)

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"  -> {OUTPUT_FILE}")

    # 统计摘要
    print("\n=== 数据摘要 ===")
    print(f"快照数: {len(df_clean)}")
    print(f"视频数: {df_clean['bvid'].nunique()}")
    print(f"平均播放量: {df_clean['view_count'].mean():,.0f}")
    print(f"平均三连互动率: {df_clean['engage_rate'].mean():.4f}")
    print(f"异常数据: {(df_clean['anomaly_flag'] != 'normal').sum()}")

    if "main_category" in df_clean.columns:
        print("\n=== 一级分区分布 ===")
        for k, v in df_clean["main_category"].value_counts().items():
            print(f"  {k}: {v}")

    return df_clean


if __name__ == "__main__":
    main()
