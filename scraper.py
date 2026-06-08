"""B站全站热榜数据采集器
使用 B站公开 API 抓取全站排行榜（综合热门）数据，支持单次抓取与定时采集。
API: https://api.bilibili.com/x/web-interface/ranking/v2?rid=0&type=all
"""

import csv
import os
import random
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

# --- 配置 ---
API_URL = "https://api.bilibili.com/x/web-interface/ranking/v2"
RID = 0       # 0 = 全站
TYPE = "all"  # all = 综合热门
OUTPUT_DIR = Path(__file__).parent / "data" / "raw"
REQUEST_DELAY = 1.0

CST = timezone(timedelta(hours=8))

# 多个 UA 做轮换，降低被风控的概率
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
]

# 重试配置
MAX_RETRIES = 5
RETRY_BACKOFF = 5  # 秒，指数退避基数

# tname (二级分区) -> B站官方一级分区映射
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


def _get_headers() -> dict:
    """每次请求动态生成 headers，UA 轮换降低风控命中率"""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Referer": "https://www.bilibili.com/",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9",
    }


def fetch_ranking(rid: int = 0, ranking_type: str = "all") -> list[dict]:
    """抓取排行榜，带指数退避重试"""
    params = {"rid": rid, "type": ranking_type}
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(API_URL, params=params, headers=_get_headers(), timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") == 0:
                return data.get("data", {}).get("list", [])

            # -352: 请求太频繁 / 风控拦截
            if data.get("code") == -352:
                wait = RETRY_BACKOFF * (2 ** attempt) + random.uniform(0, 2)
                print(f"  风控拦截(-352)，{wait:.1f}s 后重试 (第{attempt+1}/{MAX_RETRIES}次)...")
                time.sleep(wait)
                last_error = RuntimeError(f"API code=-352, message={data.get('message')}")
                continue

            raise RuntimeError(f"API code={data.get('code')}, message={data.get('message')}")
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF * (2 ** attempt) + random.uniform(0, 1)
                print(f"  网络异常: {e}，{wait:.1f}s 后重试 (第{attempt+1}/{MAX_RETRIES}次)...")
                time.sleep(wait)
            else:
                raise

    raise last_error or RuntimeError("API 请求在多次重试后仍然失败")


def extract_fields(item: dict, rank_date: str, rank_position: int) -> dict:
    """从原始 API 条目中提取目标字段。snapshot_time 为精确到秒的系统时间戳。"""
    stat = item.get("stat", {})
    snapshot_time = datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S")
    tname = item.get("tname", "")
    return {
        "rank_date": rank_date,
        "snapshot_time": snapshot_time,
        "rank_position": rank_position,
        "bvid": item.get("bvid", ""),
        "aid": item.get("aid", ""),
        "title": item.get("title", "").replace("\n", " ").replace("\r", " "),
        "author": item.get("owner", {}).get("name", ""),
        "category": tname,
        "main_category": map_main_category(tname),
        "tid": item.get("tid", 0),
        "duration": item.get("duration", 0),
        "view_count": stat.get("view", 0),
        "danmaku_count": stat.get("danmaku", 0),
        "reply_count": stat.get("reply", 0),
        "like_count": stat.get("like", 0),
        "coin_count": stat.get("coin", 0),
        "favorite_count": stat.get("favorite", 0),
        "share_count": stat.get("share", 0),
        "pubdate": item.get("pubdate", 0),
        "video_url": f"https://www.bilibili.com/video/{item.get('bvid', '')}",
        "cover_url": item.get("pic", ""),
    }


def save_to_csv(rows: list[dict], filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank_date", "snapshot_time", "rank_position", "bvid", "aid",
        "title", "author", "category", "main_category", "tid", "duration",
        "view_count", "danmaku_count", "reply_count",
        "like_count", "coin_count", "favorite_count", "share_count",
        "pubdate", "video_url", "cover_url",
    ]
    write_header = not filepath.exists()
    with open(filepath, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def run_once(output_dir: Path | None = None) -> Path:
    if output_dir is None:
        output_dir = OUTPUT_DIR
    now = datetime.now(CST)
    rank_date = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename = f"bilibili_ranking_{timestamp}.csv"
    filepath = output_dir / filename

    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] B站全站热榜...")
    items = fetch_ranking(rid=RID, ranking_type=TYPE)
    print(f"  API返回: {len(items)} 条")

    rows = []
    for idx, item in enumerate(items, start=1):
        row = extract_fields(item, rank_date, idx)
        rows.append(row)

    save_to_csv(rows, filepath)
    print(f"  保存: {len(rows)} -> {filepath}")

    daily_file = output_dir / f"bilibili_ranking_daily_{rank_date}.csv"
    if daily_file != filepath:
        save_to_csv(rows, daily_file)
        print(f"  daily: {daily_file}")
    return filepath


def main():
    import argparse
    parser = argparse.ArgumentParser(description="B站全站热榜采集器")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--loop", action="store_true", help="循环模式（每4h触发）")
    parser.add_argument("--interval", type=int, default=14400, help="循环间隔秒（默认4h）")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    run_once(output_dir)

    if args.loop:
        print(f"循环模式，间隔 {args.interval}s...")
        while True:
            time.sleep(args.interval)
            run_once(output_dir)


if __name__ == "__main__":
    main()