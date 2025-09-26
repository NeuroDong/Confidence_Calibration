import requests
import re
import os
from typing import List, Optional


import re
from difflib import SequenceMatcher

def is_same_title(title_a: str, title_b: str, threshold: float = 0.9) -> bool:
    """
    判断两个标题是否对应同一篇文章。
    
    参数:
        title_a, title_b: 待比较的两个标题
        threshold: 相似度阈值 (0~1)，默认 0.9
    
    返回:
        bool: True 表示认为是同一篇文章
    """
    def normalize(text: str) -> str:
        # 转小写，去掉多余空格和标点
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)  # 只保留字母数字空格
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    norm_a = normalize(title_a)
    norm_b = normalize(title_b)
    
    # 计算相似度
    ratio = SequenceMatcher(None, norm_a, norm_b).ratio()
    return ratio >= threshold


def get_citations_by_title(title: str) -> Optional[int]:
    """Query OpenAlex for the most relevant work matching title and return cited_by_count.

    Returns an int citation count, or None on not found, or raises on HTTP/network errors.
    """
    url = "https://api.openalex.org/works"
    params = {
        "search": title,
        "per_page": 1,  # 只取最相关的一条
        "mailto": "e1710947@u.nus.edu"  # 建议填自己的邮箱
    }
    headers = {"User-Agent": "CitationFetcher/1.0"}

    resp = requests.get(url, params=params, headers=headers, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} from OpenAlex")

    data = resp.json()
    if not data.get("results"):
        return None

    paper = data["results"][0]
    if is_same_title(title, paper.get("title", "")):
        return paper.get("cited_by_count")
    else:
        print(f"Title mismatch: '{title}' vs '{paper.get('title', '')}'")
        return None

def s2_by_title(title: str):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": title, "limit": 1, "fields": "title,year,citationCount,url"}
    r = requests.get(url, params=params, headers={"User-Agent": "CitationChecker/0.1"}, timeout=15)
    if r.status_code != 200:
        return {"source": "Semantic Scholar", "error": f"HTTP {r.status_code}"}
    js = r.json()
    items = js.get("data") or []
    if not items:
        return {"source": "Semantic Scholar", "error": "no result"}
    it = items[0]
    return it.get("citationCount")



def _extract_titles_from_readme(readme_text: str) -> List[str]:
    """Return a list of paper titles (link text) from the README markdown tables.

    We look for markdown links in table rows: [Title](url)
    """
    titles = []
    for line in readme_text.splitlines():
        line = line.strip()
        # only consider table rows (start with '|')
        if not line.startswith("|"):
            continue
        m = re.search(r"\[([^\]]+)\]\([^\)]+\)", line)
        if m:
            titles.append(m.group(1))
    return titles


def _make_shield_url(label: str, message: str, color: str = "blue") -> str:
    """Construct a shields.io badge URL for a label/message pair."""
    from urllib.parse import quote_plus

    label_enc = quote_plus(label)
    msg_enc = quote_plus(str(message))
    return f"https://img.shields.io/badge/{label_enc}-{msg_enc}-{color}?style=for-the-badge"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def generate_badges_for_readme(readme_path: str, badges_dir: str = "badges") -> None:
    """Read README, fetch citation counts for each paper link, download badge images,
    save them to badges_dir, and update README to include a Citations column with
    the badge image markdown.
    """
    ensure_dir(badges_dir)

    with open(readme_path, "r", encoding="utf-8") as f:
        text = f.read()

    titles = _extract_titles_from_readme(text)
    if not titles:
        print("No titles/links found in README")
        return

    # Map title -> badge filename or remote badge URL
    badge_map = {}
    for title in titles:
        print(f"Fetching citations for: {title}")
        a = get_citations_by_title(title) 
        b = s2_by_title(title)
        h = 1
        while (not isinstance(b,int)) and h < 50:
            b = s2_by_title(title)
            h += 1

        if a is None and not isinstance(b,int):
            count = None
            print(f"OpenAlex and Semantic Scholar both failed for {title}!")
        elif a is None and isinstance(b,int):
            count = b
            print(f"OpenAlex failed for {title}!")
        elif a is not None and not isinstance(b,int):
            count = a
            print(f"Semantic Scholar failed for {title}!")
        elif a is not None and isinstance(b,int):
            count = a+b

        msg = count if count is not None else "n/a"

        badge_url = _make_shield_url("Citations", msg)

        # Download the badge image (SVG)
        try:
            resp = requests.get(badge_url, headers={"User-Agent": "CitationFetcher/1.0"}, timeout=15)
            if resp.status_code == 200:
                # sanitize filename
                safe_name = re.sub(r"[^0-9A-Za-z._-]", "_", title)[:120]
                filename = f"{safe_name}.svg"
                filepath = os.path.join(badges_dir, filename)
                with open(filepath, "wb") as bf:
                    bf.write(resp.content)
                badge_map[title] = (filepath, badge_url)
            else:
                print(f"Failed to download badge for {title}: HTTP {resp.status_code}")
                badge_map[title] = (None, badge_url)
        except Exception as e:
            print(f"Error downloading badge for {title}: {e}")
            badge_map[title] = (None, badge_url)


if __name__ == "__main__":
    # Assume README is in the same directory as this script
    base_dir = os.path.dirname(__file__)
    readme = os.path.join(base_dir, "README.md")
    badges_folder = os.path.join(base_dir, "badges")
    generate_badges_for_readme(readme, badges_folder)