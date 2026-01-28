import yaml
import feedparser
from google import genai
import PyRSS2Gen
import datetime
import requests
import json
import os
import re
import logging
import xml.etree.ElementTree as ET
from newspaper import Article
from dateutil import parser as date_parser

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = "sources.yaml"
STATE_PATH = "read_log.txt"

def load_state():
    if not os.path.exists(STATE_PATH):
        return datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
    try:
        with open(STATE_PATH, "r") as f:
            content = f.read().strip()
            return date_parser.parse(content) if content else datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
    except Exception:
        return datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)

def save_state():
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(STATE_PATH, "w") as f:
        f.write(now)

def load_existing_links(xml_file):
    seen = set()
    if not os.path.exists(xml_file):
        return seen
    try:
        tree = ET.parse(xml_file)
        channel = tree.getroot().find("channel")
        for item in channel.findall("item"):
            link = item.find("link")
            if link is not None and link.text:
                seen.add(link.text.strip())
    except Exception:
        pass
    return seen

def is_blocked(content, patterns):
    return any(p.search(content) for p in patterns) if content else False

def fetch_url_content(url):
    try:
        article = Article(url, request_timeout=15)
        article.download()
        article.parse()
        return article.title, article.text
    except Exception:
        return None, None

def main():
    if not os.path.exists(CONFIG_PATH):
        return

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # NEW SDK CLIENT
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    blocked = [re.compile(p, re.IGNORECASE) for p in config.get("blocked_content", [])]
    last_run = load_state()

    for cat_name, sources in config.get("categories", {}).items():
        output_file = f"{cat_name}.xml"
        seen_links = load_existing_links(output_file)
        candidates = []

        # RSS Sources
        for rss_url in sources.get("rss", []):
            feed = feedparser.parse(rss_url)
            for entry in feed.entries:
                link = entry.get("link", "")
                if not link or link in seen_links: continue
                
                pub_struct = entry.get("published_parsed") or entry.get("updated_parsed")
                if pub_struct:
                    if datetime.datetime(*pub_struct[:6], tzinfo=datetime.timezone.utc) <= last_run:
                        continue

                if not is_blocked(entry.get("summary", ""), blocked):
                    candidates.append({"title": entry.get("title", "Untitled"), "link": link, "content": entry.get("summary", "")[:3000]})

        # URL Sources
        for url in sources.get("urls", []):
            if url not in seen_links:
                title, text = fetch_url_content(url)
                if title and text and not is_blocked(text, blocked):
                    candidates.append({"title": title, "link": url, "content": text[:3000]})

        if not candidates: continue

        # AI Summarization using native JSON mode
        prompt = f"Summarize these articles as a JSON list with 'title', 'link', and 'summary' keys:\n{json.dumps(candidates)}"
        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt,
                config={'response_mime_type': 'application/json'}
            )
            ai_results = json.loads(response.text)
        except Exception as e:
            logger.error(f"AI error: {e}")
            continue

        rss_items = [PyRSS2Gen.RSSItem(
            title=r["title"], link=r["link"], description=r["summary"], 
            guid=PyRSS2Gen.Guid(r["link"]), pubDate=datetime.datetime.now()
        ) for r in ai_results]

        rss = PyRSS2Gen.RSS2(title=f"AI News - {cat_name}", link="https://github.com/USER/REPO", 
                             description="AI Generated Feed", items=rss_items)
        
        with open(output_file, "w", encoding="utf-8") as f:
            rss.write_xml(f)

    save_state()

if __name__ == "__main__":
    main()
