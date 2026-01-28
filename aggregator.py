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

def load_state(): [cite: 12]
    if not os.path.exists(STATE_PATH):
        return datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
    try:
        with open(STATE_PATH, "r") as f:
            content = f.read().strip()
            if not content:
                return datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
            return date_parser.parse(content)
    except Exception:
        return datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)

def save_state():
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(STATE_PATH, "w") as f:
        f.write(now)

def load_existing_links(xml_file): [cite: 13, 14]
    seen = set()
    if not os.path.exists(xml_file):
        return seen
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        channel = root.find("channel")
        if channel is not None:
            for item in channel.findall("item"):
                link = item.find("link")
                if link is not None and link.text:
                    seen.add(link.text.strip())
    except Exception:
        pass
    return seen

def is_blocked(content, patterns):
    if not content:
        return False
    return any(p.search(content) for p in patterns)

def fetch_url_content(url): [cite: 15]
    try:
        article = Article(url, request_timeout=15)
        article.download()
        article.parse()
        return article.title, article.text
    except Exception:
        return None, None

def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Initialize the NEW Gemini Client
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    blocked = [re.compile(p, re.IGNORECASE) for p in config.get("blocked_content", [])]
    last_run = load_state()

    for cat_name, sources in config.get("categories", {}).items():
        output_file = f"{cat_name}.xml"
        seen_links = load_existing_links(output_file)
        candidates = []

        # Process RSS Feeds [cite: 16, 17, 18, 19]
        for rss_url in sources.get("rss", []):
            feed = feedparser.parse(rss_url)
            for entry in feed.entries:
                link = entry.get("link", "")
                if not link or link in seen_links:
                    continue

                pub_struct = entry.get("published_parsed") or entry.get("updated_parsed")
                if pub_struct:
                    pub_dt = datetime.datetime(*pub_struct[:6], tzinfo=datetime.timezone.utc)
                    if pub_dt <= last_run:
                        continue

                content = entry.get("summary", "")
                if not is_blocked(content, blocked):
                    candidates.append({
                        "title": entry.get("title", "Untitled"),
                        "link": link,
                        "content": content[:3000]
                    })

        # Process Direct URLs [cite: 20, 21]
        for url in sources.get("urls", []):
            if url in seen_links:
                continue
            title, text = fetch_url_content(url)
            if title and text and not is_blocked(text, blocked):
                candidates.append({
                    "title": title,
                    "link": url,
                    "content": text[:3000]
                })

        if not candidates:
            continue

        # AI Summarization with native JSON mode 
        prompt = (
            f"Summarize these articles into a JSON list. "
            f"Each item MUST have 'title', 'link', and 'summary' keys. "
            f"Articles:\n{json.dumps(candidates)}"
        )
        
        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt,
                config={
                    'response_mime_type': 'application/json',
                }
            )
            ai_results = json.loads(response.text)
        except Exception as e: [cite: 23]
            logger.error(f"Failed to generate or parse AI response for {cat_name}: {e}")
            continue

        # Build the RSS items [cite: 24, 25]
        rss_items = []
        now = datetime.datetime.now(datetime.timezone.utc)

        for r in ai_results:
            rss_items.append(
                PyRSS2Gen.RSSItem(
                    title=r["title"],
                    link=r["link"],
                    description=r["summary"],
                    guid=PyRSS2Gen.Guid(r["link"]),
                    pubDate=now
                )
            )

        # Write to file [cite: 26]
        rss = PyRSS2Gen.RSS2(
            title=f"AI News - {cat_name}",
            link="https://github.com/USER/REPO",
            description="AI Generated Feed",
            lastBuildDate=now,
            items=rss_items
        )

        with open(output_file, "w", encoding="utf-8") as f:
            rss.write_xml(f)

    save_state()

if __name__ == "__main__":
    main()
