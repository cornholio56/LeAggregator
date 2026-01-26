pyyaml
feedparser
google-generativeai
PyRSS2Gen
requests
newspaper3k
python-dateutil
lxml

================================================================================
6. CORE SCRIPT (aggregator.py)
================================================================================
import yaml
import feedparser
import google.generativeai as genai
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
from email.utils import formatdate

# SETUP LOGGING
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# CONSTANTS
CONFIG_PATH = "sources.yaml"
STATE_PATH = "read_log.txt"

# --- HELPER FUNCTIONS ---

def load_state():
    """Reads the last successful run timestamp."""
    if not os.path.exists(STATE_PATH):
        return datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
    try:
        with open(STATE_PATH, "r") as f:
            content = f.read().strip()
            if not content:
                return datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
            return date_parser.parse(content)
    except Exception:
        logger.warning("Could not parse state file, defaulting to all-time.")
        return datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)

def save_state():
    """Writes the current timestamp to the state file."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(STATE_PATH, "w") as f:
        f.write(now)

def load_existing_links(xml_file):
    """Parses output XML to find links we have already processed (History)."""
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
    except Exception as e:
        logger.error(f"Error parsing existing XML {xml_file}: {e}")
    return seen

def is_blocked(content, patterns):
    if not content:
        return False
    return any(pat.search(content) for pat in patterns)

def fetch_url_content(url):
    """Scrapes content from a direct URL using newspaper3k."""
    try:
        article = Article(url, request_timeout=15)
        article.download()
        article.parse()
        return article.title, article.text
    except Exception as e:
        logger.warning(f"Failed to scrape {url}: {e}")
        return None, None

# --- MAIN PIPELINE ---

def main():
    # 1. LOAD CONFIG
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # 2. SETUP GEMINI
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    
    # 3. SETUP FILTERS
    blocked = [re.compile(p, re.IGNORECASE) for p in config.get("blocked_content", [])]
    last_run = load_state()
    logger.info(f"Last run: {last_run}")

    # 4. PROCESS CATEGORIES
    for cat_name, sources in config.get("categories", {}).items():
        logger.info(f"--- Processing {cat_name} ---")
        output_file = f"{cat_name}.xml"
        
        seen_links = load_existing_links(output_file)
        candidates = []

        # A. PROCESS RSS FEEDS
        for rss_url in sources.get("rss", []):
            try:
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
                    
                    content = ""
                    if "content" in entry:
                        content = entry.content[0].value
                    elif "summary" in entry:
                        content = entry.summary
                    
                    if not is_blocked(content, blocked):
                        candidates.append({
                            "title": entry.get("title", "Untitled"),
                            "link": link,
                            "content": content[:3000]
                        })
            except Exception as e:
                logger.error(f"RSS Error {rss_url}: {e}")

        # B. PROCESS DIRECT URLS
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
            logger.info("No new items found.")
            continue

        logger.info(f"Found {len(candidates)} new items. Sending to Gemini...")

        # 5. AI SUMMARIZATION
        prompt = f"""
Act as a news editor. Process these articles.
1. Deduplicate stories covering the same event.
2. Create a JSON summary for each unique story.

Format: JSON List of objects:
{{ "title": "Headline", "summary": "3 sentences.", "link": "Primary URL" }}

Input Data:
{json.dumps(candidates, default=str)}
"""

        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            ai_results = json.loads(response.text)
        except Exception as e:
            logger.error(f"AI Error: {e}")
            continue

        # 6. UPDATE RSS FEED
        existing_rss_data = []
        if os.path.exists(output_file):
            try:
                tree = ET.parse(output_file)
                root = tree.getroot()
                channel = root.find("channel")
                if channel:
                    for item in channel.findall("item"):
                        existing_rss_data.append({
                            "title": item.findtext("title"),
                            "link": item.findtext("link"),
                            "description": item.findtext("description"),
                            "pubDate": item.findtext("pubDate"),
                            "guid": item.findtext("guid")
                        })
            except Exception:
                pass

        current_time = datetime.datetime.now(datetime.timezone.utc)
        new_rss_data = []
        for res in ai_results:
            new_rss_data.append({
                "title": res.get("title"),
                "link": res.get("link"),
                "description": res.get("summary"),
                "pubDate": current_time,
                "guid": res.get("link")
            })

        combined_data = (new_rss_data + existing_rss_data)[:50]

        rss_items = []
        for data in combined_data:
            d_obj = data["pubDate"]
            if isinstance(d_obj, str):
                try:
                    d_obj = date_parser.parse(d_obj)
                except Exception:
                    d_obj = current_time

            rss_items.append(
                PyRSS2Gen.RSSItem(
                    title=data["title"],
                    link=data["link"],
                    description=data["description"],
                    guid=PyRSS2Gen.Guid(data["guid"]),
                    pubDate=d_obj
                )
            )

        rss = PyRSS2Gen.RSS2(
            title=f"AI News - {cat_name}",
            link="https://github.com/USER/REPO",
            description="AI Generated Feed",
            lastBuildDate=current_time,
            items=rss_items
        )
        
        with open(output_file, "w", encoding="utf-8") as f:
            rss.write_xml(f)

        logger.info(f"Wrote {len(rss_items)} items to {output_file}")

    # 7. UPDATE STATE
    save_state()
    logger.info("Run Complete.")

if __name__ == "__main__":
    main()
