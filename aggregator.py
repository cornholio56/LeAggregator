import yaml
import feedparser
import asyncio
import os
import json
import logging
import datetime
import hashlib
import html  # ← Added for escaping
import PyRSS2Gen
import trafilatura
import xml.etree.ElementTree as ET
from google.generativeai import GenerativeModel, GenerationConfig
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from newspaper import Article
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from dateutil import parser as date_parser
from xml.dom.minidom import parseString

CONFIG_PATH = "sources.yaml"
STATE_PATH = "read_log.txt"
BASE_REPO_URL = "https://github.com/USER/REPO"  # ← replace with actual repo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

OMIT_CATEGORIES = '\n'.join([f"  - {cat}" for cat in config['content_filters']['omit_categories']])

GENERAL_INSTRUCTION_TEMPLATE = """
You are an Expert news editor specializing in thematic synthesis.

OBJECTIVE: Group the provided articles into 0–3 cohesive thematic Stories. Each Story synthesizes related articles, explaining relevance and insights.

RULES:
- Strictly enforce these negative filters and omit any matching content:
{omit_categories}
- Output only high-confidence themes (prefer omission over weak or uncertain ones).
- Be factual, analytical, and neutral; no speculation or opinion.
- Titles: 5–10 words, concise and thematic.
- Use this exact HTML format for each Story's content (repeat blocks for sub-themes):

<div>
  <p>Intro paragraph (2–3 sentences explaining relevance and context)</p>
  <h3>Sub-theme title</h3>
  <p>Analysis paragraph</p>
  <ul>
    <li><a href="URL">Source title or short description</a></li>
  </ul>
</div>

ARTICLES (each separated by ---):
{articles}

Output ONLY valid JSON:
{{
  "themes": [
    {{"title": "Story title", "expanded_html": "<div>...</div>"}},
    ...
  ]
}}
"""

SOCIAL_PULSE_INSTRUCTION_TEMPLATE = """
You are a Social Signal Analyst focused on emerging patterns and sentiment.

TASK: Produce exactly one observational Story titled "Social Pulse: [Short descriptive suffix]". Base it solely on the provided signals.

RULES:
- Analysis-only: 2–3 concise paragraphs summarizing key observations and trends.
- No source links, speculation, predictions, or external references.
- Remain factual, neutral, and descriptive; focus on observable patterns.

SOCIAL SIGNALS / INPUT:
{social_signals}

Output ONLY valid JSON:
{{
  "themes": [
    {{"title": "Social Pulse: ...", "expanded_html": "<div><p>...</p><p>...</p></div>"}}
  ]
}}
"""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "themes": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "title": {"type": "STRING"},
                    "expanded_html": {"type": "STRING"}
                },
                "required": ["title", "expanded_html"]
            },
            "minItems": 0
        }
    },
    "required": ["themes"]
}

def get_seen_urls():
    seen = set()
    for cat in config['categories']:
        path = f"{cat}.xml"
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                xml_content = f.read()
            dom = parseString(xml_content)
            for desc in dom.getElementsByTagName('description'):
                if desc.firstChild and desc.firstChild.nodeValue:
                    try:
                        inner = parseString(f"<root>{desc.firstChild.nodeValue}</root>")
                        for a in inner.getElementsByTagName('a'):
                            href = a.getAttribute('href')
                            if href:
                                seen.add(href.strip())
                    except:
                        pass
        except Exception as e:
            logger.warning(f"Error extracting seen URLs from {path}: {e}")
    return seen

async def fetch_url_content(url: str) -> str | None:
    for attempt in range(3):
        try:
            downloaded = trafilatura.fetch_url(url)
            text = trafilatura.extract(downloaded, include_links=False)
            if text and len(text) > 600:
                return text
        except:
            pass
        try:
            article = Article(url, fetch_images=False)
            article.download()
            article.parse()
            if len(article.text) > 600:
                return article.text
        except:
            pass
        try:
            async with AsyncWebCrawler(config=BrowserConfig(headless=True, verbose=False)) as crawler:
                result = await crawler.arun(
                    url=url,
                    config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS, word_count_threshold=150)
                )
                if result.success and result.markdown and len(result.markdown) > 600:
                    return result.markdown
        except:
            pass
    logger.warning(f"Failed to fetch content from {url} after 3 attempts")
    return None

def get_ai_response(client: GenerativeModel, prompt: str) -> dict:
    try:
        response = client.generate_content(
            contents=prompt,
            generation_config=GenerationConfig(
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
                temperature=0.1,
                top_p=0.95,
                max_output_tokens=2048
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        return json.loads(response.text)
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return {"themes": []}

def load_existing_items(category: str, cutoff: datetime.datetime):
    filename = f"{category}.xml"
    if not os.path.exists(filename):
        return []
    try:
        parsed = feedparser.parse(filename)
        items = []
        for entry in parsed.entries:
            try:
                dt = date_parser.parse(entry.published)
                if dt > cutoff:
                    items.append(
                        PyRSS2Gen.RSSItem(
                            title=entry.title,
                            link=entry.link,
                            description=entry.description,
                            guid=PyRSS2Gen.Guid(entry.id if 'id' in entry else entry.link, isPermaLink=False),
                            pubDate=dt
                        )
                    )
            except:
                continue
        return items
    except Exception as e:
        logger.warning(f"Failed to parse existing {filename}: {e}")
        return []

def update_xml_feed(category: str, new_themes: list, now: datetime.datetime):
    filename = f"{category}.xml"
    cutoff = now - datetime.timedelta(days=14)
    items = load_existing_items(category, cutoff)

    for theme in new_themes:
        # Escape potentially dangerous characters in user/AI-generated HTML content
        safe_title = html.escape(theme["title"])
        safe_html = theme["expanded_html"]  # Assume AI outputs safe HTML; if not, escape inner text too
        content_hash = hashlib.sha256(safe_html.encode('utf-8')).hexdigest()
        guid_str = f"{category}-{content_hash}"

        items.append(
            PyRSS2Gen.RSSItem(
                title=safe_title,
                link=BASE_REPO_URL,
                description=safe_html,
                guid=PyRSS2Gen.Guid(guid_str, isPermaLink=False),
                pubDate=now
            )
        )

    # Fallback if no items at all (prevents "empty feed" rejection)
    if not items:
        logger.info(f"No stories for {category}; adding placeholder item")
        items.append(
            PyRSS2Gen.RSSItem(
                title="No recent thematic stories",
                link=BASE_REPO_URL,
                description="<div><p>No new high-confidence AI-curated themes in the last run. Check back after the next update (every 4 hours).</p></div>",
                guid=PyRSS2Gen.Guid(f"{category}-placeholder-{now.isoformat()}", isPermaLink=False),
                pubDate=now
            )
        )

    items.sort(key=lambda i: i.pubDate, reverse=True)

    rss = PyRSS2Gen.RSS2(
        title=f"AI News – {category.capitalize()}",
        link=BASE_REPO_URL,
        description="AI-curated thematic RSS feed",
        lastBuildDate=now,
        items=items
    )

    # Write to StringIO first to control XML declaration
    from io import StringIO
    f = StringIO()
    rss.write_xml(f, encoding='utf-8')  # Try utf-8 here
    xml_str = f.getvalue()

    # Force correct UTF-8 declaration (PyRSS2Gen often defaults to iso-8859-1)
    if '<?xml' in xml_str:
        xml_str = xml_str.replace('encoding="iso-8859-1"', 'encoding="utf-8"', 1)
    else:
        xml_str = '<?xml version="1.0" encoding="utf-8"?>\n' + xml_str

    with open(filename, "w", encoding="utf-8") as out:
        out.write(xml_str)

    logger.info(f"Updated {filename} with {len(items)} items (including fallback if needed)")

async def process_category(category: str, last_run: datetime.datetime, seen_urls: set, client: GenerativeModel):
    sources = config['categories'][category]
    articles = []
    for rss_url in sources.get('rss', []):
        feed = feedparser.parse(rss_url)
        for entry in feed.entries:
            if 'published' not in entry:
                continue
            pub = date_parser.parse(entry.published)
            if pub <= last_run:
                continue
            url = entry.link
            if url in seen_urls:
                continue
            content = await fetch_url_content(url)
            if content:
                seen_urls.add(url)
                short_title = (entry.title or "Untitled")[:80]
                articles.append(f"Title: {short_title}\nURL: {url}\nPublished: {pub.isoformat()}\nContent: {content[:2200]}...\n---")
    if not articles:
        logger.info(f"No new articles for {category}")
        return []
    prompt = GENERAL_INSTRUCTION_TEMPLATE.format(
        omit_categories=OMIT_CATEGORIES,
        articles='\n'.join(articles)
    )
    result = get_ai_response(client, prompt)
    themes = result.get('themes', [])
    logger.info(f"Generated {len(themes)} themes for {category}")
    return themes

async def process_social_pulse(last_run: datetime.datetime, seen_urls: set, client: GenerativeModel):
    sources = config['categories']['general']
    signals = []
    for rss_url in sources.get('rss', []):
        feed = feedparser.parse(rss_url)
        for entry in feed.entries:
            if 'published' not in entry:
                continue
            pub = date_parser.parse(entry.published)
            if pub <= last_run:
                continue
            url = entry.link
            if url in seen_urls:
                continue
            content = await fetch_url_content(url)
            if content:
                signals.append(f"Discussion signal from {url}:\nTitle: {entry.title}\nExcerpt: {content[:800]}...\n---")
    if len(signals) < 2:
        logger.info("Insufficient signals for Social Pulse")
        return []
    prompt = SOCIAL_PULSE_INSTRUCTION_TEMPLATE.format(social_signals='\n'.join(signals))
    result = get_ai_response(client, prompt)
    themes = result.get('themes', [])
    logger.info(f"Generated {len(themes)} Social Pulse themes")
    return themes

async def main():
    now = datetime.datetime.now(datetime.timezone.utc)
    last_run_str = "1970-01-01T00:00:00Z"
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, 'r', encoding='utf-8') as f:
            last_run_str = f.read().strip()
    last_run = date_parser.parse(last_run_str)

    seen_urls = get_seen_urls()

    client = GenerativeModel('gemini-1.5-pro-latest')  # adjust model as needed

    for category in config['categories']:
        themes = await process_category(category, last_run, seen_urls, client)
        update_xml_feed(category, themes, now)

    pulse_themes = await process_social_pulse(last_run, seen_urls, client)
    if pulse_themes:
        update_xml_feed('general', pulse_themes, now)

    with open(STATE_PATH, 'w', encoding='utf-8') as f:
        f.write(now.isoformat())

if __name__ == "__main__":
    asyncio.run(main())
