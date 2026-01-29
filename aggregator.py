import yaml
import feedparser
import asyncio
import os
import json
import logging
import datetime
import hashlib
import PyRSS2Gen
import trafilatura
import xml.etree.ElementTree as ET
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from newspaper import Article, Config
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from dateutil import parser as date_parser
from xml.dom.minidom import parseString
import time

CONFIG_PATH = "sources.yaml"
STATE_PATH = "read_log.txt"
BASE_REPO_URL = "https://github.com/cornholio56/LeAggregator"  # ← your actual repo

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
- Use this exact HTML format for each Story's content:

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
    {{"title": "Story title", "expanded_html": "<div>...</div>"}}
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
            }
            # minItems removed → fixes Gemini schema validation error
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
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9,hr;q=0.8',
    }

    for attempt in range(3):
        try:
            # 1. Trafilatura with custom headers
            downloaded = trafilatura.fetch_url(url, decode=True, headers=headers)
            if downloaded:
                text = trafilatura.extract(downloaded, include_links=False)
                if text and len(text.strip()) > 600:
                    logger.info(f"Trafilatura success: {url}")
                    return text
        except Exception as e:
            logger.debug(f"Trafilatura attempt {attempt+1} failed: {e}")

        try:
            # 2. Newspaper3k fallback
            n_config = Config()
            n_config.browser_user_agent = headers['User-Agent']
            article = Article(url, config=n_config, fetch_images=False)
            article.download()
            article.parse()
            if len(article.text) > 600:
                logger.info(f"Newspaper success: {url}")
                return article.text
        except Exception as e:
            logger.debug(f"Newspaper attempt {attempt+1} failed: {e}")

        try:
            # 3. Crawl4AI (strongest anti-bot)
            async with AsyncWebCrawler(config=BrowserConfig(headless=True, verbose=False)) as crawler:
                result = await crawler.arun(
                    url=url,
                    config=CrawlerRunConfig(
                        cache_mode=CacheMode.BYPASS,
                        word_count_threshold=150,
                        user_agent=headers['User-Agent']
                    )
                )
                if result.success and result.markdown and len(result.markdown.strip()) > 600:
                    logger.info(f"Crawl4AI success: {url}")
                    return result.markdown
        except Exception as e:
            logger.debug(f"Crawl4AI attempt {attempt+1} failed: {e}")

        time.sleep(1.5)  # polite backoff

    logger.warning(f"All fetch methods failed for {url}")
    return None

def get_ai_response(model, prompt: str) -> dict:
    try:
        response = model.generate_content(
            contents=prompt,
            generation_config=genai.GenerationConfig(
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
        logger.error(f"Gemini error: {str(e)}")
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
        content_hash = hashlib.sha256(theme['expanded_html'].encode('utf-8')).hexdigest()
        guid_str = f"{category}-{content_hash[:16]}"
        items.append(
            PyRSS2Gen.RSSItem(
                title=theme["title"],
                link=BASE_REPO_URL,
                description=theme["expanded_html"],
                guid=PyRSS2Gen.Guid(guid_str, isPermaLink=False),
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

    with open(filename, "w", encoding="utf-8") as f:
        rss.write_xml(f)

async def process_category(category: str, last_run: datetime.datetime, seen_urls: set, model):
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
        return []

    prompt = GENERAL_INSTRUCTION_TEMPLATE.format(
        omit_categories=OMIT_CATEGORIES,
        articles='\n'.join(articles)
    )
    result = get_ai_response(model, prompt)
    return result.get('themes', [])

async def process_social_pulse(last_run: datetime.datetime, seen_urls: set, model):
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
        return []

    prompt = SOCIAL_PULSE_INSTRUCTION_TEMPLATE.format(social_signals='\n'.join(signals))
    result = get_ai_response(model, prompt)
    return result.get('themes', [])

async def main():
    now = datetime.datetime.now(datetime.timezone.utc)
    last_run_str = "1970-01-01T00:00:00Z"
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, 'r', encoding='utf-8') as f:
            last_run_str = f.read().strip()
    last_run = date_parser.parse(last_run_str)

    seen_urls = get_seen_urls()

    # Modern Gemini initialization
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-002",  # or gemini-1.5-pro-002 if quota allows
        generation_config=genai.GenerationConfig(
            temperature=0.1,
            top_p=0.95,
            max_output_tokens=2048
        )
    )

    for category in config['categories']:
        themes = await process_category(category, last_run, seen_urls, model)
        update_xml_feed(category, themes, now)

    # Social Pulse → only added to general
    pulse_themes = await process_social_pulse(last_run, seen_urls, model)
    if pulse_themes:
        update_xml_feed('general', pulse_themes, now)

    # Update last run time
    with open(STATE_PATH, 'w', encoding='utf-8') as f:
        f.write(now.isoformat())

if __name__ == "__main__":
    asyncio.run(main())
