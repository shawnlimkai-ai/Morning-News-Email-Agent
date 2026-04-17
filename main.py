#!/usr/bin/env python3
"""Daily news briefing bot — NewsAPI + RSS → Groq LLM → Resend email."""

import argparse
import logging
import os
import re
import sys
from datetime import datetime, timedelta

from openai import OpenAI
import feedparser
import requests
import resend
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
FROM_EMAIL = os.getenv("FROM_EMAIL", "briefing@example.com")
TO_EMAIL = os.getenv("TO_EMAIL", "shawnlimkai@gmail.com")
TO_EMAIL2 = os.getenv("TO_EMAIL2", "jozeeang@gmail.com")  # optional second recipient, comma-separated if multiple
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
FALLBACK_MODELS: list[str] = []

# ── RSS Feeds by Category ─────────────────────────────────────────────────────

FEEDS = {
    "world_markets": [
        "https://feeds.reuters.com/reuters/topNews",
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://feeds.bbci.co.uk/news/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "https://feeds.a.dj.com/rss/RSSWorldNews.xml",            # WSJ World
        "https://www.aljazeera.com/xml/rss/all.xml",
        "https://feeds.skynews.com/feeds/rss/world.xml",
        "https://rss.dw.com/rdf/rss-en-world",                    # Deutsche Welle
    ],
    "macro_economics": [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://feeds.bbci.co.uk/news/business/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/Economy.xml",
        "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml",
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "https://www.ft.com/rss/home/uk",
    ],
    "deals": [
        "https://feeds.reuters.com/reuters/mergersNews",
        "https://techcrunch.com/category/venture/feed/",
        "https://rss.nytimes.com/services/xml/rss/nyt/DealBook.xml",
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "https://pehub.com/feed/",
    ],
    "ai_tech": [
        "https://techcrunch.com/category/artificial-intelligence/feed/",
        "https://techcrunch.com/feed/",
        "https://feeds.arstechnica.com/arstechnica/index",
        "https://www.theverge.com/rss/index.xml",
        "https://venturebeat.com/category/ai/feed/",
        "https://venturebeat.com/feed/",
        "https://www.wired.com/feed/category/business/latest/rss",
        "https://feeds.reuters.com/reuters/technologyNews",
        "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    ],
}

# ── NewsAPI Queries ────────────────────────────────────────────────────────────

NEWSAPI_QUERIES = {
    "world_markets": (
        "geopolitical OR war OR sanctions OR election OR commodities "
        "OR oil OR shipping OR \"Strait of Hormuz\" OR \"stock market\" "
        "OR \"trade dispute\" OR coup OR ceasefire OR NATO OR UN"
    ),
    "macro_economics": (
        "\"Federal Reserve\" OR \"central bank\" OR inflation OR GDP "
        "OR employment OR \"interest rates\" OR \"monetary policy\" "
        "OR \"fiscal policy\" OR \"trade balance\" OR \"rate cut\" OR \"rate hike\" "
        "OR \"CPI\" OR \"PCE\" OR IMF OR \"World Bank\" OR tariff"
    ),
    "deals": (
        "acquisition OR merger OR IPO OR buyout OR \"private equity\" "
        "OR takeover OR \"take-private\" OR fundraise OR fundraising "
        "OR \"venture capital\" OR \"secondary sale\" OR \"sponsor exit\""
    ),
    "ai_tech": (
        "\"artificial intelligence\" OR \"large language model\" OR semiconductor "
        "OR OpenAI OR Anthropic OR \"machine learning\" OR \"AI model\" "
        "OR GPU OR Nvidia OR \"foundation model\" OR robotics OR quantum"
    ),
}

# ── Claude Prompts ────────────────────────────────────────────────────────────

PROMPTS = {
    "world_markets": """\
You are a sharp geopolitical and markets analyst writing a buy-side morning briefing for hedge fund PMs and senior executives. Cut through noise and give a genuine analytical view — not a summary of headlines.

Write a WORLD & MARKETS section with 4–6 bullets on the most important geopolitical and market developments. If fewer than 4 stories genuinely matter today, cover only those — do not pad with filler.

Each bullet must:
- Open with a bold headline title on its own line (format: • **Punchy headline — 6-10 words, specific**)
- Follow with a full analytical paragraph of 3–5 sentences on the next line (no bullet prefix on the paragraph line)
- Cover what happened (specific numbers, names, countries), why it matters, and the second-order implications
- Use language like "this matters because...", "the implication is...", "markets should price in..."
- Be opinionated — give a view, flag genuine escalation vs. noise

Include: major market moves and drivers (equities, oil, FX), geopolitical flashpoints (wars, elections, sanctions, trade disputes, shipping), political developments with clear economic/market implications, cross-border crises.

Exclude entirely: entertainment, sports, lifestyle, celebrity, routine political noise with zero market impact.

CRITICAL FORMAT — output exactly like this, nothing else:
• **Punchy Headline — Specific Detail**
Full analytical paragraph here. 3–5 sentences. No bullet prefix on this line.

• **Next Headline**
Next paragraph.""",

    "macro_economics": """\
You are a senior macro economist writing a buy-side morning briefing for portfolio managers and CFOs. Identify what actually moves markets and policy — not every data point.

Write a MACRO & ECONOMICS section with 4–6 bullets on the most important macro developments. If only 2 stories genuinely matter, cover only those — do not pad.

Each bullet must:
- Open with a bold headline title on its own line (format: • **Headline — specific rate/number if applicable**)
- Follow with a full analytical paragraph of 3–5 sentences on the next line (no bullet prefix on the paragraph line)
- Include specific numbers (rate levels, % changes, index readings) wherever available
- Cover what happened, why it matters for rates/FX/equities, and the policy or market implication
- Be opinionated — flag pivots in central bank language, surprise vs. consensus

Include: central bank decisions and forward guidance, CPI/PCE/NFP surprises (only if genuinely surprising vs. consensus), GDP prints and major forecast revisions, fiscal policy changes, notable commentary from Fed/ECB/BoJ/BoE heads.

Exclude entirely: minor forecasts from second-tier institutions, routine weekly data with no surprise, consumer sentiment trivia, commodity moves unless ±5%+ on a meaningful catalyst.

CRITICAL FORMAT — output exactly like this, nothing else:
• **Punchy Headline — Specific Rate or Number**
Full analytical paragraph here. 3–5 sentences. No bullet prefix on this line.

• **Next Headline**
Next paragraph.""",

    "deals": """\
You are a seasoned M&A banker writing a buy-side morning briefing for dealmakers and investors. Surface transactions that actually matter — not every press release.

Write a DEALS & TRANSACTIONS section with 4–6 bullets on the most important deals. If fewer than 4 deals genuinely matter, cover only those — quality over quantity.

Each bullet must:
- Open with a bold headline title on its own line (format: • **Acquirer Buys Target for $Xbn — Deal Type**)
- Follow with a full analytical paragraph of 3–5 sentences on the next line (no bullet prefix on the paragraph line)
- Always state: deal size (or "undisclosed"), buyer/investor, target, and deal type (acquisition, merger, IPO, PE buyout, take-private, sponsor exit, Series X)
- Cover deal terms, strategic rationale, market significance, and what it signals about the sector

Prioritize in order: (1) closed/announced M&A and mergers, (2) PE buyouts and take-privates, (3) IPOs and public market debuts, (4) sponsor exits, (5) VC fundraises ONLY if $1B+ or genuinely strategic.

Exclude entirely: seed rounds, Series A/B below $100M, minor partnerships, small VC rounds, sports team purchases (a footballer buying a fifth-tier club is NOT deals news), real estate under $5B.

CRITICAL FORMAT — output exactly like this, nothing else:
• **Buyer Acquires Target for $Xbn — Acquisition**
Full analytical paragraph. Always include deal size, buyer, target, deal type. 3–5 sentences. No bullet prefix on this line.

• **Next Deal Headline**
Next paragraph.""",

    "ai_tech": """\
You are a senior technology analyst covering AI and frontier tech, writing a buy-side morning briefing for executives and investors. Distinguish genuine signal from hype.

Write an AI & TECH section with 4–6 bullets on the most important AI and technology developments. If only 2 stories genuinely matter, cover only those — do not pad.

Each bullet must:
- Open with a bold headline title on its own line (format: • **Company/Model — What Happened — Key Significance**)
- Follow with a full analytical paragraph of 3–5 sentences on the next line (no bullet prefix on the paragraph line)
- Cover what was released/announced (with specifics — benchmarks, parameters, pricing), competitive implications, and what it means for the industry
- Be opinionated — distinguish genuine breakthroughs from marketing, call out hype vs. substance

Include: major AI model releases (with benchmarks or concrete claims), large funding rounds, key partnerships, leadership changes at major AI labs, semiconductor/compute developments (Nvidia, TSMC, AMD, custom silicon), notable research breakthroughs, significant product launches from Apple/Google/Microsoft/Meta/Amazon.

Exclude entirely: minor product updates, routine partnership announcements, speculative rumors, incremental version bumps, gadget reviews, consumer tech trivia.

CRITICAL FORMAT — output exactly like this, nothing else:
• **Company Releases Model-X — Key Benchmark or Claim**
Full analytical paragraph here. 3–5 sentences. No bullet prefix on this line.

• **Next Headline**
Next paragraph.""",
}

HEADERS = {
    "world_markets": "WORLD & MARKETS",
    "macro_economics": "MACRO & ECONOMICS",
    "deals": "DEALS & TRANSACTIONS",
    "ai_tech": "AI & TECH",
}

CATEGORIES = ["world_markets", "macro_economics", "deals", "ai_tech"]


# ── Fetching ──────────────────────────────────────────────────────────────────

def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text).strip()


def fetch_newsapi(category: str, query: str) -> list[dict]:
    if not NEWSAPI_KEY:
        return []
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "from": yesterday,
                "pageSize": 20,
                "apiKey": NEWSAPI_KEY,
            },
            timeout=10,
        )
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        log.info(f"  NewsAPI [{category}]: {len(articles)} articles")
        return [
            {
                "title": a.get("title", "").strip(),
                "description": strip_html(a.get("description", "") or "")[:150],
                "source": a.get("source", {}).get("name", "NewsAPI"),
            }
            for a in articles
            if a.get("title") and "[Removed]" not in (a.get("title", ""))
        ]
    except Exception as e:
        log.warning(f"  NewsAPI [{category}] error: {e}")
        return []


def fetch_rss(category: str, urls: list[str]) -> list[dict]:
    articles = []
    for url in urls:
        try:
            feed = feedparser.parse(url)
            count = 0
            for entry in feed.entries[:8]:
                title = (entry.get("title") or "").strip()
                if not title:
                    continue
                summary = strip_html(
                    entry.get("summary") or entry.get("description") or ""
                )[:150]
                articles.append({
                    "title": title,
                    "description": summary,
                    "source": feed.feed.get("title") or url,
                })
                count += 1
            if count:
                log.info(f"  RSS [{category}] {url.split('/')[2]}: {count} entries")
        except Exception as e:
            log.warning(f"  RSS [{category}] {url.split('/')[2]} failed: {e}")
    return articles


def format_for_claude(articles: list[dict]) -> str:
    seen: set[str] = set()
    lines = []
    for a in articles:
        title = a["title"]
        key = title.lower()[:55]
        if key in seen:
            continue
        seen.add(key)
        line = f"- [{a['source']}] {title}"
        if a.get("description"):
            line += f" — {a['description']}"
        lines.append(line)
        if len(lines) >= 20:
            break
    return "\n".join(lines) if lines else "No articles available."


# ── Summarization ─────────────────────────────────────────────────────────────

def _build_combined_prompt(articles_text_by_cat: dict[str, str]) -> str:
    sections = ""
    for cat in CATEGORIES:
        sections += f"\n\n---\n## {HEADERS[cat]}\n\n{PROMPTS[cat]}\n\nARTICLES:\n{articles_text_by_cat[cat]}"
    return (
        "You are writing a complete morning briefing with 4 sections. "
        "Write ALL 4 sections in order, following each section's instructions exactly.\n"
        + sections
        + "\n\n---\nOUTPUT: Use EXACTLY these markers, start immediately with the first one, no preamble:\n\n"
        "===WORLD & MARKETS===\n[bullets]\n\n"
        "===MACRO & ECONOMICS===\n[bullets]\n\n"
        "===DEALS & TRANSACTIONS===\n[bullets]\n\n"
        "===AI & TECH===\n[bullets]"
    )


def _parse_sections(text: str, articles_by_cat: dict[str, list[dict]]) -> dict[str, str]:
    summaries = {}
    for cat in CATEGORIES:
        marker = re.escape(HEADERS[cat])
        m = re.search(rf"==={marker}===(.*?)(?====|\Z)", text, re.DOTALL)
        if m:
            content = m.group(1).strip()
            # Strip any preamble before the first bullet/bold line
            lines = content.splitlines()
            for idx, ln in enumerate(lines):
                if re.match(r"^\s*[•\-\*]|\*\*", ln.strip()):
                    content = "\n".join(lines[idx:])
                    break
            summaries[cat] = content
        else:
            summaries[cat] = "\n".join(
                f"• {a['title']}" for a in articles_by_cat.get(cat, [])[:5]
            ) or "No data available."
    return summaries


def summarize_all(client: OpenAI, articles_by_cat: dict[str, list[dict]]) -> dict[str, str]:
    import time
    articles_text_by_cat = {cat: format_for_claude(arts) for cat, arts in articles_by_cat.items()}
    prompt = _build_combined_prompt(articles_text_by_cat)

    for i, model in enumerate([MODEL] + FALLBACK_MODELS):
        if i > 0:
            time.sleep(15)  # brief pause before trying next model
        try:
            log.info(f"Summarizing all sections with {model}...")
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
                temperature=0.7,
            )
            text = resp.choices[0].message.content.strip()
            log.info(f"  Response: {len(text)} chars")
            return _parse_sections(text, articles_by_cat)
        except Exception as e:
            log.warning(f"  {model} failed: {e}")

    log.error("All models failed — using raw article titles")
    return {
        cat: "\n".join(f"• {a['title']}" for a in arts[:5]) or "No data available."
        for cat, arts in articles_by_cat.items()
    }


# ── Email Building ─────────────────────────────────────────────────────────────

def bullets_to_html(text: str) -> str:
    items = []
    lines = [l.strip() for l in text.splitlines()]
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line:
            i += 1
            continue

        # Strip leading bullet prefix
        for prefix in ("• ", "- ", "* ", "· ", "– "):
            if line.startswith(prefix):
                line = line[len(prefix):]
                break
        line = re.sub(r"^\d+\.\s+", "", line)
        if not line:
            i += 1
            continue

        # New format: **Title** alone on this line → next non-empty line is the body paragraph
        if re.match(r"^\*\*(.+?)\*\*\s*$", line):
            headline = re.match(r"^\*\*(.+?)\*\*\s*$", line).group(1).strip()
            body = ""
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                nxt = lines[j].strip()
                is_next_title = bool(
                    re.match(r"^[•\-\*·–]\s+\*\*", nxt) or re.match(r"^\*\*", nxt)
                )
                if not is_next_title:
                    body = nxt
                    i = j + 1
                else:
                    i += 1
            else:
                i += 1
            content = f'<span class="bullet-title">{headline}</span>'
            if body:
                content += f'<span class="bullet-body">{body}</span>'
            items.append(f"<li>{content}</li>")

        # Fallback: **Title**: body or **Title** — body on the same line
        elif re.match(r"^\*\*(.+?)\*\*[:\s—–\-]+(.*)", line):
            m = re.match(r"^\*\*(.+?)\*\*[:\s—–\-]+(.*)", line, re.DOTALL)
            headline = m.group(1).strip()
            body = m.group(2).strip()
            content = f'<span class="bullet-title">{headline}</span>'
            if body:
                content += f'<span class="bullet-body">{body}</span>'
            items.append(f"<li>{content}</li>")
            i += 1

        else:
            items.append(f"<li>{line}</li>")
            i += 1

    return "\n".join(items) if items else "<li>No data available.</li>"


def build_html(summaries: dict, date_str: str) -> str:
    sections = ""
    for cat in CATEGORIES:
        header = HEADERS[cat]
        bullets = bullets_to_html(summaries.get(cat, "No data available."))
        sections += f"""
        <div class="section">
          <h2>{header}</h2>
          <ul>{bullets}</ul>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Morning Briefing — {date_str}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Georgia, serif;
    background: #f6f6f3;
    color: #1a1a1a;
    margin: 0;
    padding: 24px 0;
  }}
  .wrapper {{
    max-width: 660px;
    margin: 0 auto;
    background: #ffffff;
    border: 1px solid #e4e4e0;
    border-radius: 6px;
    overflow: hidden;
  }}
  .header {{
    background: #111111;
    color: #ffffff;
    padding: 22px 32px;
  }}
  .header h1 {{
    margin: 0 0 4px;
    font-size: 18px;
    font-weight: 600;
    letter-spacing: 0.3px;
  }}
  .header p {{
    margin: 0;
    font-size: 12px;
    color: #888;
    letter-spacing: 0.5px;
  }}
  .body {{
    padding: 4px 32px 28px;
  }}
  .section {{
    margin-top: 26px;
  }}
  h2 {{
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #888;
    border-bottom: 1px solid #ebebeb;
    padding-bottom: 7px;
    margin: 0 0 10px;
  }}
  ul {{
    margin: 0;
    padding: 0;
    list-style: none;
  }}
  li {{
    position: relative;
    padding: 7px 0 7px 18px;
    font-size: 13.5px;
    line-height: 1.65;
    color: #252525;
    border-bottom: 1px solid #f3f3f0;
  }}
  li:last-child {{ border-bottom: none; }}
  li::before {{
    content: "▸";
    position: absolute;
    left: 0;
    color: #bbb;
    font-size: 11px;
    top: 9px;
  }}
  .bullet-title {{
    display: block;
    font-size: 14.5px;
    font-weight: 700;
    color: #111;
    margin-bottom: 3px;
  }}
  .bullet-body {{
    display: block;
    font-size: 13px;
    color: #444;
    line-height: 1.6;
  }}
  .footer {{
    text-align: center;
    padding: 14px 32px;
    font-size: 11px;
    color: #bbb;
    border-top: 1px solid #ebebeb;
    background: #fafaf8;
  }}
</style>
</head>
<body>
<div class="wrapper">
  <div class="header">
    <h1>Morning Briefing</h1>
    <p>{date_str.upper()}</p>
  </div>
  <div class="body">{sections}</div>
  <div class="footer">Generated by {MODEL.split("/")[-1]} · {date_str}</div>
</div>
</body>
</html>"""


def build_text(summaries: dict, date_str: str) -> str:
    lines = [f"MORNING BRIEFING — {date_str}", "=" * 52, ""]
    for cat in CATEGORIES:
        header = HEADERS[cat]
        lines += [header, "-" * len(header), summaries.get(cat, "No data."), ""]
    return "\n".join(lines)


# ── Email Sending ─────────────────────────────────────────────────────────────

def send_email(html: str, subject: str) -> bool:
    if not RESEND_API_KEY:
        log.error("RESEND_API_KEY not set")
        return False
    resend.api_key = RESEND_API_KEY
    try:
        result = resend.Emails.send({
            "from": FROM_EMAIL,
            "to": [TO_EMAIL],
            "subject": subject,
            "html": html,
        })
        log.info(f"Email sent — id: {result.get('id', result)}")
        return True
    except Exception as e:
        log.error(f"Resend error: {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Daily news briefing bot")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print briefing to console, skip email",
    )
    args = parser.parse_args()

    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not args.dry_run and not RESEND_API_KEY:
        missing.append("RESEND_API_KEY")
    if missing:
        log.error(f"Missing required env vars: {', '.join(missing)}")
        sys.exit(1)

    if not NEWSAPI_KEY:
        log.warning("NEWSAPI_KEY not set — relying on RSS feeds only")

    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
    )
    date_str = datetime.now().strftime("%B %d, %Y")
    subject = f"Morning Briefing — {date_str}"

    # Fetch all categories, applying cross-category dedup
    articles_by_cat: dict[str, list[dict]] = {}
    global_seen: set[str] = set()

    for cat in CATEGORIES:
        log.info(f"[{cat}] Fetching news...")
        all_articles = fetch_newsapi(cat, NEWSAPI_QUERIES[cat]) + fetch_rss(cat, FEEDS[cat])
        filtered = [a for a in all_articles if a["title"].lower()[:55] not in global_seen]
        n_deduped = len(all_articles) - len(filtered)
        log.info(f"[{cat}] {len(all_articles)} total ({n_deduped} cross-deduped) -> {len(filtered)}")
        articles_by_cat[cat] = filtered
        for a in filtered[:40]:
            global_seen.add(a["title"].lower()[:55])

    summaries = summarize_all(client, articles_by_cat)

    if args.dry_run:
        print("\n" + build_text(summaries, date_str))
        log.info("Dry run complete — no email sent.")
    else:
        html = build_html(summaries, date_str)
        if not send_email(html, subject):
            sys.exit(1)
        log.info(f"Briefing delivered to {TO_EMAIL}.")


if __name__ == "__main__":
    main()
