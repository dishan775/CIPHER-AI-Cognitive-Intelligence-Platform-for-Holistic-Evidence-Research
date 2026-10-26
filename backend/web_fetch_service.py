"""
Unified web content fetch, advanced HTML cleanup, multi-source image extraction,
lead-article expansion, and structured NLP analysis.
Used by /api/scrape, project URL import, and research intelligence pipelines.
Integrates secure backend Gemini structuring with robust local NLP fallback.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from gemini_service import structure_article_content
from nlp_service import nlp_service

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
}

# Substrings indicating tracking pixels, ad banners, gravatars, icons or spacer graphics
IMAGE_IGNORE_PATTERNS = [
    r"1x1",
    r"spacer",
    r"pixel",
    r"beacon",
    r"analytics",
    r"tracking",
    r"gravatar\.com",
    r"avatar",
    r"logo-small",
    r"badge",
    r"sprite",
    r"wp-includes",
    r"doubleclick",
    r"ad-banner",
    r"advert",
    r"sponsor",
    r"\.svg(\?|$)",
    r"\.ico(\?|$)",
]


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _is_valid_image_url(img_url: str) -> bool:
    """Validate that an image URL is an HTTP(S) link and not an ad or tracking asset."""
    if not img_url or not img_url.startswith(("http://", "https://")):
        return False
    lower_url = img_url.lower()
    for pat in IMAGE_IGNORE_PATTERNS:
        if re.search(pat, lower_url):
            return False
    return True


def extract_article_images(soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
    """
    Extract high-relevance article images:
    1. OpenGraph / Twitter Card hero images (og:image, twitter:image)
    2. In-article content images (resolving data-src, data-original, srcset, src)
    Filtering out tiny icons, ads, placeholders, and tracking pixels.
    """
    extracted_images: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()

    # 1. Meta / OpenGraph Hero Image (highest resolution cover)
    for meta_prop in ("og:image", "og:image:secure_url", "twitter:image", "twitter:image:src"):
        meta_node = soup.find("meta", property=meta_prop) or soup.find("meta", attrs={"name": meta_prop})
        if meta_node and meta_node.get("content"):
            content_url = meta_node["content"].strip()
            abs_hero = urljoin(base_url, content_url)
            if _is_valid_image_url(abs_hero) and abs_hero not in seen_urls:
                seen_urls.add(abs_hero)
                extracted_images.append({
                    "url": abs_hero,
                    "alt": "Article Cover Visual",
                    "is_cover": True,
                })
                break

    # 2. Main content images: Search all img tags across page
    for img in soup.find_all("img"):
        raw_src = img.get("src", "").strip()
        data_src = (img.get("data-src") or img.get("data-original") or img.get("data-lazy-src") or "").strip()

        # Prioritize real image source if src is a placeholder or blank gif
        if any(ph in raw_src.lower() for ph in ["placeholder", "spacer", "blank", "transparent", "data:image"]):
            src = data_src
        else:
            src = data_src or raw_src

        # Handle srcset if src was missing or empty
        if (not src or src.startswith("data:")) and img.get("srcset"):
            srcset_parts = [p.strip().split(" ")[0] for p in img["srcset"].split(",") if p.strip()]
            if srcset_parts:
                src = srcset_parts[-1]

        if not src or src.startswith("data:"):
            continue

        abs_img = urljoin(base_url, src)
        if not _is_valid_image_url(abs_img) or abs_img in seen_urls:
            continue

        # Check explicit dimensions to discard small decorative icons
        width = img.get("width")
        height = img.get("height")
        try:
            if width and int(str(width).replace("px", "")) < 45:
                continue
            if height and int(str(height).replace("px", "")) < 45:
                continue
        except (ValueError, TypeError):
            pass

        # Capture alt text or caption
        alt_text = _normalize_whitespace(img.get("alt") or img.get("title") or "")
        # Fallback to parent figure / article caption if available
        if not alt_text:
            fig = img.find_parent("figure") or img.find_parent("article")
            if fig:
                fig_head = fig.find(["figcaption", "h2", "h3", "h4"])
                if fig_head:
                    alt_text = _normalize_whitespace(fig_head.get_text())

        if not alt_text:
            alt_text = "Scientific Research Visual"

        seen_urls.add(abs_img)
        extracted_images.append({
            "url": abs_img,
            "alt": alt_text[:200],
            "is_cover": False if extracted_images else True,
        })

        if len(extracted_images) >= 16:
            break

    return extracted_images


def fetch_url_content(url: str, timeout: int = 25) -> dict[str, Any]:
    """
    Fetch a public URL and return cleaned readable text, structured summary,
    key news bullet points, and extracted relevant images.
    Intelligently expands lead stories on news portals/homepages.
    """
    response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")

    # Extract initial images BEFORE decomposing non-content tags
    images = extract_article_images(soup, url)

    # Decompose scripts, styles, navigation, footer, forms, and ads
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "iframe", "svg", "form", "aside"]):
        tag.decompose()

    # Extract Page Title
    title_tag = soup.title.string if soup.title and soup.title.string else ""
    if not title_tag:
        h1 = soup.find("h1")
        title_tag = h1.get_text() if h1 else "Untitled Web Article"
    title = _normalize_whitespace(title_tag)

    # Extract Domain / Host
    parsed_url = urlparse(url)
    domain_name = parsed_url.netloc.replace("www.", "")

    # Extract Headings
    headings: list[dict[str, str]] = []
    for level in ("h1", "h2", "h3", "h4"):
        for node in soup.find_all(level)[:30]:
            txt = _normalize_whitespace(node.get_text(" ", strip=True))
            if txt and len(txt) > 2:
                headings.append({"level": level, "text": txt[:500]})

    # Check for lead article link if this is a portal / homepage
    lead_article_content = ""
    lead_article_title = ""
    lead_art_tag = soup.find("article")
    if lead_art_tag:
        a_tag = lead_art_tag.find("a", href=True)
        if a_tag and a_tag["href"]:
            href = a_tag["href"].strip()
            if not href.startswith("#") and not href.lower().startswith("javascript:"):
                cand_url = urljoin(url, href)
                # If candidate is a deep article link
                if cand_url != url and any(marker in cand_url for marker in ["/202", "/article", "/story", "/news/", "/post/"]):
                    try:
                        lead_resp = requests.get(cand_url, headers=DEFAULT_HEADERS, timeout=12)
                        if lead_resp.status_code == 200:
                            lead_soup = BeautifulSoup(lead_resp.content, "html.parser")
                            # Extract lead story images
                            lead_imgs = extract_article_images(lead_soup, cand_url)
                            existing_urls = {img["url"] for img in images}
                            for limg in lead_imgs:
                                if limg["url"] not in existing_urls:
                                    images.append(limg)
                                    existing_urls.add(limg["url"])
                            # Clean lead soup
                            for t in lead_soup(["script", "style", "noscript", "header", "footer", "nav", "iframe", "svg", "form", "aside"]):
                                t.decompose()
                            lead_ps = [
                                _normalize_whitespace(p.get_text(" ", strip=True))
                                for p in lead_soup.find_all("p")
                                if len(_normalize_whitespace(p.get_text(" ", strip=True))) > 25
                            ]
                            if lead_ps:
                                lead_article_content = "\n\n".join(lead_ps)
                            lead_h1 = lead_soup.find("h1")
                            if lead_h1:
                                lead_article_title = _normalize_whitespace(lead_h1.get_text())
                    except Exception as e:
                        print(f"Lead article expansion notice: {e}")

    # Extract clean paragraphs from current page
    articles = soup.find_all("article")
    page_paragraphs: List[str] = []

    if len(articles) > 1:
        # Multi-article portal or news feed: compile digest
        for art in articles[:25]:
            art_head = art.find(["h2", "h3", "h4", "h1"])
            head_text = _normalize_whitespace(art_head.get_text()) if art_head else ""
            p_nodes = [
                _normalize_whitespace(p.get_text())
                for p in art.find_all("p")
                if len(_normalize_whitespace(p.get_text())) > 20
            ]
            story_body = " ".join(p_nodes)
            if head_text and story_body:
                page_paragraphs.append(f"{head_text}: {story_body}")
            elif story_body:
                page_paragraphs.append(story_body)
            elif head_text:
                page_paragraphs.append(head_text)

    if len(page_paragraphs) < 3:
        body_container = soup.find("main") or soup.find("article") or soup.find("div", class_=re.compile(r"content|post|article|body", re.I)) or soup
        page_paragraphs = [
            _normalize_whitespace(p.get_text(" ", strip=True))
            for p in body_container.find_all("p")
            if len(_normalize_whitespace(p.get_text(" ", strip=True))) > 25
        ]

    if len(page_paragraphs) < 3:
        page_paragraphs = [
            _normalize_whitespace(p.get_text(" ", strip=True))
            for p in soup.find_all(["p", "li"])
            if len(_normalize_whitespace(p.get_text(" ", strip=True))) > 30
        ]

    # Combine lead story full text with portal digest
    all_paragraphs = []
    if lead_article_content:
        all_paragraphs.extend(lead_article_content.split("\n\n"))
    if page_paragraphs:
        all_paragraphs.extend(page_paragraphs)

    if all_paragraphs:
        text_content = "\n\n".join(all_paragraphs)
    else:
        text_content = _normalize_whitespace(soup.get_text(" ", strip=True))

    if lead_article_title and ("MIT News" in title or "Home" in title or title == "Untitled Web Article"):
        title = f"{lead_article_title} | {title}"

    # Extract Links
    link_items: list[dict[str, str]] = []
    seen_link_urls: set[str] = set()
    for link in soup.find_all("a", href=True)[:80]:
        link_text = _normalize_whitespace(link.get_text(" ", strip=True))
        href = link["href"].strip()
        if not href or href.startswith("#") or href.lower().startswith("javascript:"):
            continue
        abs_url = urljoin(url, href)
        if abs_url in seen_link_urls or abs_url == url:
            continue
        seen_link_urls.add(abs_url)
        if link_text and len(link_text) > 3:
            link_items.append({"text": link_text[:200], "url": abs_url})
        if len(link_items) >= 20:
            break

    # Meta Description
    meta_description = ""
    desc_tag = soup.find("meta", attrs={"name": re.compile(r"description", re.I)}) or soup.find("meta", property="og:description")
    if desc_tag and desc_tag.get("content"):
        meta_description = _normalize_whitespace(desc_tag["content"])

    # Canonical URL
    canonical = url
    canon_tag = soup.find("link", rel=lambda v: v and "canonical" in v.lower())
    if canon_tag and canon_tag.get("href"):
        canonical = urljoin(url, canon_tag["href"])

    full_text = text_content[:500_000]
    word_count = len(nlp_service.get_words(full_text))

    # Structured Content Formatting (Backend Gemini with seamless local NLP fallback)
    structured_info = structure_article_content(
        title=title,
        text=full_text,
        url=canonical,
        nlp_service=nlp_service,
    )
    summary_text = structured_info.get("summary", "")
    key_news_items = structured_info.get("key_news", [])
    structured_by = structured_info.get("structured_by", "nlp_local")

    # Local NLP outline
    structured_outline = nlp_service.structure_web_content(full_text, title=title, url=canonical)

    # Simple list of URLs for backwards compatibility
    image_urls = [img["url"] for img in images]

    return {
        "url": url,
        "canonical_url": canonical,
        "title": title,
        "domain": domain_name,
        "meta_description": meta_description,
        "full_text": full_text,
        "full_content": full_text,
        "word_count": word_count,
        "reading_time_min": max(1, round(word_count / 200)),
        "summary": summary_text,
        "key_news": key_news_items,
        "images": images,
        "image_urls": image_urls,
        "headings": headings[:20],
        "links": link_items,
        "structured": structured_outline,
        "structured_by": structured_by,
    }
