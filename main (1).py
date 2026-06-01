from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncio
import httpx
import anthropic
import re
import os
import logging

logger = logging.getLogger("benchmark")

app = FastAPI(title="Outdoor Benchmark API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ── Models ────────────────────────────────────────────────────────────────────
class Dealer(BaseModel):
    name: str
    url: str

class BenchmarkRequest(BaseModel):
    product_name: str
    my_price: Optional[float] = None
    material: Optional[str] = None
    description: Optional[str] = None
    currency: str = "EUR"
    dealers: list[Dealer] = []
    platforms: list[str] = []

class PriceResult(BaseModel):
    source: str
    platform: str
    flag: str
    product_name: str
    price: Optional[float]
    original_price: Optional[float]
    url: Optional[str]
    image_url: Optional[str]
    notes: Optional[str]
    similarity: Optional[int]
    found: bool
    is_dealer: bool = False

# ── Helpers ───────────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

async def fetch_page(url: str, timeout: int = 15) -> str:
    try:
        async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True, timeout=timeout) as client:
            r = await client.get(url)
            return r.text if r.status_code == 200 else ""
    except Exception:
        return ""

def extract_price(html: str) -> Optional[float]:
    """Extract first price-like number from HTML."""
    patterns = [
        r'(\d{1,4}[.,]\d{2})\s*€',
        r'€\s*(\d{1,4}[.,]\d{2})',
        r'"price"\s*:\s*"?(\d{1,4}[.,]\d{2})',
        r'itemprop="price"[^>]*content="(\d{1,4}\.?\d{0,2})"',
        r'data-price="(\d{1,4}\.?\d{0,2})"',
    ]
    for p in patterns:
        m = re.search(p, html)
        if m:
            raw = m.group(1).replace(",", ".")
            try:
                val = float(raw)
                if 10 < val < 5000:
                    return val
            except:
                pass
    return None

def extract_image(html: str, base_url: str) -> Optional[str]:
    """Extract first product image from HTML."""
    patterns = [
        r'og:image["\s]+content="([^"]+)"',
        r'"image"\s*:\s*"([^"]+\.(?:jpg|jpeg|png|webp))"',
        r'<img[^>]+src="([^"]+(?:product|catalog|artikel)[^"]+\.(?:jpg|jpeg|png|webp))"',
    ]
    for p in patterns:
        m = re.search(p, html, re.IGNORECASE)
        if m:
            img = m.group(1)
            if img.startswith("http"):
                return img
            elif img.startswith("//"):
                return "https:" + img
            elif img.startswith("/"):
                from urllib.parse import urlparse
                parsed = urlparse(base_url)
                return f"{parsed.scheme}://{parsed.netloc}{img}"
    return None

def ask_claude_about_page(html: str, product_name: str, url: str) -> dict:
    """Use Claude to extract price + image from a page when regex fails."""
    snippet = html[:8000]
    msg = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system="Extract product info from HTML. Return ONLY JSON: {\"price\": number_or_null, \"image_url\": \"string_or_null\", \"product_found\": bool, \"product_name\": \"string_or_null\", \"notes\": \"string_or_null\"}",
        messages=[{"role": "user", "content": f"URL: {url}\nSearching for: {product_name}\n\nHTML snippet:\n{snippet}"}]
    )
    text = msg.content[0].text
    try:
        import json
        m = re.search(r'\{.*\}', text, re.DOTALL)
        return json.loads(m.group()) if m else {}
    except:
        return {}

async def scan_dealer(dealer: Dealer, product_name: str, my_price: Optional[float]) -> PriceResult:
    """Fetch dealer URL and extract price."""
    html = await fetch_page(dealer.url)
    price = extract_price(html) if html else None
    image = extract_image(html, dealer.url) if html else None
    product_found_name = None
    notes = None

    if html and not price:
        result = ask_claude_about_page(html, product_name, dealer.url)
        price = result.get("price")
        image = image or result.get("image_url")
        product_found_name = result.get("product_name")
        notes = result.get("notes")

    return PriceResult(
        source=dealer.name,
        platform=dealer.name,
        flag="🏪",
        product_name=product_found_name or product_name,
        price=price,
        original_price=None,
        url=dealer.url,
        image_url=image,
        notes=notes,
        similarity=100 if price else None,
        found=price is not None,
        is_dealer=True,
    )

async def search_platform(platform_name: str, product_name: str, material: str, currency: str) -> list[PriceResult]:
    """Use Claude with web search to find competitors on a given platform."""
    try:
        msg = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            system=f"""You are a European outdoor furniture pricing analyst.
Search for the specified product on the specified platform and return real results.
Return ONLY valid JSON array, no explanation:
[{{"product_name":"str","brand":"str","price":number,"original_price":number_or_null,"url":"str","image_url":"str_or_null","similarity":number_0_to_100,"notes":"str"}}]
Currency: {currency}. Return up to 3 results. If not found, return [].""",
            messages=[{"role": "user", "content": f"Platform: {platform_name}\nProduct to find: {product_name}\nMaterial: {material or 'steel + rope, outdoor dining chair, stackable'}"}],
            tools=[{"type": "web_search_20250305", "name": "web_search"}]
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        import json
        m = re.search(r'\[.*\]', text, re.DOTALL)
        if not m:
            return []
        items = json.loads(m.group())
        results = []
        for item in items:
            results.append(PriceResult(
                source=platform_name,
                platform=platform_name,
                flag=PLATFORM_FLAGS.get(platform_name, "🏪"),
                product_name=item.get("product_name", product_name),
                price=item.get("price"),
                original_price=item.get("original_price"),
                url=item.get("url"),
                image_url=item.get("image_url"),
                notes=item.get("notes"),
                similarity=item.get("similarity"),
                found=item.get("price") is not None,
                is_dealer=False,
            ))
        return results
    except Exception as e:
        logger.error(f"search_platform({platform_name}) failed: {e}")
        return []

PLATFORM_FLAGS = {
    "Amazon.de": "🇩🇪", "Amazon.fr": "🇫🇷", "Amazon.it": "🇮🇹",
    "Amazon.nl": "🇳🇱", "Amazon.es": "🇪🇸", "ManoMano": "🏪",
    "OTTO": "🏪", "Wayfair EU": "🏪", "Hornbach": "🏗", "OBI": "🏗",
    "Lusini": "🍽", "XXL Horeca": "🍽", "Metro": "🏬", "Skandic": "🏪",
    "Gastro Hero": "🍽", "Zederkof": "🇩🇰", "Manutan": "🏪",
    "Profishop": "🏪", "Blickfang": "🏪", "Natur24": "🌿",
    "Soennecken": "🏪", "Gastprodo": "🍽",
}

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}

@app.post("/benchmark")
async def benchmark(req: BenchmarkRequest):
    if not req.product_name:
        raise HTTPException(400, "product_name is required")

    results = []

    # 1. Scan dealer URLs in parallel (real fetch)
    if req.dealers:
        dealer_tasks = [scan_dealer(d, req.product_name, req.my_price) for d in req.dealers]
        dealer_results = await asyncio.gather(*dealer_tasks, return_exceptions=True)
        for r in dealer_results:
            if isinstance(r, PriceResult):
                results.append(r)

    # 2. Search platforms via Claude + web_search (batches of 4)
    platforms = req.platforms or list(PLATFORM_FLAGS.keys())[:12]
    BATCH = 4
    for i in range(0, len(platforms), BATCH):
        batch = platforms[i:i+BATCH]
        tasks = [search_platform(p, req.product_name, req.material, req.currency) for p in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        for group in batch_results:
            if isinstance(group, list):
                results.extend(group)

    # 3. Strategy summary
    found_prices = [r.price for r in results if r.price and not r.is_dealer]
    dealer_prices = [(r.source, r.price) for r in results if r.price and r.is_dealer]
    summary = None
    try:
        avg = sum(found_prices) / len(found_prices) if found_prices else None
        min_p = min(found_prices) if found_prices else None
        max_p = max(found_prices) if found_prices else None
        undercutters = [(s, p) for s, p in dealer_prices if req.my_price and p < req.my_price]

        msg = claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system="You are a pricing strategist. Write 3-4 sentences in English. No bullets.",
            messages=[{"role": "user", "content": f"""Product: {req.product_name}
My price: {req.my_price} {req.currency}
Market avg: {avg:.2f if avg else 'N/A'} | Min: {min_p} | Max: {max_p}
Dealers undercutting me: {undercutters}
Give a concise pricing strategy assessment."""}]
        )
        summary = msg.content[0].text.strip()
    except:
        pass

    found = [r for r in results if r.found]
    not_found_platforms = [r.source for r in results if not r.found and not r.is_dealer]

    return {
        "product": req.product_name,
        "my_price": req.my_price,
        "currency": req.currency,
        "results": [r.model_dump() for r in found],
        "not_found": list(set(not_found_platforms)),
        "summary": summary,
        "stats": {
            "market_avg": round(sum(found_prices) / len(found_prices), 2) if found_prices else None,
            "market_min": min(found_prices) if found_prices else None,
            "market_max": max(found_prices) if found_prices else None,
            "total_found": len(found),
        }
    }
