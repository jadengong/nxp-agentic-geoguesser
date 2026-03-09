import re
import json
import urllib.request
import urllib.parse
import ollama

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "GeoGuesserAI/1.0 (educational project)"}


def _wiki_get(params: dict) -> dict:
    url = WIKI_API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_location(ollama_result: str) -> str | None:
    """Extract 'Country, City' from the ollama result and return just the city."""
    match = re.search(
        r"Most likely location[:\s\[]*([A-Za-z ,]+?)[\]\.]*$",
        ollama_result,
        re.MULTILINE | re.IGNORECASE,
    )
    if not match:
        return None
    raw = match.group(1).strip()
    # Ollama returns "Country, City" — use just the city for Wikipedia
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) == 2:
        return parts[1]  # e.g. "France, Paris" -> "Paris"
    return raw


def _get_summary(location: str) -> str | None:
    """
    Try two strategies to get a Wikipedia summary:
    1. Direct page extract by title (works for most cities)
    2. Geosearch by coordinates as fallback
    """
    # Strategy 1: direct extract by opensearch title
    search_data = _wiki_get({
        "action": "opensearch",
        "search": location,
        "limit": 1,
        "format": "json",
    })
    titles = search_data[1]
    if titles:
        title = titles[0]
        print(f"[fun_fact] Found Wikipedia page: {title}")
        extract_data = _wiki_get({
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "format": "json",
        })
        pages = extract_data.get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract", "").strip()
            if extract:
                return extract[:1500]

    # Strategy 2: geosearch fallback via coordinates
    print(f"[fun_fact] Falling back to geosearch for: {location}")
    coord_data = _wiki_get({
        "action": "query",
        "titles": titles[0] if titles else location,
        "prop": "coordinates",
        "format": "json",
    })
    pages = coord_data.get("query", {}).get("pages", {})
    lat, lon = None, None
    for page in pages.values():
        coords = page.get("coordinates", [])
        if coords:
            lat, lon = coords[0]["lat"], coords[0]["lon"]
            break

    if not lat:
        print(f"[fun_fact] No coordinates found for: {location}")
        return None

    geo_data = _wiki_get({
        "action": "query",
        "list": "geosearch",
        "gscoord": f"{lat}|{lon}",
        "gsradius": 10000,
        "gslimit": 1,
        "format": "json",
    })
    nearby = geo_data.get("query", {}).get("geosearch", [])
    if not nearby:
        return None
    nearby_title = nearby[0]["title"]
    print(f"[fun_fact] Geosearch top result: {nearby_title}")

    summary_data = _wiki_get({
        "action": "query",
        "titles": nearby_title,
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "format": "json",
    })
    pages = summary_data.get("query", {}).get("pages", {})
    for page in pages.values():
        extract = page.get("extract", "").strip()
        if extract:
            return extract[:1500]

    return None


def get_fun_fact(location: str) -> str:
    """Get a fun fact about the location using Wikipedia + Ollama."""
    summary = _get_summary(location)
    if not summary:
        return f"Could not find Wikipedia data for {location}."

    print(f"[fun_fact] Asking Ollama for a fun fact about {location}...")
    response = ollama.chat(
        model="llava",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Based on this Wikipedia excerpt about {location}:\n\n"
                    f"{summary}\n\n"
                    "Give me one short, surprising, and genuinely interesting fun fact. "
                    "1-2 sentences only. No preamble like 'Fun fact:' — just the fact."
                ),
            }
        ],
    )
    return (response.get("message") or {}).get("content", "").strip()