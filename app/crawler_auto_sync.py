import hashlib
import os
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag

import chromadb
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

VECTORSTORE_DIR = BASE_DIR / "vectorstore"

START_URL = "https://www.vamadevatechnosolutions.com/"

MAX_PAGES = 50

CHUNK_SIZE = 1200

CHUNK_OVERLAP = 200

REQUEST_DELAY = 0.5

# Official Snowflake curriculum currently published by
# Vamadeva Techno Solutions.
SNOWFLAKE_CURRICULUM_URL = (
    "https://vamadevatechnosolutions.com/assets/curriculum/snowflake.pdf"
)


# ============================================================
# OPENAI
# ============================================================

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# ============================================================
# CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=str(VECTORSTORE_DIR)
)

collection = chroma_client.get_or_create_collection(
    name="business_documents"
)


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "User-Agent": (
        "BusinessAIChatbot/1.0 "
        "(website knowledge crawler)"
    )
})


# ============================================================
# URL HELPERS
# ============================================================

def normalize_url(url):
    """
    Remove fragments and normalize URLs.
    """

    url, _ = urldefrag(url)

    parsed = urlparse(url)

    scheme = parsed.scheme.lower()

    netloc = parsed.netloc.lower()

    path = parsed.path or "/"

    normalized = f"{scheme}://{netloc}{path}"

    return normalized


def is_same_domain(url, base_url):
    """
    Check whether the URL belongs to the same domain.
    """

    url_domain = urlparse(url).netloc.lower()

    base_domain = urlparse(base_url).netloc.lower()

    return url_domain == base_domain


def is_pdf_url(url):
    """
    Check whether a URL points to a PDF.
    """

    return urlparse(url).path.lower().endswith(".pdf")


def should_crawl(url):
    """
    Decide whether a URL should be crawled.
    """

    parsed = urlparse(url)

    if parsed.scheme not in ["http", "https"]:
        return False

    blocked_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".svg",
        ".ico",
        ".mp4",
        ".mp3",
        ".zip",
        ".rar",
        ".css",
        ".js",
        ".xml",
        ".json",
        ".xlsx",
        ".xls",
        ".doc",
        ".docx",
        ".ppt",
        ".pptx",
    )

    path_lower = parsed.path.lower()

    if path_lower.endswith(blocked_extensions):
        return False

    return True


# ============================================================
# TEXT CLEANING
# ============================================================

def repair_mojibake(text):
    """
    Repair common UTF-8 -> Windows-1252/Latin-1 decoding
    problems such as:
        â€™
        â€œ
        â€
        â€“
        â€” 
        â†’
    """

    if not text:
        return text

    # Only attempt the conversion when typical mojibake
    # markers are present.
    markers = (
        "Ã",
        "Â",
        "â",
        "ð",
    )

    if not any(marker in text for marker in markers):
        return text

    try:
        repaired = text.encode(
            "latin-1"
        ).decode(
            "utf-8"
        )

        # Use the repaired version only if it reduces
        # obvious mojibake markers.
        original_bad = sum(
            text.count(marker)
            for marker in markers
        )

        repaired_bad = sum(
            repaired.count(marker)
            for marker in markers
        )

        if repaired_bad < original_bad:
            return repaired

    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    return text


def clean_extracted_text(text):
    """
    Clean website/PDF text without aggressively changing
    legitimate content.
    """

    if not text:
        return ""

    text = repair_mojibake(text)

    # Normalize common non-breaking spaces.
    text = text.replace("\xa0", " ")

    # Repair a small set of common extraction joins.
    replacements = {
        "andperformance": "and performance",
        "cloudproviders": "cloud providers",
        "withhands-on": "with hands-on",
        "real-worldexperience": "real-world experience",
        "datawarehousing": "data warehousing",
        "dataengineering": "data engineering",
        "performanceoptimization": "performance optimization",
        "multi-clustermanagement": "multi-cluster management",
        "cloud-based": "cloud-based",
        "Snowflakeâ€™s": "Snowflake's",
        "Snowflakeâ€™": "Snowflake'",
        "â€™": "'",
        "â€œ": '"',
        "â€": '"',
        "â€“": "-",
        "â€”": "-",
        "â†’": "->",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Put a space between common lower-case/upper-case extraction
    # joins, but do not globally split normal words.
    text = re.sub(
        r"(?<=[a-z])(?=[A-Z][a-z]{2,})",
        " ",
        text,
    )

    # Collapse whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def deduplicate_sentences(text):
    """
    Remove exact repeated long fragments that sometimes occur
    because a website repeats navigation/footer blocks.
    """

    if not text:
        return ""

    words = text.split()

    if len(words) < 80:
        return text

    # Preserve content while removing repeated adjacent blocks.
    result = []
    seen_blocks = set()

    block_size = 30

    for index in range(0, len(words), block_size):

        block = words[
            index:index + block_size
        ]

        if not block:
            continue

        block_text = " ".join(block)

        key = re.sub(
            r"[^a-z0-9]+",
            " ",
            block_text.lower(),
        ).strip()

        if key in seen_blocks:
            continue

        seen_blocks.add(key)

        result.extend(block)

    return " ".join(result)


# ============================================================
# DOWNLOAD PAGE
# ============================================================

def download_page(url):
    """
    Download a webpage and return its HTML.
    """

    try:

        response = session.get(
            url,
            timeout=30,
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type",
            "",
        ).lower()

        if "text/html" not in content_type:
            return None

        # requests normally detects the encoding from the
        # response headers. If the website omits it, UTF-8 is
        # the safest default for this site.
        if not response.encoding:
            response.encoding = "utf-8"

        return response.text

    except requests.RequestException as error:

        print(
            f"Could not download {url}: {error}"
        )

        return None


# ============================================================
# DOWNLOAD PDF
# ============================================================

def download_pdf(url):
    """
    Download and extract text from a PDF.
    """

    try:

        response = session.get(
            url,
            timeout=60,
        )

        response.raise_for_status()

        content = response.content

        if not content.startswith(b"%PDF"):
            print(
                f"  Not a valid PDF: {url}"
            )
            return None

        temporary_pdf = (
            BASE_DIR
            / "data"
            / "_crawler_temp.pdf"
        )

        temporary_pdf.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_pdf.write_bytes(
            content
        )

        reader = PdfReader(
            str(temporary_pdf)
        )

        pages = []

        for page in reader.pages:

            try:
                page_text = (
                    page.extract_text()
                    or ""
                )
            except Exception as error:
                print(
                    f"  Could not extract one PDF page: {error}"
                )
                page_text = ""

            if page_text.strip():
                pages.append(
                    page_text
                )

        try:
            temporary_pdf.unlink()
        except OSError:
            pass

        text = "\n".join(pages)

        return clean_extracted_text(text)

    except Exception as error:

        print(
            f"Could not download/read PDF {url}: {error}"
        )

        return None


# ============================================================
# EXTRACT PAGE CONTENT
# ============================================================

def extract_page_content(html, url):
    """
    Extract useful text and links from HTML.
    """

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    # Remove elements that usually contain
    # non-content information.
    for tag in soup([
        "script",
        "style",
        "noscript",
        "svg",
        "iframe",
    ]):

        tag.decompose()

    # Page title
    title = ""

    if soup.title:

        title = soup.title.get_text(
            " ",
            strip=True,
        )

    # Prefer main/article content when the website provides it.
    # This reduces repeated header/footer/navigation text.
    content_root = (
        soup.find("main")
        or soup.find("article")
        or soup.find(
            id=re.compile(
                r"(main|content|course)",
                re.I,
            )
        )
    )

    if content_root is None:
        content_root = soup

    # Main text
    text_parts = []

    for tag in content_root.find_all([
        "h1",
        "h2",
        "h3",
        "h4",
        "p",
        "li",
        "td",
        "th",
    ]):

        text = tag.get_text(
            " ",
            strip=True,
        )

        if text:
            text_parts.append(text)

    text = "\n".join(
        text_parts
    )

    text = clean_extracted_text(
        text
    )

    text = deduplicate_sentences(
        text
    )

    # Find links
    links = set()

    for link in soup.find_all(
        "a",
        href=True,
    ):

        absolute_url = urljoin(
            url,
            link["href"],
        )

        absolute_url = normalize_url(
            absolute_url
        )

        if should_crawl(
            absolute_url
        ):

            links.add(
                absolute_url
            )

    return title, text, links


# ============================================================
# TEXT CHUNKING
# ============================================================

def split_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
):
    """
    Split text into overlapping chunks.
    """

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[
            start:end
        ].strip()

        if chunk:
            chunks.append(chunk)

        start += (
            chunk_size
            - overlap
        )

    return chunks


# ============================================================
# EMBEDDINGS
# ============================================================

def create_embeddings(texts):
    """
    Create OpenAI embeddings.
    """

    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )

    return [
        item.embedding
        for item in response.data
    ]


# ============================================================
# DELETE OLD WEBSITE DATA
# ============================================================

def delete_existing_website_data():
    """
    Delete previously indexed website chunks.

    This is important because the previous crawler stored
    corrupted/duplicated website text. PDF documents such as
    fabric-admin.pdf are not touched.
    """

    print()
    print(
        "Removing old website chunks..."
    )

    try:

        existing = collection.get(
            where={
                "source_type": "website"
            },
            include=[],
        )

        ids = existing.get(
            "ids",
            [],
        )

        if ids:

            collection.delete(
                ids=ids
            )

            print(
                f"  Removed {len(ids)} old website chunks."
            )

        else:

            print(
                "  No old website chunks found."
            )

    except Exception as error:

        print(
            "Could not remove old website data:"
        )

        print(error)

        raise


# ============================================================
# STORE DOCUMENT
# ============================================================

def store_document(
    url,
    title,
    text,
    source_type="website",
):
    """
    Chunk and store content inside ChromaDB.
    """

    if not text:
        return 0

    chunks = split_text(
        text
    )

    if not chunks:
        return 0

    print(
        f"  Creating {len(chunks)} chunks..."
    )

    batch_size = 50

    total_stored = 0

    for start in range(
        0,
        len(chunks),
        batch_size,
    ):

        batch = chunks[
            start:start + batch_size
        ]

        embeddings = create_embeddings(
            batch
        )

        ids = []

        metadatas = []

        for index, chunk in enumerate(
            batch
        ):

            absolute_index = (
                start + index
            )

            raw_id = (
                f"{source_type}|"
                f"{url}|"
                f"{absolute_index}"
            )

            chunk_id = hashlib.sha256(
                raw_id.encode(
                    "utf-8"
                )
            ).hexdigest()

            ids.append(
                chunk_id
            )

            metadatas.append({
                "source": url,
                "source_type": source_type,
                "title": title,
                "chunk": absolute_index,
            })

        collection.upsert(
            ids=ids,
            documents=batch,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        total_stored += len(
            batch
        )

    return total_stored


# ============================================================
# STORE OFFICIAL SNOWFLAKE CURRICULUM
# ============================================================

def store_snowflake_curriculum():
    """
    Download and index the official Snowflake curriculum PDF.

    The PDF is kept separate from website HTML so the chatbot
    can retrieve reliable course-specific information such as
    the published 60-hour session duration.
    """

    print()
    print(
        "Indexing official Snowflake curriculum..."
    )

    text = download_pdf(
        SNOWFLAKE_CURRICULUM_URL
    )

    if not text:

        print(
            "  Snowflake curriculum could not be read."
        )

        return 0

    # Add an explicit title so retrieval can distinguish this
    # document from general Snowflake webpage text.
    title = (
        "Vamadeva Snowflake Training "
        "Official Curriculum"
    )

    document_text = (
        "Vamadeva Techno Solutions - "
        "Snowflake Training Official Curriculum. "
        + text
    )

    stored = store_document(
        SNOWFLAKE_CURRICULUM_URL,
        title,
        document_text,
        source_type="website_curriculum",
    )

    print(
        f"  Snowflake curriculum stored: {stored} chunks"
    )

    return stored


# ============================================================
# AUTOMATIC CHANGE-DETECTION CRAWLER
# ============================================================

STATE_FILE = BASE_DIR / "website_sync_state.json"
SYNC_INTERVAL_SECONDS = 3 * 60 * 60


def load_sync_state():
    if not STATE_FILE.exists():
        return {
            "pages": {},
            "curriculum": {},
            "last_sync": None,
        }

    try:
        import json
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception as error:
        print(f"[AUTO SYNC] Could not read state file: {error}")
        return {
            "pages": {},
            "curriculum": {},
            "last_sync": None,
        }


def save_sync_state(state):
    import json
    temp_file = STATE_FILE.with_suffix(".tmp")
    temp_file.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temp_file.replace(STATE_FILE)


def content_hash(text):
    return hashlib.sha256(
        (text or "").encode("utf-8")
    ).hexdigest()


def delete_source(source, source_type):
    try:
        existing = collection.get(
            where={
                "$and": [
                    {"source": source},
                    {"source_type": source_type},
                ]
            },
            include=[],
        )
        ids = existing.get("ids", [])
        if ids:
            collection.delete(ids=ids)
        return len(ids)
    except Exception as error:
        print(
            f"[AUTO SYNC] Could not delete old chunks for {source}: {error}"
        )
        raise



def get_source_ids(source, source_type):
    existing = collection.get(
        where={
            "$and": [
                {"source": source},
                {"source_type": source_type},
            ]
        },
        include=[],
    )
    return existing.get("ids", [])


def replace_source(source, title, text, source_type):
    """Upsert new chunks first, then remove obsolete old chunks."""
    old_ids = set(get_source_ids(source, source_type))
    stored = store_document(source, title, text, source_type=source_type)
    current_ids = set(get_source_ids(source, source_type))
    obsolete_ids = list(old_ids - current_ids)
    if obsolete_ids:
        collection.delete(ids=obsolete_ids)
    return stored

def discover_website(start_url=START_URL, max_pages=MAX_PAGES):
    """Download the site and return successfully read pages and links."""
    start_url = normalize_url(start_url)
    queue = deque([start_url])
    visited = set()
    pages = {}
    failed_urls = set()

    while queue and len(pages) < max_pages:
        url = queue.popleft()

        if url in visited:
            continue
        visited.add(url)

        if not is_same_domain(url, start_url):
            continue
        if not should_crawl(url):
            continue
        if is_pdf_url(url):
            continue

        print(f"  Checking: {url}")
        html = download_page(url)

        if not html:
            failed_urls.add(url)
            continue

        title, text, links = extract_page_content(html, url)

        if text:
            pages[url] = {
                "title": title,
                "text": text,
                "hash": content_hash(title + "\n" + text),
            }
        else:
            pages[url] = {
                "title": title,
                "text": "",
                "hash": content_hash(title),
            }

        for link in sorted(links):
            if link not in visited and is_same_domain(link, start_url):
                queue.append(link)

        time.sleep(REQUEST_DELAY)

    return pages, failed_urls


def sync_website_once():
    """Run one incremental website synchronization."""
    from datetime import datetime

    print()
    print("=" * 60)
    print("VAMADEVA AUTOMATIC KNOWLEDGE SYNC")
    print("=" * 60)
    print("Checking website for changes...")

    state = load_sync_state()
    old_pages = state.get("pages", {})

    pages, failed_urls = discover_website()

    changed = 0
    unchanged = 0
    added = 0
    removed = 0
    chunks_added = 0

    for url, page in pages.items():
        old = old_pages.get(url)

        if old and old.get("hash") == page["hash"]:
            unchanged += 1
            continue

        if old:
            print(f"  CHANGED: {url}")
            changed += 1
        else:
            print(f"  NEW: {url}")
            added += 1

        stored = replace_source(
            url,
            page["title"],
            page["text"],
            source_type="website",
        )
        chunks_added += stored

    # Only remove pages when the crawl was complete enough to make
    # a reliable deletion decision. A failed request must not erase
    # good knowledge from ChromaDB.
    if not failed_urls:
        current_urls = set(pages.keys())
        for old_url in set(old_pages.keys()) - current_urls:
            print(f"  REMOVED: {old_url}")
            delete_source(old_url, "website")
            removed += 1
            old_pages.pop(old_url, None)
    else:
        print(
            f"  {len(failed_urls)} page(s) could not be checked; "
            "skipping deletion of missing pages."
        )

    # Save successful page fingerprints.
    new_state_pages = dict(old_pages)
    for url, page in pages.items():
        new_state_pages[url] = {
            "hash": page["hash"],
            "title": page["title"],
        }

    # Official Snowflake curriculum PDF is checked separately.
    curriculum_changed = sync_snowflake_curriculum(state)

    state["pages"] = new_state_pages
    state["last_sync"] = datetime.now().isoformat(timespec="seconds")
    save_sync_state(state)

    print()
    print("SYNC COMPLETED")
    print(f"  Pages checked: {len(pages)}")
    print(f"  Unchanged: {unchanged}")
    print(f"  New: {added}")
    print(f"  Changed: {changed}")
    print(f"  Removed: {removed}")
    print(f"  Website chunks updated: {chunks_added}")
    print(f"  Snowflake curriculum changed: {'YES' if curriculum_changed else 'NO'}")
    print(f"  Next automatic check: 3 hours")
    print()

    return {
        "pages_checked": len(pages),
        "unchanged": unchanged,
        "new": added,
        "changed": changed,
        "removed": removed,
        "chunks_added": chunks_added,
        "curriculum_changed": curriculum_changed,
    }


def sync_snowflake_curriculum(state):
    """Update the official Snowflake curriculum only when its content changes."""
    print("  Checking official Snowflake curriculum...")

    text = download_pdf(SNOWFLAKE_CURRICULUM_URL)
    if not text:
        print("  Snowflake curriculum check failed; keeping existing data.")
        return False

    title = "Vamadeva Snowflake Training Official Curriculum"
    document_text = (
        "Vamadeva Techno Solutions - "
        "Snowflake Training Official Curriculum. "
        + text
    )
    current_hash = content_hash(document_text)
    old = state.get("curriculum", {})

    if old.get("hash") == current_hash:
        print("  Snowflake curriculum: unchanged")
        return False

    if old:
        print("  Snowflake curriculum: CHANGED")
    else:
        print("  Snowflake curriculum: NEW")

    stored = replace_source(
        SNOWFLAKE_CURRICULUM_URL,
        title,
        document_text,
        source_type="website_curriculum",
    )

    print(f"  Snowflake curriculum chunks updated: {stored}")

    state["curriculum"] = {
        "hash": current_hash,
        "title": title,
    }
    return True


def run_once():
    try:
        return sync_website_once()
    except Exception as error:
        print()
        print("[AUTO SYNC ERROR]")
        print(error)
        print("Existing ChromaDB knowledge has been left in place where possible.")
        return None


def run_forever():
    """Run immediately, then repeat every 3 hours."""
    print()
    print("VAMADEVA AUTO SYNC STARTED")
    print("Automatic change detection: every 3 hours")
    print("Press CTRL+C to stop.")
    print()

    while True:
        run_once()
        print("[AUTO SYNC] Sleeping for 3 hours...")
        try:
            time.sleep(SYNC_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print()
            print("[AUTO SYNC] Stopped.")
            break


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_forever()
