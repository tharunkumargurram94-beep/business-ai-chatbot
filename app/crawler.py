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


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

VECTORSTORE_DIR = BASE_DIR / "vectorstore"

START_URL = "https://www.vamadevatechnosolutions.com/"

MAX_PAGES = 50

CHUNK_SIZE = 1200

CHUNK_OVERLAP = 200

REQUEST_DELAY = 0.5


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
        ".pptx"
    )

    path_lower = parsed.path.lower()

    if path_lower.endswith(blocked_extensions):
        return False

    return True


# ============================================================
# DOWNLOAD PAGE
# ============================================================

def download_page(url):
    """
    Download a webpage.
    """

    try:

        response = session.get(
            url,
            timeout=20
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type",
            ""
        ).lower()

        if "text/html" not in content_type:
            return None

        return response.text

    except requests.RequestException as error:

        print(
            f"Could not download {url}: {error}"
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
        "html.parser"
    )


    # Remove elements that usually contain
    # non-content information.
    for tag in soup([
        "script",
        "style",
        "noscript",
        "svg",
        "iframe"
    ]):

        tag.decompose()


    # Page title
    title = ""

    if soup.title:
        title = soup.title.get_text(
            " ",
            strip=True
        )


    # Main text
    text_parts = []


    for tag in soup.find_all([
        "h1",
        "h2",
        "h3",
        "h4",
        "p",
        "li"
    ]):

        text = tag.get_text(
            " ",
            strip=True
        )

        if text:
            text_parts.append(text)


    text = "\n".join(
        text_parts
    )


    # Clean whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()


    # Find links
    links = set()


    for link in soup.find_all(
        "a",
        href=True
    ):

        absolute_url = urljoin(
            url,
            link["href"]
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
    overlap=CHUNK_OVERLAP
):
    """
    Split website text into overlapping chunks.
    """

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

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
        input=texts
    )

    return [
        item.embedding
        for item in response.data
    ]


# ============================================================
# STORE PAGE
# ============================================================

def store_page(
    url,
    title,
    text
):
    """
    Chunk and store webpage content
    inside ChromaDB.
    """

    if not text:
        return 0


    chunks = split_text(text)


    if not chunks:
        return 0


    print(
        f"  Creating {len(chunks)} chunks..."
    )


    # Process embeddings in batches
    batch_size = 50

    total_stored = 0


    for start in range(
        0,
        len(chunks),
        batch_size
    ):

        batch = chunks[
            start:start + batch_size
        ]


        embeddings = create_embeddings(
            batch
        )


        ids = []

        metadatas = []


        for index, chunk in enumerate(batch):

            absolute_index = start + index


            raw_id = (
                f"{url}_{absolute_index}"
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
                "source_type": "website",
                "title": title,
                "chunk": absolute_index
            })


        collection.upsert(
            ids=ids,
            documents=batch,
            embeddings=embeddings,
            metadatas=metadatas
        )


        total_stored += len(batch)


    return total_stored


# ============================================================
# CRAWLER
# ============================================================

def crawl_website(
    start_url=START_URL,
    max_pages=MAX_PAGES
):
    """
    Crawl the website and store its content.
    """

    start_url = normalize_url(
        start_url
    )


    queue = deque([
        start_url
    ])


    visited = set()


    pages_crawled = 0

    chunks_stored = 0


    print()
    print("=" * 60)
    print("VAMADEVA WEBSITE CRAWLER")
    print("=" * 60)
    print()
    print(
        f"Starting URL: {start_url}"
    )
    print(
        f"Maximum pages: {max_pages}"
    )
    print()


    while queue and pages_crawled < max_pages:

        url = queue.popleft()


        if url in visited:
            continue


        visited.add(url)


        if not is_same_domain(
            url,
            start_url
        ):
            continue


        if not should_crawl(url):
            continue


        print(
            f"[{pages_crawled + 1}/{max_pages}] "
            f"Crawling: {url}"
        )


        html = download_page(
            url
        )


        if not html:
            continue


        title, text, links = extract_page_content(
            html,
            url
        )


        if not text:

            print(
                "  No readable text found."
            )

        else:

            print(
                f"  Title: {title}"
            )

            print(
                f"  Text: {len(text)} characters"
            )


            stored = store_page(
                url,
                title,
                text
            )


            print(
                f"  Stored: {stored} chunks"
            )


            chunks_stored += stored


        pages_crawled += 1


        # Add new links to queue
        for link in sorted(links):

            if link not in visited:

                if is_same_domain(
                    link,
                    start_url
                ):

                    queue.append(
                        link
                    )


        time.sleep(
            REQUEST_DELAY
        )


    print()
    print("=" * 60)
    print("CRAWLING COMPLETED")
    print("=" * 60)
    print()
    print(
        f"Pages crawled: {pages_crawled}"
    )
    print(
        f"Chunks stored: {chunks_stored}"
    )
    print()
    print(
        "Website content is now available in ChromaDB."
    )
    print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    crawl_website()