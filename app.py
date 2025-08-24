# app.py
# To run this, you'll need to install the required libraries:
# pip install Flask Flask-Cors requests beautifulsoup4 spacy scikit-learn numpy
#
# You also need the spaCy model:
# python -m spacy download en_core_web_md

import re
import json
import requests
import numpy as np
import spacy
from collections import OrderedDict, Counter
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from sklearn.cluster import AgglomerativeClustering

# --- Basic Setup ---
app = Flask(__name__)

# --- Load the spaCy model once on startup ---
# This is a medium-sized model with word vectors, perfect for this task.
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading spaCy model 'en_core_web_md'...")
    print("This will happen only once. Please wait.")
    from spacy.cli import download
    download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")
    print("Model downloaded and loaded successfully.")


# --- CORS Configuration ---
# Allows requests from your frontend domains.
CORS(app, origins=[
    "https://seoblogy.com",
    "https://www.seoblogy.com",
    "http://localhost:5500", # For local testing
    "http://127.0.0.1:5500" # For local testing
])

# --- Constants ---
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
REQUEST_TIMEOUT = 20
MAX_CONTENT_SIZE = 500000 # Limit content size to 500KB to avoid memory issues

# --- Utility Functions ---

def analyze_semantic_clusters(text: str) -> dict:
    """
    Analyzes text using NLP to find and cluster semantically related key phrases.
    This uses scikit-learn's AgglomerativeClustering for more accurate results.
    """
    # Process the text with spaCy, limiting its size for performance.
    doc = nlp(text[:100000]) # Limit to first 100k chars

    # 1. Extract high-quality key phrases (noun chunks).
    # We filter for phrases that are between 2 and 5 words long.
    key_phrases = [
        chunk.text.lower() for chunk in doc.noun_chunks
        if 2 <= len(chunk.text.split()) < 5
    ]

    if not key_phrases:
        return {"message": "No suitable key phrases found for clustering."}

    # 2. Get the most frequent phrases to form a representative sample.
    phrase_counts = Counter(key_phrases)
    # We'll cluster the top 75 most frequent unique phrases.
    elite_phrases = [phrase for phrase, count in phrase_counts.most_common(75)]

    if len(elite_phrases) < 5: # Need at least a few phrases to find meaningful clusters
        return {"message": "Not enough unique phrases to perform clustering."}

    # 3. Get vector representations for our elite phrases.
    phrase_docs = [nlp(phrase) for phrase in elite_phrases]
    vectors = np.array([
        doc.vector for doc in phrase_docs if doc.has_vector and doc.vector_norm > 0
    ])

    if len(vectors) < 5:
        return {"message": "Could not generate enough vectors for clustering."}

    # 4. Perform Agglomerative Clustering.
    # This algorithm groups similar items together into clusters.
    # - n_clusters=None: Lets the algorithm decide the optimal number of clusters.
    # - distance_threshold: Controls how "tight" the clusters are. Lower is tighter.
    # - affinity='cosine': Measures similarity based on the angle between vectors (great for text).
    # - linkage='average': The criterion for merging clusters.
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.3, # This is a key parameter to tune. 0.3 is a good starting point.
        metric='cosine',
        linkage='average'
    ).fit(vectors)

    # 5. Format the results into a clean dictionary.
    num_clusters = clustering.n_clusters_
    clusters = {f"Topic Cluster {i+1}": [] for i in range(num_clusters)}
    
    # Map each phrase back to its assigned cluster ID.
    for i, phrase in enumerate(elite_phrases):
        if i < len(clustering.labels_):
            cluster_id = clustering.labels_[i]
            # We map the internal cluster ID to our human-readable key.
            clusters[f"Topic Cluster {cluster_id+1}"].append(phrase)
            
    # Filter out any empty clusters that might have been created
    final_clusters = {k: v for k, v in clusters.items() if v}

    return final_clusters if final_clusters else {"message": "No strong semantic clusters were identified."}


def clean_text(text: str) -> str:
    """Removes special characters and converts text to lowercase."""
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text or "")
    return text.lower()


def extract_keywords(text: str, total_words: int, top_n: int = 20) -> tuple[dict, dict]:
    """Extracts top keywords and calculates their density."""
    stopwords = {
        "the","and","to","of","a","in","for","is","on","with","that","as","by",
        "this","an","be","or","it","are","at","from","was","but","not","have","has",
        "you","your","our","their","they","we","he","she","them","his","her","its"
    }
    words = clean_text(text).split()
    words = [w for w in words if w not in stopwords and len(w) > 2]
    freq = Counter(words)
    top_keywords = dict(freq.most_common(top_n))
    
    density = {
        k: f"{(v / total_words * 100):.2f}%" for k, v in top_keywords.items()
    } if total_words > 0 else {}
    
    return top_keywords, density


def get_domain(url: str) -> str:
    """Extracts the netloc (domain) from a URL."""
    try:
        return urlparse(url).netloc
    except Exception:
        return ""


def heading_structure_score(soup: BeautifulSoup) -> int:
    """Scores heading structure (h1, h2, h3...) for logical order."""
    headings = [tag.name for tag in soup.find_all(re.compile(r"h[1-6]"))]
    if not headings:
        return 0
    
    score = 100
    last_level = 0
    # Penalize for skipping heading levels (e.g., h1 -> h3)
    for h in headings:
        level = int(h[1])
        if last_level != 0 and level - last_level > 1:
            score -= 20
        last_level = level
        
    # Penalize if there is no H1 tag
    if 'h1' not in headings:
        score -= 25
        
    return max(score, 0)


def build_report(url: str, soup: BeautifulSoup, response_status: int) -> OrderedDict:
    """Builds the final SEO report by calling all metric functions."""
    domain = get_domain(url)
    full_text = soup.get_text(" ", strip=True)
    total_words = len(full_text.split())

    report = OrderedDict()
    
    # --- Core Metrics (28+) ---
    report["1. URL"] = url
    report["2. Status Code"] = response_status
    
    title = (soup.title.string or "").strip() if soup.title else "Not Found"
    report["3. Title"] = title
    report["4. Title Length"] = len(title) if title != "Not Found" else 0

    desc = soup.find("meta", attrs={"name": "description"})
    meta_desc = desc.get("content", "").strip() if desc else "Not Found"
    report["5. Meta Description"] = meta_desc
    report["6. Meta Description Length"] = len(meta_desc) if meta_desc != "Not Found" else 0

    h_tags = {}
    for i in range(1, 4):
        hs = soup.find_all(f"h{i}")
        h_tags[f"{6+i}. H{i} Tags"] = " | ".join([h.get_text(strip=True) for h in hs]) or "Not Found"
    report.update(h_tags)

    report["10. Body Content (Preview)"] = full_text[:1000] + "..."
    report["11. Word Count"] = total_words

    canonical = soup.find("link", rel="canonical")
    report["12. Canonical URL"] = canonical.get("href") if canonical else "Not Found"
    
    robots = soup.find("meta", attrs={"name": "robots"})
    report["13. Robots Meta Tag"] = robots.get("content") if robots else "Not Found"

    report["14. HTTPS Usage"] = "Yes" if url.startswith("https") else "No"
    
    is_mixed = any(
        (tag.get('src') and str(tag.get('src')).startswith("http://")) or
        (tag.get('href') and str(tag.get('href')).startswith("http://"))
        for tag in soup.find_all(['img', 'script', 'link'])
    ) if report["14. HTTPS Usage"] == "Yes" else False
    report["15. Mixed Content"] = "Yes" if is_mixed else "No"

    internal_links, external_links = [], []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
            continue
        abs_url = urljoin(url, href)
        link_domain = get_domain(abs_url)
        entry = {"url": abs_url, "anchor": a.get_text(strip=True)[:100]} # Limit anchor length
        if link_domain == domain:
            internal_links.append(entry)
        else:
            external_links.append(entry)
    
    report["16. Internal Links Count"] = len(internal_links)
    report["17. External Links Count"] = len(external_links)
    report["18. Internal Links (Sample)"] = internal_links[:20]
    report["19. External Links (Sample)"] = external_links[:20]

    images = soup.find_all("img")
    report["20. Total Images"] = len(images)
    report["21. Images Missing ALT Text"] = sum(1 for img in images if not (img.get("alt") or "").strip())

    schema_scripts = soup.find_all("script", type="application/ld+json")
    schemas = []
    for script in schema_scripts:
        try:
            schemas.append(json.loads(script.string or "{}"))
        except json.JSONDecodeError:
            continue # Ignore malformed JSON
    report["22. Schema Markup Found"] = "Yes" if schemas else "No"
    report["22a. Schema Markup"] = schemas if schemas else "Not Found"

    favicon = soup.find("link", rel=lambda x: x and "icon" in x.lower())
    report["23. Favicon"] = favicon.get("href") if favicon else "Not Found"

    hreflangs = [link.get("href") for link in soup.find_all("link", rel="alternate", hreflang=True)]
    report["24. Hreflang Tags"] = " | ".join(hreflangs) if hreflangs else "Not Found"

    top_keywords, density = extract_keywords(full_text, total_words)
    report["25. Top Keywords (1-word)"] = top_keywords
    report["26. Keyword Density"] = density
    
    report["27. Heading Structure Score"] = heading_structure_score(soup)

    # --- ADVANCED METRIC ---
    # This is our new, powerful semantic analysis.
    try:
        report["28. Semantic Keyword Clusters"] = analyze_semantic_clusters(full_text)
    except Exception as e:
        report["28. Semantic Keyword Clusters"] = {"error": f"Failed during semantic analysis: {str(e)}"}

    return report

# --- API Routes ---

@app.route("/")
def home():
    """Home route to confirm the API is running."""
    return "✅ SEO Audit API with Advanced Semantics is running!"

@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})

@app.route("/audit")
def audit():
    """The main endpoint to perform the SEO audit."""
    url = request.args.get("url", "").strip()
    if not url:
        return jsonify({"success": False, "error": "URL parameter is missing."}), 400

    # Add http:// if no scheme is present
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    try:
        # Fetch the URL content
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=REQUEST_HEADERS, allow_redirects=True)
        resp.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        # Limit content size to prevent memory overload
        content = resp.text[:MAX_CONTENT_SIZE]
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(content, "html.parser")
        
        # Build the comprehensive report
        report = build_report(resp.url, soup, resp.status_code) # Use final URL after redirects
        
        return jsonify({"success": True, "data": report})

    except requests.exceptions.Timeout:
        return jsonify({"success": False, "error": "Request timed out. The target site took too long to respond."}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({"success": False, "error": f"Failed to fetch the URL: {str(e)}"}), 400
    except Exception as e:
        # General error handler for any other issues
        return jsonify({"success": False, "error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    # This allows you to run the app directly with `python app.py`
    app.run(debug=True, port=5001)
