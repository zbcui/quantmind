"""
LLM Wiki — Notion-based personal knowledge base (Karpathy pattern).

Usage:
    python llm_wiki.py ingest  <file_or_url>   # Ingest a source into the wiki
    python llm_wiki.py query   "your question"  # Ask the wiki a question
    python llm_wiki.py lint                     # Health-check the wiki
    python llm_wiki.py status                   # Show wiki stats

Requires: NOTION_TOKEN and NOTION_WIKI_ROOT_ID in data/llm_wiki_config.json
          LLM config in data/llm_config.json (same as QuantMind)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from notion_client import Client

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent / "data" / "llm_wiki_config.json"
LLM_CONFIG_PATH = Path(__file__).resolve().parent / "data" / "llm_config.json"

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise SystemExit(f"Config not found: {CONFIG_PATH}\nRun setup first.")
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

def load_llm_config() -> dict:
    if not LLM_CONFIG_PATH.exists():
        raise SystemExit(f"LLM config not found: {LLM_CONFIG_PATH}")
    return json.loads(LLM_CONFIG_PATH.read_text(encoding="utf-8"))

def get_notion() -> Client:
    cfg = load_config()
    return Client(auth=cfg["notion_token"])

def get_root_id() -> str:
    return load_config()["wiki_root_id"]

# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------
def llm_call(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """Call the configured LLM (OpenAI-compatible or Ollama)."""
    cfg = load_llm_config()
    provider = cfg.get("provider", "OpenAI")
    model = cfg.get("model", "gpt-4o-mini")
    temperature = cfg.get("temperature", 0.3)

    if provider == "Ollama":
        import requests
        base = cfg.get("base_url", "http://localhost:11434")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = requests.post(f"{base}/api/chat", json={
            "model": model, "messages": messages, "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    else:
        from openai import OpenAI
        base_url = cfg.get("base_url", "https://api.openai.com")
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        client = OpenAI(api_key=cfg.get("api_key", ""), base_url=base_url)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=temperature,
        )
        return resp.choices[0].message.content

# ---------------------------------------------------------------------------
# Notion helpers
# ---------------------------------------------------------------------------
def find_child_page(notion: Client, parent_id: str, title: str) -> str | None:
    """Find a child page by title under a parent. Returns page_id or None."""
    children = notion.blocks.children.list(block_id=parent_id)
    for block in children["results"]:
        if block["type"] == "child_page" and block["child_page"]["title"] == title:
            return block["id"]
    return None

def create_page(notion: Client, parent_id: str, title: str, markdown: str = "") -> str:
    """Create a child page with optional markdown content. Returns page_id."""
    children = markdown_to_blocks(markdown) if markdown else []
    page = notion.pages.create(
        parent={"page_id": parent_id},
        properties={"title": [{"text": {"content": title}}]},
        children=children[:100],  # Notion limit: 100 blocks per create
    )
    # append remaining blocks if > 100
    pid = page["id"]
    if len(children) > 100:
        for i in range(100, len(children), 100):
            notion.blocks.children.append(block_id=pid, children=children[i:i+100])
    return pid

def append_to_page(notion: Client, page_id: str, markdown: str) -> None:
    """Append markdown content as blocks to an existing page."""
    blocks = markdown_to_blocks(markdown)
    for i in range(0, len(blocks), 100):
        notion.blocks.children.append(block_id=page_id, children=blocks[i:i+100])

def read_page_text(notion: Client, page_id: str) -> str:
    """Read all text content from a Notion page."""
    blocks = notion.blocks.children.list(block_id=page_id)
    lines = []
    for b in blocks["results"]:
        btype = b["type"]
        data = b.get(btype, {})
        rich = data.get("rich_text", [])
        text = "".join(r.get("plain_text", "") for r in rich)
        if btype.startswith("heading"):
            level = btype[-1]  # heading_1 -> 1
            lines.append(f"{'#' * int(level)} {text}")
        elif btype == "bulleted_list_item":
            lines.append(f"- {text}")
        elif btype == "numbered_list_item":
            lines.append(f"1. {text}")
        elif btype == "to_do":
            checked = "x" if data.get("checked") else " "
            lines.append(f"- [{checked}] {text}")
        elif btype == "code":
            lang = data.get("language", "")
            lines.append(f"```{lang}\n{text}\n```")
        elif btype == "divider":
            lines.append("---")
        elif btype == "child_page":
            lines.append(f"📄 [[{b['child_page']['title']}]]")
        else:
            lines.append(text)
    return "\n".join(lines)

def markdown_to_blocks(md: str) -> list[dict]:
    """Convert simple markdown to Notion blocks."""
    allowed_code_languages = {
        "abap", "abc", "agda", "arduino", "ascii art", "assembly", "bash", "basic", "bnf",
        "c", "c#", "c++", "clojure", "coffeescript", "coq", "css", "dart", "dhall", "diff",
        "docker", "ebnf", "elixir", "elm", "erlang", "f#", "flow", "fortran", "gherkin",
        "glsl", "go", "graphql", "groovy", "haskell", "hcl", "html", "idris", "java",
        "javascript", "json", "julia", "kotlin", "latex", "less", "lisp", "livescript",
        "llvm ir", "lua", "makefile", "markdown", "markup", "matlab", "mathematica",
        "mermaid", "nix", "notion formula", "objective-c", "ocaml", "pascal", "perl", "php",
        "plain text", "powershell", "prolog", "protobuf", "purescript", "python", "r",
        "racket", "reason", "ruby", "rust", "sass", "scala", "scheme", "scss", "shell",
        "smalltalk", "solidity", "sql", "swift", "toml", "typescript", "vb.net", "verilog",
        "vhdl", "visual basic", "webassembly", "xml", "yaml", "java/c/c++/c#",
    }
    language_aliases = {
        "plaintext": "plain text",
        "plain_text": "plain text",
        "text": "plain text",
        "js": "javascript",
        "ts": "typescript",
        "py": "python",
        "sh": "shell",
    }

    def normalize_code_language(lang: str) -> str:
        normalized = language_aliases.get(lang.strip().lower(), lang.strip().lower())
        return normalized if normalized in allowed_code_languages else "plain text"

    blocks = []
    lines = md.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        # code block
        if line.startswith("```"):
            lang = normalize_code_language(line[3:].strip() or "plain text")
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            blocks.append({
                "type": "code",
                "code": {
                    "rich_text": [{"type": "text", "text": {"content": "\n".join(code_lines)}}],
                    "language": lang,
                },
            })
            i += 1
            continue
        # headings
        if line.startswith("### "):
            blocks.append({"type": "heading_3", "heading_3": {
                "rich_text": [{"type": "text", "text": {"content": line[4:]}}]}})
        elif line.startswith("## "):
            blocks.append({"type": "heading_2", "heading_2": {
                "rich_text": [{"type": "text", "text": {"content": line[3:]}}]}})
        elif line.startswith("# "):
            blocks.append({"type": "heading_1", "heading_1": {
                "rich_text": [{"type": "text", "text": {"content": line[2:]}}]}})
        # bullets
        elif line.startswith("- ") or line.startswith("* "):
            blocks.append({"type": "bulleted_list_item", "bulleted_list_item": {
                "rich_text": [{"type": "text", "text": {"content": line[2:]}}]}})
        # numbered
        elif re.match(r"^\d+\.\s", line):
            text = re.sub(r"^\d+\.\s", "", line)
            blocks.append({"type": "numbered_list_item", "numbered_list_item": {
                "rich_text": [{"type": "text", "text": {"content": text}}]}})
        # divider
        elif line.strip() == "---":
            blocks.append({"type": "divider", "divider": {}})
        # paragraph
        elif line.strip():
            content = line[:2000]  # Notion limit
            blocks.append({"type": "paragraph", "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": content}}]}})
        i += 1
    return blocks

def get_all_wiki_pages(notion: Client, root_id: str) -> list[dict]:
    """Get all child pages (categories) and their sub-pages."""
    pages = []
    children = notion.blocks.children.list(block_id=root_id)
    for block in children["results"]:
        if block["type"] == "child_page":
            cat_title = block["child_page"]["title"]
            cat_id = block["id"]
            pages.append({"id": cat_id, "title": cat_title, "parent": "root"})
            sub = notion.blocks.children.list(block_id=cat_id)
            for sb in sub["results"]:
                if sb["type"] == "child_page":
                    pages.append({
                        "id": sb["id"],
                        "title": sb["child_page"]["title"],
                        "parent": cat_title,
                    })
    return pages

# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert knowledge base maintainer. You help build a structured,
    interlinked personal wiki following the Karpathy LLM Wiki pattern.
    
    Rules:
    - Write clear, concise markdown
    - Use ## and ### headings for structure
    - Cross-reference related topics with [[Topic Name]] notation
    - Note contradictions or open questions explicitly
    - Include a "Key Takeaways" section at the top of each page
    - Include a "Related Topics" section at the bottom
    - Be factual and cite sources when available
    - Use bullet points for lists of facts
    - Keep each section focused and scannable
""")

def ensure_wiki_structure(notion: Client, root_id: str) -> dict[str, str]:
    """Ensure the wiki has its core category pages. Returns {name: page_id}."""
    categories = ["📋 Index", "📝 Log", "📂 Sources", "📂 Concepts", "📂 Entities", "📂 Syntheses"]
    cat_ids = {}
    for cat in categories:
        pid = find_child_page(notion, root_id, cat)
        if not pid:
            pid = create_page(notion, root_id, cat, f"# {cat}\n\nAuto-generated by LLM Wiki.")
        cat_ids[cat] = pid
    return cat_ids

def cmd_status(args):
    """Show wiki stats."""
    notion = get_notion()
    root_id = get_root_id()
    pages = get_all_wiki_pages(notion, root_id)
    cats = [p for p in pages if p["parent"] == "root"]
    subs = [p for p in pages if p["parent"] != "root"]
    print(f"Wiki root: {root_id}")
    print(f"Categories: {len(cats)}")
    print(f"Sub-pages:  {len(subs)}")
    for c in cats:
        count = len([s for s in subs if s["parent"] == c["title"]])
        print(f"  {c['title']}: {count} pages")

def cmd_ingest(args):
    """Ingest a source file or URL into the wiki."""
    source = args.source
    notion = get_notion()
    root_id = get_root_id()
    cat_ids = ensure_wiki_structure(notion, root_id)

    # Read source content
    if source.startswith("http://") or source.startswith("https://"):
        print(f"Fetching URL: {source}")
        import requests
        resp = requests.get(source, timeout=30, headers={"User-Agent": "LLM-Wiki/1.0"})
        resp.raise_for_status()
        # Simple HTML to text
        from html.parser import HTMLParser
        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text = []
            def handle_data(self, data):
                self.text.append(data)
        parser = TextExtractor()
        parser.feed(resp.text)
        raw_text = "\n".join(parser.text)[:15000]
        source_name = source
    else:
        path = Path(source)
        if not path.exists():
            raise SystemExit(f"File not found: {source}")
        raw_text = path.read_text(encoding="utf-8", errors="ignore")[:15000]
        source_name = path.name

    print(f"Source: {source_name} ({len(raw_text)} chars)")
    print("Asking LLM to analyze and generate wiki pages...")

    # Step 1: LLM generates structured summary + topic pages
    prompt = textwrap.dedent(f"""\
        I'm ingesting a new source into my personal knowledge wiki.

        SOURCE: {source_name}
        CONTENT:
        {raw_text[:12000]}

        Please produce the following in valid JSON format:
        {{
            "source_summary": {{
                "title": "short title for this source",
                "markdown": "full markdown summary page with Key Takeaways, Main Content, Related Topics"
            }},
            "concepts": [
                {{
                    "title": "Concept Name",
                    "markdown": "markdown page for this concept with Key Takeaways, content, Related Topics"
                }}
            ],
            "entities": [
                {{
                    "title": "Person/Company/Product Name",
                    "markdown": "markdown page for this entity"
                }}
            ],
            "index_entry": "one-line description for the index"
        }}

        Extract 2-5 key concepts and 1-3 key entities. Be thorough but concise.
        Cross-reference between pages using [[Topic Name]] notation.
        Output ONLY the JSON, no other text.
    """)

    response = llm_call(prompt, system=SYSTEM_PROMPT, max_tokens=4000)

    # Parse JSON from response
    try:
        # Try to find JSON in the response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            raise ValueError("No JSON found")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"LLM response parsing error: {e}")
        print("Raw response (first 500 chars):")
        print(response[:500])
        print("\nFalling back to simple summary...")
        data = {
            "source_summary": {"title": source_name, "markdown": response[:3000]},
            "concepts": [],
            "entities": [],
            "index_entry": source_name,
        }

    # Step 2: Create pages in Notion
    # Source summary
    src = data["source_summary"]
    src_title = src["title"]
    print(f"  Creating source page: {src_title}")
    create_page(notion, cat_ids["📂 Sources"], src_title, src["markdown"])

    # Concepts
    for concept in data.get("concepts", []):
        title = concept["title"]
        existing = find_child_page(notion, cat_ids["📂 Concepts"], title)
        if existing:
            print(f"  Updating concept: {title}")
            append_to_page(notion, existing, f"\n---\n## Update from: {src_title}\n\n{concept['markdown']}")
        else:
            print(f"  Creating concept: {title}")
            create_page(notion, cat_ids["📂 Concepts"], title, concept["markdown"])

    # Entities
    for entity in data.get("entities", []):
        title = entity["title"]
        existing = find_child_page(notion, cat_ids["📂 Entities"], title)
        if existing:
            print(f"  Updating entity: {title}")
            append_to_page(notion, existing, f"\n---\n## Update from: {src_title}\n\n{entity['markdown']}")
        else:
            print(f"  Creating entity: {title}")
            create_page(notion, cat_ids["📂 Entities"], title, entity["markdown"])

    # Update index
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    index_entry = data.get("index_entry", src_title)
    append_to_page(notion, cat_ids["📋 Index"],
                   f"- [{now}] **{src_title}** — {index_entry}")

    # Update log
    n_concepts = len(data.get("concepts", []))
    n_entities = len(data.get("entities", []))
    append_to_page(notion, cat_ids["📝 Log"],
                   f"## [{now}] ingest | {src_title}\n\n"
                   f"Source: {source_name}\n"
                   f"Created: 1 source page, {n_concepts} concepts, {n_entities} entities")

    print(f"\n✅ Ingested! Created {1 + n_concepts + n_entities} pages.")

def cmd_query(args):
    """Query the wiki."""
    question = args.question
    notion = get_notion()
    root_id = get_root_id()
    cat_ids = ensure_wiki_structure(notion, root_id)

    # Read index to find relevant pages
    index_text = read_page_text(notion, cat_ids["📋 Index"])

    # Read all page titles for context
    all_pages = get_all_wiki_pages(notion, root_id)
    page_list = "\n".join(f"- [{p['parent']}] {p['title']}" for p in all_pages if p["parent"] != "root")

    # Ask LLM which pages to read
    pick_prompt = f"""Given this wiki index and page list, which pages are most relevant to answer: "{question}"?

INDEX:
{index_text[:3000]}

ALL PAGES:
{page_list[:3000]}

Return a JSON list of page titles to read (max 5): ["Page Title 1", "Page Title 2", ...]
Output ONLY the JSON array."""

    pick_response = llm_call(pick_prompt, max_tokens=500)
    try:
        json_match = re.search(r"\[.*\]", pick_response, re.DOTALL)
        relevant_titles = json.loads(json_match.group()) if json_match else []
    except (json.JSONDecodeError, ValueError):
        relevant_titles = []

    # Read relevant pages
    context_parts = []
    for page in all_pages:
        if page["title"] in relevant_titles:
            try:
                text = read_page_text(notion, page["id"])
                context_parts.append(f"## {page['title']}\n{text}")
                print(f"  Reading: {page['title']}")
            except Exception:
                pass

    context = "\n\n---\n\n".join(context_parts) if context_parts else "No specific pages found."

    # Generate answer
    answer_prompt = f"""Based on the following wiki pages, answer this question: "{question}"

WIKI CONTEXT:
{context[:10000]}

Provide a thorough answer with references to the wiki pages using [[Page Name]] notation.
If the wiki doesn't contain enough information, say so and suggest what to ingest."""

    print(f"\n💬 Question: {question}\n")
    answer = llm_call(answer_prompt, system=SYSTEM_PROMPT, max_tokens=2000)
    print(answer)

    # Optionally save to Syntheses
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    append_to_page(notion, cat_ids["📝 Log"],
                   f"## [{now}] query | {question[:80]}\n\nAnswer generated from {len(context_parts)} pages.")
    print(f"\n📝 Logged to wiki.")

def cmd_lint(args):
    """Health-check the wiki."""
    notion = get_notion()
    root_id = get_root_id()
    cat_ids = ensure_wiki_structure(notion, root_id)

    all_pages = get_all_wiki_pages(notion, root_id)
    page_list = "\n".join(f"- [{p['parent']}] {p['title']}" for p in all_pages if p["parent"] != "root")

    # Read a sample of pages for deeper analysis
    sample_text = ""
    sub_pages = [p for p in all_pages if p["parent"] != "root"]
    for page in sub_pages[:10]:
        try:
            text = read_page_text(notion, page["id"])
            sample_text += f"\n## {page['title']} ({page['parent']})\n{text[:500]}\n"
        except Exception:
            pass

    lint_prompt = f"""You are auditing a personal knowledge wiki. Review the structure and content below.

PAGE LIST:
{page_list[:3000]}

SAMPLE CONTENT:
{sample_text[:6000]}

Report:
1. **Orphan pages** — pages with no cross-references
2. **Missing pages** — topics mentioned with [[...]] but no page exists
3. **Contradictions** — conflicting claims between pages
4. **Gaps** — important topics that should have pages but don't
5. **Suggestions** — new sources to ingest, questions to explore

Be specific and actionable."""

    print("🔍 Running wiki health check...\n")
    result = llm_call(lint_prompt, system=SYSTEM_PROMPT, max_tokens=2000)
    print(result)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    append_to_page(notion, cat_ids["📝 Log"],
                   f"## [{now}] lint | Wiki health check\n\n{len(sub_pages)} pages reviewed.")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LLM Wiki — Notion-based personal knowledge base")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Show wiki stats")

    p_ingest = sub.add_parser("ingest", help="Ingest a source file or URL")
    p_ingest.add_argument("source", help="File path or URL to ingest")

    p_query = sub.add_parser("query", help="Ask the wiki a question")
    p_query.add_argument("question", help="Your question")

    sub.add_parser("lint", help="Health-check the wiki")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    {"status": cmd_status, "ingest": cmd_ingest, "query": cmd_query, "lint": cmd_lint}[args.command](args)

if __name__ == "__main__":
    main()
