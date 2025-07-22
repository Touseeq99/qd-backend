import re
from docx import Document
from typing import List, Dict, Optional

def extract_docx_text(file_path: str) -> List[str]:
    """Extract non-empty paragraphs from the .docx file."""
    doc = Document(file_path)
    return [para.text.strip() for para in doc.paragraphs if para.text.strip()]

def is_clause_line(text: str) -> Optional[str]:
    """
    Match only lines that truly start with a clause pattern like '6.1', '6.2.3'
    and extract the top-level clause number like '6', '7', etc.
    """
    match = re.match(r"^(\d{1,2})(\.\d+)+\s", text)  # must be at least like '6.1 '
    return match.group(1) if match else None

def is_heading_line(text: str) -> bool:
    """
    Basic heuristic for detecting non-numbered headings.
    Accepts short, capitalized lines with no clause numbering.
    """
    return (
        not is_clause_line(text)
        and len(text.split()) <= 6
        and text[0].isupper()
        and text == text.title()
    )

def chunk_by_clause_or_heading(paragraphs: List[str]) -> List[Dict[str, str]]:
    """
    Hybrid chunker:
    - If numbered clauses (1., 2.1, etc.) exist: chunks by top-level number
    - If no clauses: falls back to heading-based chunking
    """
    chunks = []
    current_chunk = []
    current_title = None
    clause_found = False

    for para in paragraphs:
        clause = is_clause_line(para)
        heading = is_heading_line(para)

        if clause:
            clause_found = True
            if clause != current_title:
                if current_chunk:
                    chunks.append({
                        "clause": current_title or "Preamble",
                        "content": "\n".join(current_chunk).strip()
                    })
                    current_chunk = []
                current_title = clause
        elif heading and not clause_found:
            # Use headings as chunks only if no clauses are detected at all
            if current_chunk:
                chunks.append({
                    "clause": current_title or "Section",
                    "content": "\n".join(current_chunk).strip()
                })
                current_chunk = []
            current_title = para

        current_chunk.append(para)

    if current_chunk:
        chunks.append({
            "clause": current_title or "Final",
            "content": "\n".join(current_chunk).strip()
        })

    return chunks

def inspect_chunks(file_path: str):
    print(f"\n📄 Processing: {file_path}")
    paragraphs = extract_docx_text(file_path)
    chunks = chunk_by_clause_or_heading(paragraphs)
    print(f"\n✅ Total chunks: {len(chunks)}\n")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1}: {chunk['clause']} ---\n{chunk['content'][:500]}...\n{'='*60}")

# Replace the file path with your actual file
# Example:
# inspect_chunks(r"C:\Users\user\Desktop\QD_HR_ASSISTANT\3. Attendance Policy 3.0.docx")
# or
inspect_chunks(r"C:\Users\user\Desktop\QD_HR_ASSISTANT\9. QG Code of Conduct.docx")
