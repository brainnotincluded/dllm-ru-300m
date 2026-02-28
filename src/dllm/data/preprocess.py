"""Data preprocessing utilities."""
import re
import unicodedata
from pathlib import Path
from typing import Iterator, List
import json


def clean_text(text: str) -> str:
    """Clean raw text."""
    # Normalize unicode
    text = unicodedata.normalize("NFC", text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters except newlines
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char == '\n')
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    return text.strip()


def deduplicate_lines(lines: List[str]) -> List[str]:
    """Simple exact deduplication."""
    seen = set()
    result = []
    for line in lines:
        if line not in seen and len(line) > 10:  # Skip very short lines
            seen.add(line)
            result.append(line)
    return result


def detect_language(text: str) -> str:
    """Simple language detection based on character frequency."""
    # Count Russian vs English characters
    ru_chars = len(re.findall(r'[а-яА-Я]', text))
    en_chars = len(re.findall(r'[a-zA-Z]', text))
    
    if ru_chars > en_chars * 2:
        return "ru"
    elif en_chars > ru_chars * 2:
        return "en"
    else:
        return "mixed"


def process_file(input_path: Path, output_path: Path, min_length: int = 10) -> int:
    """Process a single file."""
    lines = []
    
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    # Split into documents (assuming double newline separation)
    documents = content.split("\n\n")
    
    for doc in documents:
        doc = clean_text(doc)
        if len(doc) >= min_length:
            lang = detect_language(doc)
            lines.append({
                "text": doc,
                "language": lang,
                "length": len(doc)
            })
    
    # Deduplicate
    seen_texts = set()
    unique_lines = []
    for line in lines:
        if line["text"] not in seen_texts:
            seen_texts.add(line["text"])
            unique_lines.append(line)
    
    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for line in unique_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    
    return len(unique_lines)


def preprocess_directory(input_dir: str, output_dir: str) -> dict:
    """Preprocess all files in directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    stats = {"total_documents": 0, "files_processed": 0}
    
    for file_path in input_path.rglob("*.txt"):
        relative_path = file_path.relative_to(input_path)
        output_file = output_path / f"{relative_path.stem}.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        count = process_file(file_path, output_file)
        stats["total_documents"] += count
        stats["files_processed"] += 1
        
        print(f"Processed {file_path}: {count} documents")
    
    return stats
