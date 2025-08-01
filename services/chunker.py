
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 150) -> list[str]:
    words = text.split()
    chunks = []

    for start in range(0, len(words), chunk_size - overlap):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

    return chunks
