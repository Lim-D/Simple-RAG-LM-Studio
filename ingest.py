from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv

from rag_utils import (
    chunk_text,
    embed_texts,
    get_lmstudio_client,
    iter_supported_files,
    load_text_units,
    stable_chunk_id,
    prefixed_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index local documents for RAG.")
    parser.add_argument("documents_dir", nargs="?", default="documents", help="Folder containing documents")
    parser.add_argument("--rebuild", action="store_true", help="Delete and rebuild the collection")
    parser.add_argument("--chunk-size", type=int, default=1800, help="Maximum characters per chunk")
    parser.add_argument("--overlap", type=int, default=250, help="Overlapping characters between chunks")
    return parser.parse_args()


def main() -> int:
    project_dir = Path(__file__).resolve().parent
    load_dotenv(project_dir / ".env")
    args = parse_args()

    documents_dir = Path(args.documents_dir)
    if not documents_dir.is_absolute():
        documents_dir = project_dir / documents_dir
    documents_dir = documents_dir.resolve()
    if not documents_dir.exists() or not documents_dir.is_dir():
        print(f"ERROR: Document folder not found: {documents_dir}")
        return 1

    embed_model = os.getenv("EMBED_MODEL", "").strip()
    if not embed_model or embed_model.startswith("replace-"):
        print("ERROR: Set EMBED_MODEL in .env to the exact LM Studio embedding model identifier.")
        return 1

    db_path_value = os.getenv("CHROMA_PATH", ".rag_db").strip() or ".rag_db"
    db_path = Path(db_path_value)
    if not db_path.is_absolute():
        db_path = project_dir / db_path
    db_path = db_path.resolve()
    collection_name = os.getenv("COLLECTION_NAME", "local_documents")
    batch_size = int(os.getenv("EMBED_BATCH_SIZE", "24"))
    document_prefix = os.getenv("EMBED_DOCUMENT_PREFIX", "search_document:").strip()

    print(f"Project folder: {project_dir}")
    print(f"Documents folder: {documents_dir}")
    print(f"Chroma database: {db_path}")

    chroma_client = chromadb.PersistentClient(path=str(db_path))
    if args.rebuild:
        try:
            chroma_client.delete_collection(collection_name)
            print(f"Deleted old collection: {collection_name}")
        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=None,
        metadata={"description": "Local document chunks embedded by LM Studio"},
        configuration={"hnsw": {"space": "cosine"}},
    )
    lm_client = get_lmstudio_client()

    files = list(iter_supported_files(documents_dir))
    if not files:
        print("No supported documents found. Add PDF, DOCX, TXT, MD, HTML, or HTM files.")
        return 1

    total_chunks = 0
    empty_pdfs: list[str] = []

    for file_number, path in enumerate(files, start=1):
        relative_source = path.relative_to(documents_dir).as_posix()
        print(f"[{file_number}/{len(files)}] Reading {relative_source}")
        try:
            units = load_text_units(path)
        except Exception as exc:
            print(f"  SKIPPED: {exc}")
            continue

        if not units:
            if path.suffix.lower() == ".pdf":
                empty_pdfs.append(relative_source)
            print("  SKIPPED: no extractable text")
            continue

        records: list[tuple[str, str, dict]] = []
        file_chunk_index = 0
        for unit in units:
            for text in chunk_text(unit.text, max_chars=args.chunk_size, overlap_chars=args.overlap):
                metadata = {
                    "source": relative_source,
                    "filename": path.name,
                    "extension": path.suffix.lower(),
                    "page": unit.page if unit.page is not None else -1,
                    "chunk": file_chunk_index,
                }
                chunk_id = stable_chunk_id(relative_source, unit.page, file_chunk_index, text)
                records.append((chunk_id, text, metadata))
                file_chunk_index += 1

        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            texts = [item[1] for item in batch]
            try:
                embedding_inputs = [prefixed_text(text, document_prefix) for text in texts]
                embeddings = embed_texts(lm_client, embed_model, embedding_inputs, batch_size=batch_size)
                collection.upsert(
                    ids=[item[0] for item in batch],
                    documents=texts,
                    metadatas=[item[2] for item in batch],
                    embeddings=embeddings,
                )
            except Exception as exc:
                print(f"\nERROR while embedding/indexing {relative_source}: {exc}")
                print("Check that LM Studio is running and EMBED_MODEL matches a loaded embedding model.")
                return 1

        total_chunks += len(records)
        print(f"  Indexed {len(records)} chunks")

    print(f"\nDone. Indexed {total_chunks} chunks into '{collection_name}'.")
    print(f"Database folder: {db_path}")
    print(f"Collection contains: {collection.count()} chunks")
    if empty_pdfs:
        print("\nThese PDFs had no extractable text and may be scanned images requiring OCR:")
        for name in empty_pdfs:
            print(f"  - {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
