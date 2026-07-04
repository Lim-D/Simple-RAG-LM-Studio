from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from rag_utils import get_lmstudio_client


def main() -> int:
    load_dotenv()
    base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    print(f"Checking LM Studio at {base_url} ...")

    try:
        client = get_lmstudio_client()
        models = client.models.list().data
    except Exception as exc:
        print("\nERROR: Could not reach the LM Studio API server.")
        print("Open LM Studio > Developer, load your model(s), and turn on Start server.")
        print(f"Details: {exc}")
        return 1

    if not models:
        print("Connected, but no models were returned. Load a chat and embedding model in LM Studio.")
        return 1

    print("\nAvailable model identifiers:")
    for model in models:
        print(f"  - {model.id}")

    chat_model = os.getenv("CHAT_MODEL", "")
    embed_model = os.getenv("EMBED_MODEL", "")
    model_ids = {m.id for m in models}

    print("\n.env status:")
    print(f"  CHAT_MODEL={chat_model or '<not set>'}")
    print(f"  EMBED_MODEL={embed_model or '<not set>'}")

    if chat_model and chat_model not in model_ids:
        print("  WARNING: CHAT_MODEL is not in the model list above.")
    if embed_model and embed_model not in model_ids:
        print("  WARNING: EMBED_MODEL is not in the model list above.")

    if embed_model and embed_model in model_ids:
        try:
            vector = client.embeddings.create(model=embed_model, input=["RAG diagnostic test"]).data[0].embedding
            print(f"\nEmbedding test succeeded: {len(vector)} dimensions.")
        except Exception as exc:
            print(f"\nEmbedding test failed: {exc}")
            return 1

    print("\nLM Studio connection check complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
