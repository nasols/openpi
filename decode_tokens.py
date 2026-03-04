"""
Utility to decode token IDs to readable text.

Usage:
    python decode_tokens.py <token_ids>

Example:
    python decode_tokens.py "2 3874 503 736 15742"
"""

import sys
import sentencepiece
from openpi.shared import download

def decode_tokens(token_str: str) -> str:
    """Decode a space-separated string of token IDs to text."""
    # Parse token IDs
    token_ids = [int(t) for t in token_str.strip("[]").replace(",", " ").split() if t.strip() and t.strip().isdigit() and int(t) > 0]
    
    # Load tokenizer
    path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    with path.open("rb") as f:
        tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
    
    # Decode
    text = tokenizer.decode(token_ids)
    return text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python decode_tokens.py '<token_ids>'")
        print("Example: python decode_tokens.py '2 3874 503 736 15742'")
        sys.exit(1)
    
    token_str = " ".join(sys.argv[1:])
    decoded = decode_tokens(token_str)
    print(f"\nDecoded text: '{decoded}'\n")
