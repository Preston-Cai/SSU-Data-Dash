"""Prepare reusable frozen remark vectors before starting walk-forward validation."""
import argparse
import time

import data
import semantic_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Encoder inference batch size (reduce if GPU memory is limited).")
    parser.add_argument("--device", default=None,
                        help="Torch device, e.g. cpu, cuda or cuda:0; defaults to auto detection.")
    parser.add_argument("--since", default="2025-06-01",
                        help="Encode sales from this date (default: first validation inner fold).")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    started = time.monotonic()
    listings = data.load("listings")
    if "remarks" not in listings:
        raise ValueError("The listings training file has no remarks column.")
    import pandas as pd
    listings = listings.loc[listings.date_sold >= pd.Timestamp(args.since)]
    texts = listings["remarks"].fillna("").astype(str).drop_duplicates().tolist()
    del listings
    print(f"Preparing {len(texts):,} unique training remarks. Model: {semantic_text.MODEL_ID}",
          flush=True)
    print(f"Cache: {semantic_text.cache_path()}", flush=True)
    try:
        # Avoid materializing the whole training corpus's embedding matrix at once.
        for start in range(0, len(texts), 2048):
            block = texts[start:start + 2048]
            semantic_text.get_embeddings(block, batch_size=args.batch_size, device=args.device)
            print(f"Prepared {min(start + len(block), len(texts)):,} / {len(texts):,} remarks "
                  f"({(time.monotonic() - started) / 60:.1f} min)", flush=True)
    finally:
        semantic_text.release_encoder()
    print("Semantic cache is ready. No price model was trained or validated.", flush=True)


if __name__ == "__main__":
    main()
