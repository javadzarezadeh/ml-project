import argparse
import ast
import sys

import pandas as pd
import torch
from transformers import pipeline


ASPECTS = (
    "acting",
    "plot",
    "characters",
    "direction",
    "cinematography",
    "music",
    "visual effects",
    "pacing",
    "dialogue",
    "ending",
)


def parse_reviews(reviews_row):
    if isinstance(reviews_row, str):
        try:
            reviews_row = ast.literal_eval(reviews_row)
        except (ValueError, SyntaxError):
            return []

    if not isinstance(reviews_row, list):
        return []

    return [
        review for review in reviews_row
        if isinstance(review, str) and review.strip()
    ]


def classify_dataset(classifier, reviews):
    parsed_reviews = [parse_reviews(row) for row in reviews]
    flattened_reviews = [
        review
        for row in parsed_reviews
        for review in row
    ]
    annotations = []

    for start in range(0, len(flattened_reviews), 256):
        review_batch = flattened_reviews[start:start + 256]
        predictions = classifier(
            [
                {
                    "text": review,
                    "text_pair": aspect,
                }
                for review in review_batch
                for aspect in ASPECTS
            ],
            truncation=True,
            max_length=512,
            batch_size=16
        )

        annotations.extend([
            [
                {
                    "aspect": aspect,
                    "label": predictions[
                        index * len(ASPECTS) + aspect_index
                    ]["label"].lower(),
                    "score": predictions[
                        index * len(ASPECTS) + aspect_index
                    ]["score"],
                }
                for aspect_index, aspect in enumerate(ASPECTS)
            ]
            for index in range(len(review_batch))
        ])

    rows = []
    position = 0
    for row in parsed_reviews:
        rows.append(annotations[position:position + len(row)])
        position += len(row)

    return rows


def main(input_path, output_path):
    print(f"Loading dataset from {input_path}...")
    df = pd.read_pickle(input_path) if input_path.endswith(".pkl") else pd.read_csv(input_path)

    if "Reviews" not in df.columns:
        print("Error: Input dataset must contain a 'Reviews' column.")
        sys.exit(1)

    print("Loading aspect-based sentiment model...")
    classifier = pipeline(
        task="text-classification",
        model="yangheng/deberta-v3-base-absa-v1.1",
        dtype=torch.float16,
        device="mps" if torch.backends.mps.is_available() else "cpu"
    )

    df["review_aspect_sentiments"] = classify_dataset(
        classifier,
        df["Reviews"]
    )

    df.to_pickle(output_path) if output_path.endswith(".pkl") else df.to_csv(
        output_path,
        index=False
    )
    print(f"Successfully saved aspect sentiment features to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classifies review aspects as positive, neutral, or negative."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.input, args.output)
