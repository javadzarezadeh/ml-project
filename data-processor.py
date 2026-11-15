import argparse
import ast
import numpy as np
import pandas as pd

import ast

def transform_emotions(emotions):
    if isinstance(emotions, str):
        emotions = ast.literal_eval(emotions)

    if not isinstance(emotions, list):
        return {}

    return (
        pd.DataFrame(emotions)
        .groupby('label')['score'].max()
        .to_dict()
    )

def main(input_path: str, output_x: str, output_y: str):
    """Cleans tabular dataset, encodes categorical metadata, and prepares feature/target pickles for LightGBM."""

    # Accept both .csv and .pkl as input
    movies_dataset = pd.read_csv(input_path)

    # Drop the "Reviews "column, if it exists
    movies_dataset = movies_dataset.drop(columns=['Reviews'], errors='ignore')
    
    # Drop revenue & other review/rating indicators from the feature dataset, 
    movies_dataset_x = movies_dataset.drop(
        columns=["Rank", "Release Group", "$Worldwide", "$Domestic", "$Foreign", "Foreign %", "Domestic %"], 
        errors="ignore"
    )

    movies_dataset_x["Rating"] = movies_dataset_x["Rating"].apply(lambda rating: float(str(rating).split("/")[0]))

    if 'review_emotions' in movies_dataset_x.columns:
        emotion_features = (
            movies_dataset_x['review_emotions']
            .apply(transform_emotions)
            .apply(pd.Series)
            .fillna(0)
            .rename(columns={'neutral': 'neutral_emotion'})
        )

        movies_dataset_x = pd.concat([
            movies_dataset_x.drop(columns=['review_emotions']),
            emotion_features
        ], axis=1)

    if 'review_sentiments' in movies_dataset_x.columns:
        emotion_features = (
            movies_dataset_x['review_sentiments']
            .apply(transform_emotions)
            .apply(pd.Series)
            .fillna(0)
            .rename(columns={'neutral': 'neutral_sentiment'})
        )

        movies_dataset_x = pd.concat([
            movies_dataset_x.drop(columns=['review_sentiments']),
            emotion_features
        ], axis=1) 
    
    # We want to predict domestic and foreign income
    movies_dataset_y = movies_dataset[["$Domestic", "$Foreign", "$Worldwide"]].copy()
    # Rename columns to remove the '$' sign
    movies_dataset_y = movies_dataset_y.rename(columns={r"$Domestic": "Domestic", r"$Foreign": "Foreign", r"$Worldwide": "Worldwide"})

    # Turn original language feature values into numerical categories
    movies_dataset_x['Original_Language'] = (
        movies_dataset_x['Original_Language']
        .astype('category')
        .cat.codes
    )
        
    # Turn genres and production countries entries into multi-hot encoded entries
    multi_val_cols = ['Genres', 'Production_Countries']
    for col in multi_val_cols:
        if col in movies_dataset_x.columns:
            types = []
            for entry in movies_dataset_x[col].unique(): 
                if isinstance(entry, str):
                    entry_types = entry.split(", ")
                    for t in entry_types:
                        if t not in types:
                            types.append(t)
            types_enum = {val: i for i, val in enumerate(types)} 
            new_col = []
            for entry in movies_dataset_x[col]:
                multihot_vector = [0] * len(types)
                if isinstance(entry, str):
                    entry_types = entry.split(", ")
                    for t in entry_types:
                        if t in types_enum:
                            multihot_vector[types_enum[t]] = 1
                new_col.append(multihot_vector)
            movies_dataset_x[col] = new_col   
    # Then use those to expand them into different columns, one for each category
    for col in ['Genres', 'Production_Countries']:
        if col in movies_dataset_x.columns:
            expanded = pd.DataFrame(
                movies_dataset_x[col].tolist(),
                index=movies_dataset_x.index
            )

            expanded.columns = [f'{col}_{i}' for i in expanded.columns]

            movies_dataset_x = pd.concat([
                movies_dataset_x.drop(columns=[col]),
                expanded
            ], axis=1) 

    # Save the final feature and target datasets as .pkl files using command-line arguments
    print(movies_dataset_x.iloc[0]["Rating"])
    print(movies_dataset_y.iloc[0])
    movies_dataset_x.to_pickle(output_x)
    movies_dataset_y.to_pickle(output_y)
    print("\nProcessed Features X columns:", movies_dataset_x.columns.tolist())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process structural metadata into numeric and multi-hot features.")
    parser.add_argument("--input", required=True, help="Path to input movies dataset CSV or PKL")
    parser.add_argument("--output_x", required=True, help="Path to save the processed features X pickle file")
    parser.add_argument("--output_y", required=True, help="Path to save the target labels Y pickle file")
    
    args = parser.parse_args()
    main(input_path=args.input, output_x=args.output_x, output_y=args.output_y)