import argparse
import ast
import re
import numpy as np
import pandas as pd


def clean_category(value):
    """Clean and properly capitalize category names."""

    if pd.isna(value):
        return "Missing"

    value = re.sub(r'\s+', ' ', str(value).strip())

    if not value:
        return "Missing"

    return value.title()


def make_feature_name(prefix, value):
    """Create readable feature names from category values."""

    value = re.sub(
        r'[^A-Za-z0-9]+',
        '_',
        clean_category(value)
    ).strip('_')

    return f"{prefix}_{value}"


def parse_annotations(annotations):
    if isinstance(annotations, str):
        try:
            annotations = ast.literal_eval(annotations)
        except (ValueError, SyntaxError):
            return []

    return annotations if isinstance(annotations, list) else []


def validate_annotation_alignment(dataset):
    if "Reviews" not in dataset.columns:
        return

    for column in [
        "review_sentiments",
        "review_emotions",
        "review_aspect_sentiments"
    ]:
        if column not in dataset.columns:
            continue

        for index, (reviews, annotations) in enumerate(
            zip(dataset["Reviews"], dataset[column])
        ):
            parsed_reviews = parse_annotations(reviews)
            parsed_annotations = parse_annotations(annotations)

            if len(parsed_reviews) != len(parsed_annotations):
                raise ValueError(
                    f"{column} is not aligned with Reviews at row {index}: "
                    f"{len(parsed_reviews)} reviews, "
                    f"{len(parsed_annotations)} annotations. "
                    "Regenerate this dataset with the analyzer scripts."
                )


def annotation_labels(values):
    labels = set()

    for annotations in values:
        for annotation in parse_annotations(annotations):
            if not isinstance(annotation, dict):
                continue

            label = clean_category(annotation.get('label'))
            score = annotation.get('score')

            if score is None:
                continue
            try:
                float(score)
            except (ValueError, TypeError):
                continue

            labels.add(label)

    return sorted(labels)


def aspect_annotation_labels(values):
    labels = set()

    for annotations in values:
        for annotation_group in parse_annotations(annotations):
            annotation_values = (
                annotation_group
                if isinstance(annotation_group, list)
                else [annotation_group]
            )
            for annotation in annotation_values:
                if not isinstance(annotation, dict):
                    continue

                aspect = clean_category(annotation.get('aspect'))
                label = clean_category(annotation.get('label'))
                score = annotation.get('score')

                if score is None:
                    continue
                try:
                    float(score)
                except (ValueError, TypeError):
                    continue

                labels.add(f"{aspect}_{label}")

    return sorted(labels)


def flatten_aspect_annotations(annotations):
    flattened = []
    for annotation_group in parse_annotations(annotations):
        if isinstance(annotation_group, list):
            flattened.extend(annotation_group)
        else:
            flattened.append(annotation_group)
    return flattened


def transform_emotions(emotions, labels=None):
    emotions = parse_annotations(emotions)

    if labels is None:
        labels = annotation_labels([emotions])

    scores = {label: [] for label in labels}
    analyzed_count = 0

    for emotion in emotions:
        if not isinstance(emotion, dict):
            continue

        label = clean_category(emotion.get('label'))
        score = emotion.get('score')
        if score is None:
            continue

        try:
            score = float(score)
        except (ValueError, TypeError):
            continue

        analyzed_count += 1
        if label in scores:
            scores[label].append(score)

    features = {"Analyzed_Review_Count": float(analyzed_count)}

    for label, values in scores.items():
        count = len(values)
        features[f"{label}_Mean"] = (
            float(np.mean(values)) if count else 0.0
        )
        features[f"{label}_Max"] = float(max(values)) if count else 0.0
        features[f"{label}_Fraction"] = (
            count / analyzed_count if analyzed_count else 0
        )

    return features


def transform_aspect_sentiments(annotations, labels=None):
    annotations = flatten_aspect_annotations(annotations)

    if labels is None:
        labels = aspect_annotation_labels([annotations])

    scores = {label: [] for label in labels}
    analyzed_count = 0

    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue

        aspect = clean_category(annotation.get('aspect'))
        label = clean_category(annotation.get('label'))
        score = annotation.get('score')
        if score is None:
            continue

        try:
            score = float(score)
        except (ValueError, TypeError):
            continue

        analyzed_count += 1
        annotation_label = f"{aspect}_{label}"
        if annotation_label in scores:
            scores[annotation_label].append(score)

    features = {"Analyzed_Aspect_Sentiment_Count": float(analyzed_count)}

    for label, values in scores.items():
        count = len(values)
        features[f"{label}_Mean"] = (
            float(np.mean(values)) if count else 0.0
        )
        features[f"{label}_Max"] = float(max(values)) if count else 0.0
        features[f"{label}_Fraction"] = (
            count / analyzed_count if analyzed_count else 0
        )

    return features


def remove_duplicate_input_columns(dataset):
    """Remove duplicate columns only when their contents are identical."""

    duplicate_names = dataset.columns[
        dataset.columns.duplicated(keep=False)
    ].unique()

    for name in duplicate_names:
        positions = [
            i for i, col in enumerate(dataset.columns)
            if col == name
        ]

        first_column = dataset.iloc[:, positions[0]]

        for position in positions[1:]:
            other_column = dataset.iloc[:, position]

            if not first_column.equals(other_column):
                raise ValueError(
                    f"Duplicate input column {name!r} contains conflicting data. "
                    f"Occurrences at positions {positions}. "
                    "Inspect the input CSV before processing."
                )

        print(
            f"Duplicate input column '{name}' found "
            f"{len(positions)} times. Identical copies detected; keeping one."
        )

    return dataset.loc[:, ~dataset.columns.duplicated()].copy()


def process_dataframe(
    movies_dataset,
    emotion_labels=None,
    sentiment_labels=None,
    category_schema=None
):
    """Clean and encode one dataset split using the supplied label schema."""

    movies_dataset = remove_duplicate_input_columns(movies_dataset)
    validate_annotation_alignment(movies_dataset)

    if emotion_labels is None and 'review_emotions' in movies_dataset:
        emotion_labels = annotation_labels(movies_dataset['review_emotions'])

    if sentiment_labels is None and 'review_sentiments' in movies_dataset:
        sentiment_labels = annotation_labels(movies_dataset['review_sentiments'])

    emotion_labels = emotion_labels or []
    sentiment_labels = sentiment_labels or []
    category_schema = {} if category_schema is None else category_schema

    movies_dataset = movies_dataset.drop(columns=['Reviews'], errors='ignore')

    movies_dataset_x = movies_dataset.drop(
        columns=[
            "Rank", "Release Group", "$Worldwide", "$Domestic", "$Foreign",
            "Foreign %", "Domestic %"
        ],
        errors="ignore"
    )

    movies_dataset_y = movies_dataset[[
        "$Domestic", "$Foreign", "$Worldwide"
    ]].copy()
    movies_dataset_y = movies_dataset_y.rename(
        columns={
            "$Domestic": "Domestic",
            "$Foreign": "Foreign",
            "$Worldwide": "Worldwide"
        }
    )

    for col in movies_dataset_y.columns:
        movies_dataset_y[col] = pd.to_numeric(
            movies_dataset_y[col],
            errors='coerce'
        )

    if 'Budget' in movies_dataset_x.columns:
        movies_dataset_x['Budget'] = pd.to_numeric(
            movies_dataset_x['Budget'].astype(str).str.replace(
                r'[$,]',
                '',
                regex=True
            ),
            errors='coerce'
        )

    if 'Rating' in movies_dataset_x.columns:
        movies_dataset_x['Rating'] = pd.to_numeric(
            movies_dataset_x['Rating'].astype(str).str.extract(
                r'([-+]?\d*\.?\d+)'
            )[0],
            errors='coerce'
        )

    if 'Vote_Count' in movies_dataset_x.columns:
        movies_dataset_x['Vote_Count'] = pd.to_numeric(
            movies_dataset_x['Vote_Count'],
            errors='coerce'
        )

    annotation_columns = [
        ('review_emotions', emotion_labels, 'Review_Emotion'),
        ('review_sentiments', sentiment_labels, 'Review_Sentiment')
    ]

    for column, labels, prefix in annotation_columns:
        if column not in movies_dataset_x.columns:
            continue

        features = (
            movies_dataset_x[column]
            .apply(transform_emotions, labels=labels)
            .apply(pd.Series)
            .fillna(0)
        )
        features.columns = [
            make_feature_name(prefix, name)
            for name in features.columns
        ]

        movies_dataset_x = pd.concat([
            movies_dataset_x.drop(columns=[column]),
            features
        ], axis=1)

    if 'review_aspect_sentiments' in movies_dataset_x.columns:
        aspect_labels = aspect_annotation_labels(
            movies_dataset_x['review_aspect_sentiments']
        )
        features = (
            movies_dataset_x['review_aspect_sentiments']
            .apply(transform_aspect_sentiments, labels=aspect_labels)
            .apply(pd.Series)
            .fillna(0)
        )
        features.columns = [
            make_feature_name("Review_Aspect_Sentiment", name)
            for name in features.columns
        ]
        movies_dataset_x = pd.concat([
            movies_dataset_x.drop(columns=['review_aspect_sentiments']),
            features
        ], axis=1)

    if 'Original_Language' in movies_dataset_x.columns:
        languages = movies_dataset_x['Original_Language'].apply(clean_category)
        if 'Original_Language' not in category_schema:
            category_schema['Original_Language'] = sorted(languages.unique())
        languages = pd.Categorical(
            languages,
            categories=category_schema['Original_Language']
        )
        language_features = pd.get_dummies(
            pd.Series(languages, index=movies_dataset_x.index),
            prefix='Original_Language',
            dtype=int
        )
        movies_dataset_x = pd.concat([
            movies_dataset_x.drop(columns=['Original_Language']),
            language_features
        ], axis=1)

    multi_val_cols = {
        'Genres': ('Genre', ', '),
        'Production_Countries': ('Production_Country', ', '),
        'Production_Studios': ('Production_Studio', '; ')
    }

    for col, (prefix, separator) in multi_val_cols.items():
        if col not in movies_dataset_x.columns:
            continue

        if col not in category_schema:
            types = []
            for entry in movies_dataset_x[col].dropna().unique():
                if isinstance(entry, str):
                    for value in entry.split(separator):
                        value = clean_category(value)
                        if value not in types:
                            types.append(value)

            if movies_dataset_x[col].isna().any() and "Missing" not in types:
                types.append("Missing")
            category_schema[col] = sorted(types)
        else:
            types = category_schema[col]

        encoded_features = {}
        for value in types:
            feature_name = make_feature_name(prefix, value)
            encoded_features[feature_name] = movies_dataset_x[col].apply(
                lambda entry: int(
                    value in [
                        clean_category(item)
                        for item in entry.split(separator)
                    ]
                ) if isinstance(entry, str) else int(value == "Missing")
            )

        movies_dataset_x = pd.concat([
            movies_dataset_x.drop(columns=[col]),
            pd.DataFrame(encoded_features, index=movies_dataset_x.index)
        ], axis=1)

    categorical_cols = movies_dataset_x.select_dtypes(
        include=['object', 'category', 'string']
    ).columns.tolist()

    if categorical_cols:
        for column in categorical_cols:
            values = movies_dataset_x[column].map(clean_category)
            if column not in category_schema:
                category_schema[column] = sorted(values.unique())
            movies_dataset_x[column] = pd.Categorical(
                values,
                categories=category_schema[column]
            )
        movies_dataset_x = pd.get_dummies(
            movies_dataset_x,
            columns=categorical_cols,
            dtype=int
        )

    movies_dataset_x.columns = [
        re.sub(
            r'_+',
            '_',
            re.sub(r'[^A-Za-z0-9]+', '_', str(col).strip())
        ).strip('_')
        for col in movies_dataset_x.columns
    ]

    movies_dataset_x = movies_dataset_x.apply(pd.to_numeric, errors='coerce')
    movies_dataset_x = movies_dataset_x.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    duplicate_columns = movies_dataset_x.columns[
        movies_dataset_x.columns.duplicated(keep=False)
    ]
    if len(duplicate_columns) > 0:
        raise ValueError("Duplicate feature names detected after encoding.")

    if movies_dataset_y.isna().any().any():
        raise ValueError("Revenue targets contain missing or invalid values.")

    return (
        movies_dataset_x,
        movies_dataset_y,
        emotion_labels,
        sentiment_labels,
        category_schema
    )


def main(input_path: str, output_x: str, output_y: str):
    """Prepare numeric feature and target pickles for LightGBM."""

    if input_path.endswith('.pkl'):
        movies_dataset = pd.read_pickle(input_path)
    else:
        movies_dataset = pd.read_csv(input_path)

    features, targets, _, _, _ = process_dataframe(movies_dataset)
    features.to_pickle(output_x)
    targets.to_pickle(output_y)

    print("\nProcessed Features X columns:", features.columns.tolist())
    print("\nNumber of features:", len(features.columns))
    print("\nProcessed Targets:", targets.columns.tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process structural metadata into numeric and multi-hot features."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_x", required=True)
    parser.add_argument("--output_y", required=True)

    args = parser.parse_args()
    main(
        input_path=args.input,
        output_x=args.output_x,
        output_y=args.output_y
    )
