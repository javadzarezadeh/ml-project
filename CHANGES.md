# Code and Scientific Review

## Fixed problems

### 1. Review boundaries are corrupted

**Files:** `sentiment-analyzer.py`, `emotion-analyzer.py`

The analyzers use `reviews_row.split(",")` to recover individual reviews.
However, the `Reviews` column contains a stringified Python list, and review
text itself contains commas. This produces many more fragments than actual
reviews and sends incomplete fragments to the transformer.

For example, the first 100 rows have between 0 and 15 actual reviews, but the
comma-splitting code produces between 2 and 102 fragments for annotated rows.

**Solution:** The analyzers now parse the column with `ast.literal_eval`,
validate that the result is a list, classify each complete list element, and
return an empty annotation list for movies without reviews.

**Why it matters:** Every sentiment and emotion feature is currently based on
incorrect text units. The generated annotations must be regenerated with the
corrected analyzers before experiments are run.

### 2. Processed data was not written to the project folder

**File:** `run-experiments.py`

The experiment runner used a working-directory-relative path and no longer
wrote the split datasets used by training.

**Solution:** Processed fit, validation, and test feature/target pickles are
now written under `ml-project-main-3/processed_data`, independent of the
directory from which the runner is started. Feature rows retain their source
indices so they remain aligned with target rows.

### 3. The predictor does not save a model or predictions

**File:** `run-experiments.py`

The experiment runner prints R², RMSE, and MAE but never serializes the fitted model,
predictions, actual test values, target name, or feature names.

**Solution:** The experiment runner now saves a structured model artifact
containing the model, target, configuration, feature names, log-scale and
dollar-scale predictions, and metrics.

**Why it matters:** Results cannot be reproduced because the fitted model and
its predictions are not saved. Preprocessing labels and category schema are
also included in the artifact.

### 4. Revenue was modeled on the raw dollar scale

**File:** `run-experiments.py`

The target was passed directly to LightGBM without a `log1p` transformation.
Movie revenue is strongly right-skewed, so blockbuster titles can dominate
RMSE and training loss.

**Solution:** The predictor now trains on `log1p(revenue)`, clips inverse
predictions at zero, and reports both log-scale and dollar-scale metrics.

**Why it matters:** Log transformation generally gives the model a more stable
objective across low- and high-revenue movies.

### 5. Emotion aggregation discarded frequency information

**File:** `data-processor.py`

`transform_emotions` groups by label and keeps only the maximum score for each
emotion. This removes how often an emotion occurs and makes one highly
confident review represent the entire movie.

**Solution:** Annotation features now preserve mean score, maximum score,
label fraction, and analyzed-review count for every training-fitted label.

**Why it matters:** Frequency and confidence describe different properties.
Keeping only maxima creates unstable and difficult-to-interpret features.

### 6. Feature encoding was learned before validation

**File:** `data-processor.py`

Language, genre, and production-country mappings are built using all rows
before the predictor splits the data. This exposes validation/test category
information to preprocessing and makes the preprocessing difficult to
reproduce on future data.

**Solution:** The predictor now accepts raw data, splits rows before
preprocessing, fits the schema only on the fitting split, and reindexes
validation/test matrices to the fitting feature columns.

**Why it matters:** All learned preprocessing should be contained within the
training split.

### 7. Missing values were not handled deliberately

**File:** `data-processor.py`

The supplied data contains missing values in genres, ratings, vote counts,
original language, and production countries. Rating conversion can produce
errors for unexpected formats, while categorical missing values are not given
an explicit semantic category.

**Solution:** Numeric conversion is validated for targets, ratings and other
features are coerced deliberately, missing categorical values become
`Missing`, unknown categories are ignored, and remaining numeric missing values
are filled with zero.

**Why it matters:** Silent or inconsistent missing-value behavior can change
the feature matrix and make runs non-reproducible.

### 8. Useful numeric predictors were discarded

**File:** `data-processor.py`

`Rating` and `Vote_Count` were removed before preprocessing instead of being
converted and retained.

**Solution:** Ratings are parsed from values such as `6.126/10`, vote counts
are coerced to numeric values, and missing numeric values are filled with zero.

### 9. Annotated datasets could be consumed when reviews and annotations differed

**File:** `data-processor.py`

Existing annotated CSVs contained different numbers of reviews and annotations
in the same row, allowing corrupted review features to enter training.

**Solution:** Preprocessing now rejects misaligned annotated rows and instructs
the user to regenerate them with the corrected analyzer scripts.
