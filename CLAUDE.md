# CLAUDE.md — Disaster Response ML Pipeline

This file provides guidance for AI assistants (Claude and others) working in this repository. It describes the codebase structure, development workflows, and conventions to follow.

---

## Project Overview

A **Disaster Response Message Classifier** that processes disaster-related tweets and messages into 36 emergency categories (flood, fire, medical help, etc.) using a multi-label NLP classification pipeline. The system includes:

- **ETL pipeline** — cleans and loads raw CSV data into SQLite
- **ML pipeline** — trains a multi-label RandomForest classifier with TF-IDF features
- **Flask web app** — interactive dashboard for classifying new messages in real-time

Data source: ~26,000 disaster-response tweets from [Appen](https://appen.com) (formerly Figure Eight).

---

## Repository Layout

```
disaster-mlpipeline/
├── CLAUDE.md                         # This file
├── README.md                         # Project overview and deployment notes
├── resources/                        # Reference notebooks and raw data
│   ├── ETL Pipeline Preparation.ipynb
│   ├── ETL Pipeline Preparation-zh.ipynb   # Chinese translation
│   ├── ML Pipeline Preparation.ipynb
│   ├── ML Pipeline Preparation-zh.ipynb    # Chinese translation
│   ├── messages.csv                  # Raw messages (4.9 MB)
│   ├── categories.csv                # Raw categories (12 MB)
│   ├── Twitter-sentiment-self-drive-DFE.csv
│   └── etl_done.db                   # Reference SQLite output
└── web_app_v2/                       # Production application
    ├── Procfile                      # Heroku deployment config
    ├── requirements.txt              # Python dependencies
    ├── nltk.txt                      # NLTK data packages required
    ├── run.py                        # Flask app entry point
    ├── data/
    │   ├── process_data.py           # ETL script
    │   ├── DisasterResponse.db       # Production SQLite database
    │   ├── disaster_messages.csv     # Production raw messages
    │   └── disaster_categories.csv  # Production raw categories
    ├── models/
    │   ├── train_classifier.py       # ML training script
    │   └── classifier.pkl            # Trained model (gitignored)
    └── templates/
        ├── master.html               # Home page — visualizations + input form
        └── go.html                   # Results page — per-category predictions
```

---

## Data Flow

```
CSV files → process_data.py → DisasterResponse.db → train_classifier.py → classifier.pkl → run.py (Flask)
```

1. **ETL** (`process_data.py`): Merges message/category CSVs, cleans binary labels, saves to SQLite table `fact_messages`.
2. **Training** (`train_classifier.py`): Reads `fact_messages`, trains a sklearn `Pipeline`, serializes model to `classifier.pkl`.
3. **Serving** (`run.py`): Loads DB and model at startup; classifies user input on `/go` route; renders Plotly charts on `/`.

---

## Development Workflow

### Setup

```bash
cd web_app_v2
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

### 1. Run the ETL Pipeline

```bash
cd web_app_v2/data
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
```

Arguments: `<messages_csv> <categories_csv> <output_database_path>`

### 2. Train the ML Model

```bash
cd web_app_v2/models
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
```

Arguments: `<database_path> <model_output_path>`

Training prints per-category classification reports (precision, recall, f1-score) for all 36 labels.

### 3. Run the Web App

```bash
cd web_app_v2
python run.py
```

App runs at `http://localhost:3001` with debug mode enabled.

---

## Key Source Files

### `web_app_v2/data/process_data.py`

| Function | Purpose |
|----------|---------|
| `load_data(messages_filepath, categories_filepath)` | Merges CSVs on `id`, returns DataFrame |
| `clean_data(df)` | Splits category strings, converts to binary integers, drops duplicates |
| `save_data(df, database_filename)` | Writes to SQLite as table `fact_messages` (replaces if exists) |

Key data quirk: category values encoded as `"category_name-0"` or `"category_name-1"` — the script strips the prefix. The value `related-2` is treated as `related-1`.

### `web_app_v2/models/train_classifier.py`

| Function | Purpose |
|----------|---------|
| `load_data(database_filepath)` | Reads `fact_messages` table; returns `X` (messages), `Y` (36-column labels), `category_names` |
| `tokenize(text)` | Lowercases, removes non-alphanumeric chars, tokenizes, lemmatizes, removes English stopwords |
| `build_model()` | Returns `GridSearchCV`-wrapped sklearn `Pipeline` |
| `evaluate_model(model, X_test, Y_test, category_names)` | Prints `classification_report` per category |
| `save_model(model, model_filepath)` | Pickles fitted model |

**Model architecture:**
```python
Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
    ('clf',   MultiOutputClassifier(RandomForestClassifier()))
])
```

**GridSearchCV parameters:**
```python
{'clf__estimator__criterion': ['gini', 'entropy'],
 'clf__estimator__n_jobs':    [-1]}
```

### `web_app_v2/run.py`

- Defines the same `tokenize()` function (must stay in sync with `train_classifier.py`)
- Reads `fact_messages` from `DisasterResponse.db` via SQLAlchemy
- Loads `models/classifier.pkl` with joblib at startup
- Serves three Plotly charts:
  1. Pie — genre distribution (direct / social / news)
  2. Bar — message category distribution
  3. Box — message length by genre
- `/go?query=<text>` — returns `go.html` with 36 category predictions

---

## Database Schema

**Table: `fact_messages`** (SQLite — `web_app_v2/data/DisasterResponse.db`)

| Column | Type | Description |
|--------|------|-------------|
| `id` | BIGINT | Primary key |
| `message` | TEXT | English message |
| `original` | TEXT | Original language (may be null) |
| `genre` | TEXT | `direct`, `social`, or `news` |
| 36 category columns | INTEGER | Binary label: `0` or `1` |

**36 categories:**
`related`, `request`, `offer`, `aid_related`, `medical_help`, `medical_products`, `search_and_rescue`, `security`, `military`, `child_alone`, `water`, `food`, `shelter`, `clothing`, `money`, `missing_people`, `refugees`, `death`, `other_aid`, `infrastructure_related`, `transport`, `buildings`, `electricity`, `tools`, `hospitals`, `shops`, `aid_centers`, `other_infrastructure`, `weather_related`, `floods`, `storm`, `fire`, `earthquake`, `cold`, `other_weather`, `direct_report`

---

## Dependencies

Key packages (pinned in `web_app_v2/requirements.txt`):

| Package | Version | Role |
|---------|---------|------|
| Flask | 2.0.2 | Web framework |
| scikit-learn | 1.0.1 | ML pipeline |
| pandas | 1.3.4 | Data wrangling |
| nltk | 3.6.5 | NLP tokenization |
| SQLAlchemy | 1.4.27 | Database ORM |
| numpy | 1.21.4 | Numerical ops |
| plotly | 5.4.0 | Visualizations |
| joblib | 1.1.0 | Model serialization |
| gunicorn | 20.1.0 | Production WSGI server |

---

## Coding Conventions

- **File naming:** `snake_case.py`, `snake_case.html`
- **Variable naming:** `snake_case` throughout (e.g., `category_names`, `X_train`)
- **Function docstrings:** All public functions use Google-style docstrings with `Args:` and `Returns:` sections
- **CLI scripts:** Use `sys.argv` validation with a usage message; wrap execution in `if __name__ == '__main__':`
- **Tokenization:** The `tokenize()` function is duplicated in both `train_classifier.py` and `run.py`. Any changes **must be applied to both files** to avoid train/serve skew.
- **Model serialization:** Training saves with `pickle`; serving loads with `joblib` — both are compatible
- **Database writes:** `to_sql(..., if_exists='replace')` — re-running the ETL script overwrites the table
- **Flask debug mode:** Currently `True` in `run.py`; set to `False` for production

---

## Important Constraints

1. **`classifier.pkl` is gitignored** — it must be regenerated locally by running `train_classifier.py`. Never commit it.
2. **Hardcoded paths** — database and model paths are hardcoded relative to the working directory. Run scripts from their containing directory or adjust paths accordingly.
3. **No test suite** — there are no automated tests. Validation is done manually via classification reports and the Jupyter notebooks in `resources/`.
4. **No environment variables** — configuration (host, port, paths) is hardcoded. If adding env-var support, update `run.py` and document here.
5. **GridSearchCV is slow** — training runs a grid search; use `n_jobs=-1` and expect several minutes on a standard laptop.

---

## Deployment (Heroku)

```
# Procfile
web: gunicorn worldbank:web_app_v2
```

The app was originally deployed at: `https://my-disaster-message-classifier.herokuapp.com/`

NLTK data packages listed in `nltk.txt` are auto-installed on Heroku via a buildpack.

---

## Jupyter Notebooks (Reference)

Located in `resources/` — useful for understanding the rationale behind pipeline design:

- **`ETL Pipeline Preparation.ipynb`** — step-by-step walkthrough of the data cleaning logic
- **`ML Pipeline Preparation.ipynb`** — experiments with different models and features
- Chinese (`-zh`) versions of both notebooks are also available

These notebooks are not run as part of the pipeline — they are reference/documentation only.

---

## Common Tasks for AI Assistants

### Adding a new feature category
1. Add the category to the `categories.csv` source data
2. Re-run `process_data.py` to rebuild the DB
3. Re-run `train_classifier.py` to retrain the model
4. Update `go.html` if any display logic depends on category names

### Improving model performance
- Edit `build_model()` in `train_classifier.py`
- Add parameters to the `parameters` dict for `GridSearchCV`
- Consider adding features (e.g., `CountVectorizer`, custom transformers)
- Re-run training and check per-category F1 scores in the output

### Modifying visualizations
- All chart logic is in `run.py` in the `/` route handler
- Charts are built with `plotly.graph_objs` and JSON-encoded for the frontend
- `master.html` renders them via Plotly.js

### Changing the tokenizer
- Update `tokenize()` in **both** `train_classifier.py` and `run.py`
- Retrain the model after any tokenizer change (model was fit with the old tokenizer)
