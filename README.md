# SentimentScope

SentimentScope is a web-based sentiment analysis project for customer product reviews. It compares two classic machine learning models, Logistic Regression and Naive Bayes, and shows their predictions side by side so you can understand not only the final sentiment, but also how each model behaves on the same input.

The project is built around Flipkart product review data and includes:

- a Flask backend that loads pre-trained sentiment models
- a browser-based UI for entering or pasting review text
- a preprocessing pipeline that cleans text before prediction
- a model comparison dashboard with confidence scores, metrics, and confusion matrices
- saved training assets so predictions can be made without retraining

This README is meant to explain the project thoroughly without requiring you to run it first.

## What the project does

At a high level, SentimentScope classifies a review into one of three sentiment categories:

- Negative
- Neutral
- Positive

When a user enters a review, the app:

1. cleans the text using the same preprocessing logic that was used during training
2. vectorizes the cleaned text in two different ways
3. sends the text through two separate trained models
4. returns each model’s predicted sentiment and confidence
5. displays a comparison UI that helps you interpret the result

The key idea is that this is not just a single-model classifier. It is a comparison tool that lets you see how Logistic Regression and Naive Bayes perform on the same review.

## Main files in the project

- [`app.py`](./app.py) contains the Flask server, text preprocessing, model loading, prediction API, and evaluation metrics endpoint.
- [`index.html`](./index.html) is the main page layout for the sentiment analyzer.
- [`script.js`](./script.js) handles browser-side interaction, API calls, charts, and result rendering.
- [`style.css`](./style.css) defines the visual design and layout.
- [`FMiniProject.ipynb`](./FMiniProject.ipynb) is the notebook used to explore data, clean text, train models, and save the artifacts used by the app.
- [`Flipkart_Product.csv`](./Flipkart_Product.csv) is the dataset of product reviews used for training and evaluation.
- [`model_lr.pkl`](./model_lr.pkl) is the saved Logistic Regression model.
- [`model_nb.pkl`](./model_nb.pkl) is the saved Naive Bayes model.
- [`tfidf.pkl`](./tfidf.pkl) is the TF-IDF vectorizer used with Logistic Regression.
- [`bow.pkl`](./bow.pkl) is the Bag-of-Words vectorizer used with Naive Bayes.
- [`classes.pkl`](./classes.pkl) stores the label mapping used to convert model outputs back into human-readable sentiment names.

## User experience

The front end is designed like a model comparison dashboard rather than a plain input form.

### Input area

Users can:

- type or paste any product review
- insert a positive example
- insert a negative example
- submit the review with the predict button
- use `Ctrl + Enter` or `Cmd + Enter` as a shortcut

### Results area

After prediction, the page shows:

- Logistic Regression sentiment and confidence
- Naive Bayes sentiment and confidence
- confidence difference between the two models
- whether the models agree or disagree
- which model is more confident on that specific input
- a prediction summary
- confusion matrices for both models
- bar charts for model performance
- a comparison of confidence scores
- an overall model conclusion based on evaluation metrics

This makes the app useful both as a demo and as a teaching tool.

## Backend architecture

The backend is implemented in Flask in [`app.py`](./app.py).

### Flask routes

#### `GET /`

Serves [`index.html`](./index.html) directly from the project root.

#### `GET /<path:filename>`

Serves static files like:

- `style.css`
- `script.js`
- local assets if they are added later

#### `GET /metrics`

Returns evaluation metrics for both models on the held-out test split.

The response includes:

- accuracy
- precision
- recall
- F1 score
- confusion matrix
- per-class recall
- correct / incorrect / total counts
- a natural-language conclusion comparing the models
- a `preferred_model` label

#### `POST /predict`

Accepts a JSON payload with a review text and returns:

- Logistic Regression prediction + confidence
- Naive Bayes prediction + confidence

The returned confidence is based on the model’s highest predicted probability for the input.

## Text preprocessing pipeline

The project uses a carefully matched cleaning pipeline in both training and inference. This matters because the model should see text in the same format during prediction that it saw during training.

The `clean_text` function in [`app.py`](./app.py) performs the following steps:

1. converts emojis to text using `emoji.demojize`
2. replaces emoji separators like `:` and `_` with spaces
3. lowercases everything
4. removes non-letter characters
5. tokenizes the text using NLTK `word_tokenize`
6. removes English stopwords
7. preserves important negation words:
   - `not`
   - `no`
   - `never`
8. lemmatizes each remaining token with `WordNetLemmatizer`

Preserving negation words is important in sentiment analysis because removing them can flip meaning. For example:

- “good” and “not good” should not be treated the same

## Model design

SentimentScope uses two traditional machine learning approaches.

### Logistic Regression

- Uses TF-IDF features
- Loaded from [`model_lr.pkl`](./model_lr.pkl)
- Paired with [`tfidf.pkl`](./tfidf.pkl)

### Naive Bayes

- Uses Bag-of-Words features
- Loaded from [`model_nb.pkl`](./model_nb.pkl)
- Paired with [`bow.pkl`](./bow.pkl)

### Why two models?

Using both models gives a useful comparison:

- Logistic Regression often works well with sparse TF-IDF features and linear decision boundaries.
- Naive Bayes is a classic baseline for text classification and is fast and interpretable.

Because both are evaluated on the same test split, the app can present a fair side-by-side comparison rather than a single score in isolation.

## Label mapping

The model predictions are numeric internally, so the app loads [`classes.pkl`](./classes.pkl) to map them back to readable sentiment labels.

The code comments indicate the standard order is:

1. Negative
2. Neutral
3. Positive

That mapping is used in both:

- the prediction endpoint
- the evaluation metrics builder

## Evaluation and comparison logic

The app does more than predict a single review. It also evaluates both models on the dataset to generate a dashboard of performance information.

### How metrics are built

When the app starts, it:

1. loads the Flipkart review dataset
2. combines the review summary and review text
3. cleans the combined text with the same preprocessing pipeline
4. splits the processed data into train and test sets using an 80/20 split
5. transforms the test text using each model’s vectorizer
6. generates predictions for both models
7. computes evaluation metrics

### Metrics reported

For each model, the app computes:

- accuracy
- weighted precision
- weighted recall
- weighted F1 score
- confusion matrix
- per-class recall

The app then compares the two models and generates a human-readable conclusion explaining which model is stronger overall, or whether the split is effectively a tie.

### Why this is useful

Most sentiment apps only show “positive” or “negative.” This project is more educational:

- it shows the model prediction for a specific review
- it shows confidence for each model
- it explains how the models perform overall
- it exposes where one model may do better than the other

## Frontend behavior

The browser logic lives in [`script.js`](./script.js).

### What the frontend does

The JavaScript code:

- fills the review box with positive/negative examples
- sends review text to `/predict`
- fetches `/metrics` the first time results are needed
- renders the model cards and confidence bars
- draws charts with Chart.js
- populates the confusion matrix table
- updates the summary text and conclusion
- handles simple UX polish like loading states and textarea shake on empty submit

### Charts and visualizations

The page uses Chart.js to show:

- a bar chart of accuracy, precision, recall, and F1 for the active model
- a confidence comparison chart for the current review

### Model comparison controls

The confusion matrix section includes a toggle between:

- Logistic Regression
- Naive Bayes

This lets the viewer inspect each model’s class-wise behavior without leaving the page.

## Dataset

The dataset file is [`Flipkart_Product.csv`](./Flipkart_Product.csv).

From the notebook and backend code, the dataset contains columns similar to:

- `ProductName`
- `Price`
- `Rate`
- `Review`
- `Summary`
- `Sentiment`

The review text and summary are combined before cleaning, which gives the model more context than review text alone.

## Notebook workflow

The notebook [`FMiniProject.ipynb`](./FMiniProject.ipynb) appears to be the training and experimentation workspace for the project.

Based on the notebook and server code, the workflow was roughly:

1. load the Flipkart dataset
2. inspect the data structure and missing values
3. clean and normalize the text
4. encode sentiment labels
5. split the data into train and test sets
6. train Logistic Regression and Naive Bayes models
7. save the trained models and vectorizers
8. evaluate model performance
9. export the artifacts used by the Flask app

This separation is important:

- the notebook is for training and experimentation
- the Flask app is for inference and presentation

## NLTK resources

The app checks for required NLTK resources on startup and downloads them if needed:

- `punkt`
- `punkt_tab`
- `stopwords`
- `wordnet`

These are needed for tokenization, stopword filtering, and lemmatization.

## Dependencies and libraries used

The project uses the following main libraries:

- Flask for the backend server
- Flask-CORS for cross-origin support
- joblib for loading saved models and vectorizers
- regex (`re`) for text cleaning
- emoji for emoji-to-text conversion
- NLTK for tokenization, stopwords, and lemmatization
- scikit-learn for vectorization, model inference, metrics, and train/test split
- Chart.js for frontend charts

## Important implementation details

### 1. Training and inference preprocessing match

The same cleaning logic is used in the notebook and the Flask app. That consistency is essential for reliable results.

### 2. Each model uses its own vectorizer

- Logistic Regression uses TF-IDF
- Naive Bayes uses Bag-of-Words

This is not interchangeable in the current project structure.

### 3. Confidence is model probability, not ground truth certainty

The confidence value shown in the UI is the highest probability output from each model. It is useful as a relative signal, but it is not a guarantee of correctness.

### 4. The dashboard evaluates the models on a held-out test split

The performance metrics are not just hardcoded. They are derived from the dataset split used in evaluation.

### 5. The app is built to be read, not just used

The UI and metrics presentation are part of the project’s educational value. It helps viewers understand both the prediction and the model quality behind it.

## Project strengths

- compares two different NLP classifiers side by side
- includes both prediction and evaluation views
- uses a reproducible preprocessing pipeline
- preserves negation, which matters in sentiment tasks
- loads pre-trained artifacts for fast inference
- has a polished, explainable frontend

## Limitations and notes

- The app is designed for product-review style sentiment analysis, so it may not generalize perfectly to very different kinds of text.
- The models are classical ML models, not transformer-based language models.
- Confidence scores reflect model probability, which can still be overconfident on unfamiliar inputs.
- Evaluation metrics depend on the dataset split and the quality of the underlying labels in the Flipkart dataset.

## What someone should understand at a glance

If you only remember one thing about SentimentScope, it is this:

> It is a sentiment analysis dashboard that compares Logistic Regression and Naive Bayes on Flipkart-style product reviews, showing both the prediction for a given review and the model’s broader evaluation behavior.

## If you want to extend it

Good future improvements would be:

- adding more sentiment classes or confidence calibration
- supporting batch review uploads
- showing feature explanations for model decisions
- replacing classical models with modern transformer-based classifiers
- adding a deployment configuration or Docker setup
- exposing the training pipeline as a reproducible script

## File-by-file summary

- [`app.py`](./app.py): backend server, API routes, cleaning, model loading, evaluation metrics
- [`index.html`](./index.html): page structure and UI sections
- [`script.js`](./script.js): API integration, charts, result rendering, interaction logic
- [`style.css`](./style.css): visual design and layout
- [`FMiniProject.ipynb`](./FMiniProject.ipynb): data exploration, preprocessing, training, evaluation
- [`Flipkart_Product.csv`](./Flipkart_Product.csv): source dataset
- [`model_lr.pkl`](./model_lr.pkl): trained Logistic Regression classifier
- [`model_nb.pkl`](./model_nb.pkl): trained Naive Bayes classifier
- [`tfidf.pkl`](./tfidf.pkl): TF-IDF vectorizer
- [`bow.pkl`](./bow.pkl): Bag-of-Words vectorizer
- [`classes.pkl`](./classes.pkl): sentiment label mapping

## Short summary

SentimentScope is a sentiment analysis project for product reviews. It takes a review, cleans it, runs it through two saved text-classification models, and presents the results in a clear comparison dashboard. The project is especially useful as a mini case study in classic NLP pipelines because it shows the full path from raw review text to trained model comparison.

