# Aspect-Based Sentiment Analysis for Fine-Grained Customer Review Intelligence

## 1. Project Overview

Aspect-Based Sentiment Analysis (ABSA) is a Natural Language Processing (NLP) technique used to identify the sentiment expressed towards specific aspects or features mentioned in a text.

Traditional sentiment analysis classifies an entire review as positive, negative, or neutral. However, a single review can express different opinions about different aspects of the same product.

For example:

> "The battery life is excellent, but the screen quality is disappointing."

A general sentiment classifier may classify this review as mixed or negative. Aspect-Based Sentiment Analysis provides more detailed results:

| Aspect | Sentiment |
|---|---|
| Battery life | Positive |
| Screen quality | Negative |

This project develops a supervised machine learning system that predicts the sentiment associated with a given aspect term in laptop reviews.

The system uses TF-IDF feature extraction and compares Logistic Regression and Linear Support Vector Machine (Linear SVM) models.

## 2. Project Objectives

- Understand and analyse aspect-level customer review data.
- Prepare review sentences and their corresponding aspect terms for classification.
- Extract text features using TF-IDF.
- Develop and compare traditional machine learning classification models.
- Improve model configuration through hyperparameter tuning.
- Evaluate performance using multiple classification metrics.
- Analyse misclassified examples and identify model limitations.

## 3. Dataset Description

**Dataset:** SemEval-2014 Task 4 – Laptop Reviews

The project uses the labelled laptop review dataset containing aspect-level sentiment annotations.

The dataset contains the following columns:

| Column | Description |
|---|---|
| id | Original sentence identifier |
| Sentence | Customer review sentence |
| Aspect Term | Product aspect being discussed |
| polarity | Sentiment label |
| from | Starting position of the aspect term |
| to | Ending position of the aspect term |

**Dataset statistics:**

- Total labelled records: 2,358
- Unique sentence IDs: 1,488
- Unique aspect terms: 1,042
- Sentiment categories: 4

### Sentiment distribution

| Sentiment | Records | Percentage |
|---|---:|---:|
| Positive | 987 | 41.86% |
| Negative | 866 | 36.73% |
| Neutral | 460 | 19.51% |
| Conflict | 45 | 1.91% |
| **Total** | **2,358** | **100%** |

The dataset is divided into training and testing subsets using GroupShuffleSplit to prevent records associated with the same original sentence ID from appearing in both subsets.

- Training records: 1,880
- Testing records: 478
- Sentence ID overlap: 0

The raw dataset is not included in this repository.

## 4. Methodology

The project follows a structured machine learning workflow.

### Stage 1: Dataset Inspection
- Loaded and inspected the labelled dataset.
- Examined dataset dimensions, columns, missing values and duplicates.
- Identified the sentiment categories.

### Stage 2: Data Preparation and Exploratory Data Analysis
- Analysed sentiment distribution.
- Examined aspect-term frequency.
- Checked aspect positions and sentence-level information.
- Generated exploratory visualisations.

### Stage 3: TF-IDF Feature Extraction
- Combined each review sentence with its corresponding aspect term.
- Used TF-IDF to convert text into numerical features.
- Applied unigram and bigram feature extraction.
- Used a grouped train-test split to avoid sentence-level data leakage.
- Fitted the feature extractor using training data only.

### Stage 4: Model Training
Two supervised machine learning algorithms were implemented:

1. Logistic Regression
2. Linear Support Vector Machine (Linear SVM)

Class weighting was used to address the uneven distribution of sentiment categories.

### Stage 5: Model Evaluation and Hyperparameter Tuning
- Evaluated both baseline models.
- Compared accuracy, precision, recall and F1-score.
- Conducted grouped cross-validation for hyperparameter tuning.
- Evaluated the tuned Linear SVM on the original held-out test set.
- Selected the best-performing model based on Macro-F1.

### Stage 6: Final Model Interpretation and Error Analysis
- Analysed correct and incorrect predictions.
- Examined class-wise error rates.
- Identified common sentiment confusion patterns.
- Saved misclassified examples for further inspection.

## 5. Algorithms and Technologies

### Machine Learning Algorithms

**Logistic Regression**

A supervised classification algorithm that estimates class probabilities and assigns an input to a sentiment category.

**Linear Support Vector Machine**

A supervised learning algorithm that finds decision boundaries to separate sentiment categories. It is particularly useful for high-dimensional text classification.

### Feature Extraction

**TF-IDF (Term Frequency–Inverse Document Frequency)**

TF-IDF converts textual information into numerical features based on the importance of words and phrases in the documents.

### Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- SciPy
- Joblib
- Spyder IDE

## 6. Experimental Results

The models were evaluated using the same held-out test set containing 478 records.

### Baseline Model Comparison

| Metric | Logistic Regression | Baseline Linear SVM |
|---|---:|---:|
| Accuracy | 0.6192 | 0.6360 |
| Macro Precision | 0.4713 | 0.4735 |
| Macro Recall | 0.4430 | 0.4511 |
| Macro F1 | 0.4541 | **0.4552** |
| Weighted F1 | 0.6269 | **0.6300** |

The baseline Linear SVM achieved the highest Macro-F1 among the two baseline models.

### Hyperparameter Tuning

The Linear SVM was further investigated using grouped three-fold cross-validation.

The best cross-validation configuration was:

- Algorithm: Linear SVM
- N-gram range: (1, 1)
- C: 0.1
- Class weight: Balanced
- Stop words: None
- Mean cross-validation Macro-F1: 0.4595

### Final Model Comparison

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6192 | 0.4713 | 0.4430 | 0.4541 | 0.6269 |
| Baseline Linear SVM | 0.6360 | **0.4735** | **0.4511** | **0.4552** | **0.6300** |
| Tuned Linear SVM | **0.6402** | 0.4699 | 0.4464 | 0.4444 | 0.6232 |

**Selected Final Model: Baseline Linear SVM**

Although the tuned Linear SVM achieved slightly higher accuracy, its Macro-F1 was lower than the baseline Linear SVM.

Therefore, the baseline Linear SVM was selected as the final model because it achieved the best Macro-F1 and weighted F1 among the evaluated models.

## 7. Final Model Error Analysis

The selected baseline Linear SVM was analysed on the 478-record test set.

| Prediction Result | Records |
|---|---:|
| Correct predictions | 304 |
| Incorrect predictions | 174 |
| Total | 478 |

Final accuracy: **63.60%**

### Class-wise Error Analysis

| Sentiment | Total Records | Correct | Incorrect | Recall |
|---|---:|---:|---:|---:|
| Conflict | 4 | 0 | 4 | 0.00 |
| Negative | 187 | 130 | 57 | 0.6952 |
| Neutral | 109 | 37 | 72 | 0.3395 |
| Positive | 178 | 137 | 41 | 0.7697 |

### Key Observations

- Positive sentiment achieved the highest recall.
- Negative sentiment showed reasonable classification performance.
- Neutral sentiment was difficult to distinguish from positive and negative opinions.
- All four conflict examples in the test set were misclassified. This result must be interpreted cautiously because the test support is extremely small.

The most frequent confusion patterns included:

- Neutral predicted as negative: 38 records.
- Neutral predicted as positive: 32 records.
- Positive predicted as negative: 29 records.
- Negative predicted as positive: 28 records.

## 8. Project Outputs

The `results/` directory contains important project outputs, including:

- Exploratory data analysis summaries.
- Sentiment distribution visualisation.
- Model evaluation reports.
- Confusion matrices.
- Macro-F1 comparison plots.
- Hyperparameter tuning results.
- Final evaluation summary.
- Error analysis report.

## 9. Project Structure

```text
Aspect-Based Sentiment Analysis/
│
├── data/
│   └── Raw datasets (not included in GitHub)
│
├── models/
│   └── Trained models and feature matrices
│       (not included in GitHub)
│
├── outputs/
│
├── results/
│   ├── sentiment_distribution.png
│   ├── stage2_eda_summary.txt
│   ├── stage3_tfidf_summary.txt
│   ├── stage4_model_training_summary.txt
│   ├── stage5_model_comparison.txt
│   ├── stage5_2_tuning_results.txt
│   ├── stage5_3_final_evaluation.txt
│   ├── stage5_3_final_macro_f1_comparison.png
│   └── stage6_error_analysis.txt
│
├── stage1_dataset_inspection.py
├── stage2_data_preparation_eda.py
├── stage3_tfidf_feature_extraction.py
├── stage4_model_training.py
├── stage5_model_evaluation.py
├── stage5_1_trial_dataset_audit.py
├── stage5_2_feature_engineering_tuning.py
├── stage5_3_final_tuned_model_evaluation.py
├── stage6_final_model_error_analysis.py
│
├── .gitignore
└── README.md
```

## 10. Limitations

1. **Limited Dataset Size:** The model uses 2,358 labelled records, which may restrict its ability to generalise to larger and more diverse datasets.

2. **Class Imbalance:** The conflict category is significantly underrepresented, with only 45 examples in the complete labelled dataset.

3. **Neutral Sentiment Classification:** The model has relatively low neutral recall, indicating difficulty distinguishing neutral statements from positive and negative opinions.

4. **Predefined Aspect Terms:** The aspect term must be supplied to the model. Automatic aspect extraction is not implemented.

5. **Limited Contextual Understanding:** TF-IDF and traditional machine learning algorithms cannot fully capture complex contextual relationships, sarcasm and subtle linguistic expressions.

6. **Domain Dependency:** The model is trained on laptop reviews. Its performance on other domains has not been established.

7. **Limited Generalisation Evaluation:** The project uses one fixed grouped train-test split and does not include independent external dataset evaluation.

## 11. Future Scope

The project can be extended through:

- Automatic aspect term extraction.
- Transformer-based language models such as BERT.
- Larger and more balanced labelled datasets.
- Cross-domain sentiment classification.
- Repeated grouped cross-validation.
- Improved contextual text representations.
- Further investigation of neutral and conflict sentiment classification.

## 12. Conclusion

This project demonstrates a supervised machine learning approach to aspect-level sentiment classification of laptop reviews.

The implementation includes dataset inspection, exploratory analysis, TF-IDF feature extraction, model training, hyperparameter tuning, final model comparison and error analysis.

Among the evaluated models, the baseline Linear SVM achieved the highest Macro-F1 score of **0.4552** and was selected as the final model.

The experiments also revealed challenges in neutral and conflict sentiment classification, highlighting opportunities for further research and improvement.

**This project is an academic machine learning experiment and is not intended to represent a fully automated, production-ready ABSA system.**s