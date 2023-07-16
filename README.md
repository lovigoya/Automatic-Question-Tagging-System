# Automatic Question Tagging System

## Abstract
Tagging is a widely-used approach for organizing information and facilitating content search in information systems. It is particularly relevant for categorizing questions on platforms like Quora and Stack Overflow. In this report, we analyze the StackSample:10% of Stack Overflow Q&A dataset from Kaggle. We employ various classification algorithms, including "One-vs-Rest Classifier with Support Vector Classification," "Logistic Regression," and "Random Forest Classification," to perform question tagging. Our results demonstrate that the Random Forest algorithm outperforms the others, achieving a Jaccard Score of 0.921 and a Hamming Loss of 0.011.

## Introduction
Platforms such as Quora and Stack Overflow frequently prompt users to provide tags or labels for their questions to enhance categorization and improve searchability. However, users sometimes provide inaccurate or irrelevant tags, which hampers the effectiveness of these systems. To address this issue, we propose an automatic question tagging system capable of accurately identifying relevant tags for user-submitted questions.

## Data
The dataset used for this analysis can be accessed at [https://www.kaggle.com/stackoverflow/stacksample](https://www.kaggle.com/stackoverflow/stacksample). It contains the text of 10% of the questions and answers from the Stack Overflow programming Q&A website. The dataset is divided into three tables:

1. Questions: This table includes non-deleted Stack Overflow questions with an ID that is a multiple of 10. It provides information such as the title, body, creation date, closed date (if applicable), score, and owner ID.
2. Answers: Here, you can find the answers corresponding to the questions in the first table. It includes the body, creation date, score, owner ID, and references the Questions table using the ParentId column.
3. Tags: Each question has a set of tags associated with it, which are listed in this table.

To prepare the data for analysis, we performed data and text preprocessing steps. First, we removed unnecessary columns such as creation date, closed date, and score. Then, we cleaned the text by removing HTML formatting, converting text to lowercase, transforming abbreviations, removing punctuation, handling popular tags like "C#", lemmatizing words, and removing stop words. Exploratory Data Analysis was also conducted to identify popular patterns and tags.

## Analysis and Results
After preprocessing and data exploration, we trained our models using 80% of the data. We applied the following classification algorithms: "One-vs-Rest Classifier with Support Vector Classifier," "Logistic Regression," and "Random Forest Classification."

We evaluated the models using the following metrics:

1. Jaccard Score: The Jaccard similarity index measures the similarity between two sets of data, with values ranging from 0 to 1. A higher score indicates a greater degree of similarity.
2. Hamming Loss: This metric represents the fraction of incorrect labels to the total number of labels. In multi-label classification, Hamming Loss penalizes only individual labels.

Here are the results obtained from the models:

- Support Vector Classification:
  - Jaccard Score: 0.6473743535338152
  - Hamming Loss: 0.0123506539413220

- Logistic Regression:
  - Jaccard Score: 0.892213344137751
  - Hamming Loss: 0.0124319547543301

- Random Forest Classification:
  - Jaccard Score: 0.92153434145231
  - Hamming Loss: 0.011539876329

Based on these results, the Random Forest Classification algorithm achieved the highest Jaccard Score and the lowest Hamming Loss, making it the best fit model for the given dataset.

## Conclusion
In this analysis, we applied machine learning algorithms to the TF-IDF vectorization of the text. This approach yielded better accuracy compared to vectorizing the body and title individually. Among the classifiers tested, the Random Forest Classifier outperformed both the logistic regression classifier and support vector classifier in terms of both Jaccard score (similarity) and Hamming loss (data loss). Thus, we consider the Random Forest Classifier as the most suitable model for future analysis with respect to the dataset used.

## Future Research Directions
For future research, we recommend exploring the use of Artificial Neural Networks, specifically incorporating state-of-the-art NLP transformers such as BERT and GPT. This approach would allow for retaining sequence information and directly feeding textual data into sequence-to-sequence models, as detailed in Appendix_1. In contrast, traditional approaches involve tokenizing sentences and converting them to numerical representations using frequency calculations or the TF-IDF approach.

## Milestones and References
The milestones achieved in this project include Data & Text Preprocessing, Basic Data Analysis on Tags, and Supervised ML models. The following references were used:

1. [Multi-label classification - Wikipedia](https://en.wikipedia.org/wiki/Multi-label_classification#Statistics_and_evaluation_metrics)
2. [TF-IDF approach - Wikipedia](https://en.wikipedia.org/wiki/Multi-label_classification#Statistics_and_evaluation_metrics)
3. [Kaggle: Multi-label classification for tag prediction](https://www.kaggle.com/vikashrajluhaniwal/multi-label-classification-for-tag-prediction)

## Appendix
For further reading, you may refer to the following resources:

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)