# Machine Learning development over Spotify data 🎶

## Description

@letpires proposed this project as part of the #7DaysOfCode initiative organized by @alura. The challenge revolves around analyzing Spotify data and applying Machine Learning methodologies to predict the popularity of songs. Participants will gain hands-on experience in data manipulation, visualization, and analysis using the Python language.

This endeavor offers a chance for individuals unfamiliar with the subject to delve into Machine Learning, a technique enabling algorithms to "learn" specific tasks from data. Rather than explicitly programming rules for every scenario, a model is trained to recognize patterns within data, facilitating decision-making based on these identified patterns. The project encompasses various stages of a Machine Learning endeavor, spanning from data collection and exploratory analysis to model validation.

## Chapter 1: Exploratory Analysis 🕵️‍♂️

Exploratory data analysis assumes a pivotal role in comprehensively grasping the intricacies of the data at hand. When applied to Spotify music data, it aids in unveiling patterns, trends, and relationships among variables. This analytical process proves instrumental in extracting valuable insights related to the determinants of song popularity on Spotify, encompassing factors like musical attributes, song duration, music genres, popular artists, and other relevant aspects.

Engaging in this preliminary analysis not only unveils crucial insights within the data but also contributes to the construction of more resilient machine learning models for predicting song popularity on the Spotify platform. Additionally, exploratory analysis serves as a vital tool for detecting potential data issues, including missing values or inconsistencies, necessitating resolution before embarking on the creation of machine learning models.

## Chapter 2: Data Pre-process ⚙️

Machine learning encompasses three primary types: supervised, unsupervised, and reinforcement learning. This challenge specifically delves into supervised learning, which revolves around predicting an output variable based on a set of input variables. Within supervised learning, there are two focal categories: classification and regression. The emphasis in this challenge lies in constructing a classification model designed to predict whether a song will attain popularity or not.

The objective of predicting song popularity serves practical purposes, aiding in informed marketing decisions and enhancing the likelihood of a song's success. However, before delving into model creation, the crucial step of data preprocessing must be undertaken. This phase is pivotal in the machine learning process, involving the cleaning, organization, and transformation of raw data into a format suitable for model training. Various data preprocessing techniques, such as removing duplicate data, filling missing values, normalizing data, and feature engineering, can be applied to optimize the available data for subsequent modeling efforts.

## Chapter 3: Dataset division ➗

In the ongoing challenge, participants are tasked with the crucial process of partitioning their data into training, validation, and testing sets as a prerequisite before embarking on machine learning model development.

The significance of data partitioning lies in the objective of impartially evaluating the model's performance. Utilizing the entire dataset for training hinders the ability to gauge its effectiveness in generalizing to new data and acts as a precaution against overfitting, ensuring the model's competency extends beyond the training data. The training data serves as instruction for the model, while the testing data evaluates its performance on unfamiliar data, and the validation data aids in fine-tuning the model's hyperparameters for enhanced overall performance.

Several techniques exist for data partitioning, with random division being a common method allocating 70-80% for training, 10-20% for testing, and another 10-20% for validation. Despite its simplicity, this method may not be optimal for addressing data imbalances. Alternatively, cross-validation, exemplified by StratifiedKFold, is recommended to assess the model's generalization across diverse datasets and mitigate overfitting, particularly advantageous for handling unbalanced datasets. Following data partitioning, the subsequent step involves segregating the set into explanatory variables (X) such as musical genre and song duration and the output variable (Y) indicating the song's popularity, a variable participants aim to predict. The challenge encourages the use of cross-validation, specifically StratifiedKFold, with participants urged to compare its performance against other data partitioning techniques, underscoring the critical role of data splitting in influencing the machine learning process outcomes.

## Chapter 4: Baseline of the First Model 📈

This step involves establishing your inaugural Machine Learning model, commonly referred to as the baseline. This initial model serves as a foundational benchmark, representing a rudimentary solution to the problem at hand. Typically, after establishing the baseline, more intricate models are explored in an attempt to achieve improved results. Success is measured by surpassing the performance of the baseline model.

The baseline model, being a simplistic solution, aids in discerning whether the underlying challenge is associated with bias or variance. If the baseline model exhibits notably low accuracy, it may signal the complexity of the problem, indicating the necessity for a more advanced model to address it. Logistic Regression is frequently employed as a baseline model in classification tasks due to its simplicity and ease of interpretation. Despite its name suggesting regression, Logistic Regression is utilized for classification purposes. However, other classification models, such as Naive Bayes, Random Forest, Decision Tree, XGBoost, and others, are also viable options to explore beyond Logistic Regression. Commence the process by establishing the initial baseline model, defining the model, instantiating it, training it with the available data, and subsequently making predictions on both training and validation datasets.

## Chapter 5: Model Validation 🧪

Utilizing appropriate validation metrics for Machine Learning models is crucial in a project. The value of these metrics reflects the quality of a model; therefore, if poorly chosen, it may be impossible to assess whether the model meets the necessary requirements. The evaluation of a classification model is based on comparing the classes predicted by the model with the true classes of each example. All metrics aim to measure how far the model is from a perfect classification, but they do so in different ways. A simple way to visualize the performance of a model is through the confusion matrix, as shown in the figure. Applied to the challenge at hand: TP (true positive): the model correctly predicts a popular song; TN (true negative): the model correctly predicts a non-popular song; FN (false negative): the model predicts a song as popular, but it's not; FP (false positive): the model predicts a song as non-popular, but it is popular.

In addition to this metric, others can be used, such as Accuracy, which shows how many classifications the model got right among all; Precision, indicating how many of the model's Popular class classifications are correct; Recall, showing how many of the expected Popular class situations are correct; and F1-score, the harmonic mean between precision and recall. The choice of metric depends on the project's goal and the dataset characteristics. If the aim is to minimize false positives, precision will be the most important metric. If the goal is to minimize false negatives, recall will be the most important metric.

## Chapter 6: Resample the data and fit the selected model 🔄🤖

> Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer eu leo eget nulla volutpat tristique. Ut scelerisque odio et mi tristique, in interdum ligula laoreet. Morbi vel dapibus lectus.

## Chapter 7: Apply the result to the test data and save the result 💾

> Lorem ipsum dolor sit amet, consectetur adipiscing elit. Curabitur feugiat mi eu velit feugiat, vel auctor elit tristique. Vestibulum quis arcu et nulla euismod fringilla id et augue.