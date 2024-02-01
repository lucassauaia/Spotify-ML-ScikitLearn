# Machine Learning development over Spotify data

## Description

This project was proposed by @letpires for the #7DaysOfCode campaign made by @alura.

In this challenge, the objective is to analyze data from Spotify and apply Machine Learning techniques to predict the popularity of songs.

Additionally, participants will have the opportunity to utilize data manipulation, visualization, and analysis skills using the Python language.

For those new to the subject, Machine Learning is a technique that allows an algorithm to “learn” how to perform a specific task from data. This means that instead of programming specific rules for each situation, a model can be taught to recognize patterns in data and make decisions based on those patterns.

The stages of a Machine Learning project, from data collection and exploratory analysis to model validation, will be covered.

## Chapter 1: Exploratory Analysis

Exploratory data analysis plays a crucial role in gaining a comprehensive understanding of the data to be utilized. In the context of Spotify music data, it aids in the identification of patterns, trends, and relationships among available variables. This process facilitates the extraction of valuable insights pertaining to the factors contributing to a song's popularity on Spotify, including musical attributes, song duration, music genres, popular artists, and other relevant aspects.

Conducting this preliminary analysis allows for the discovery of pertinent insights within the data, contributing to the development of more robust machine learning models for predicting song popularity on Spotify.

Furthermore, exploratory analysis can be instrumental in recognizing potential data issues, such as missing values or inconsistencies, which require resolution prior to model creation.

## Chapter 2: Data Pre-process

There are three main types of machine learning: supervised, unsupervised, and reinforcement. In this challenge, you will focus on supervised learning, which involves predicting an output variable based on a set of input variables.

Going deeper, there are two types of supervised learning: classification and regression. In this challenge, the focus will be on developing a classification model to predict whether a song will be popular or not.

But what are you going to do this for? Understanding whether a song will be popular or not can help you make better marketing decisions and promote the success of a song, for example.

Before you start creating your models, you will need to go through the data preprocessing step. Data preprocessing is one of the most important steps in the Machine Learning process. This is the phase where you will clean, organize and transform the raw data into data that can be used to train your models.

There are several data preprocessing techniques you can apply, such as: removing duplicate data, filling missing data, normalizing data, feature engineering, and others. So you can start by applying some of these techniques to the available data.

## Chapter 3: Dataset division

In the current challenge, participants will engage in the process of partitioning their data into training, validation, and testing sets. This step is crucial before developing machine learning models.

The necessity for data partitioning arises from the objective of impartially assessing the model's performance. Employing the entire dataset for training precludes the ability to ascertain its adequacy for generalizing to new data. Moreover, this approach safeguards against overfitting, ensuring the model's proficiency extends beyond the training data.

In essence, the training data instructs the model, while testing data gauges the model's performance on unfamiliar data. Validation data is instrumental in fine-tuning the model's hyperparameters, enhancing its overall performance.

Various techniques exist for data partitioning. One such method is random division, which randomly allocates the data into three sets. Typically, 70-80% of the data is allocated for training, 10-20% for testing, and another 10-20% for validation. Although simple and expeditious, this method may not be suitable in the presence of data imbalance.

An alternative approach is cross-validation, designed to assess the model's generalization across diverse datasets. It aids in mitigating overfitting, wherein the model excessively conforms to the training data but falters in generalizing to new data.

A prevalent form of cross-validation is StratifiedKFold, particularly beneficial for handling unbalanced datasets.

Following data partitioning, the next step involves segregating the set into X and Y. In this case, X represents the explanatory variables, such as musical genre, song duration, instrumentation, and Y signifies the output, indicating the song's popularity, a variable participants aim to predict.

The suggestion for today's challenge is to employ cross-validation for data partitioning. As an additional challenge, participants are encouraged to use StratifiedKFold and compare its performance with other data partitioning techniques.

It's crucial to emphasize that data splitting constitutes a pivotal phase in the machine learning process, exerting a substantial impact on the model's outcomes.

## Chapter 4: Baseline of the First Model

> Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque vehicula, leo in aliquet dapibus, ex urna fringilla arcu, eu vehicula quam elit a ex. Etiam non justo ac nunc dignissim pellentesque.

## Chapter 5: Model Validation

> Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce efficitur risus eget lectus scelerisque, in volutpat tortor aliquet. Curabitur vel augue in felis fringilla facilisis. Aenean lacinia purus vel quam vestibulum tincidunt.

## Chapter 6: Resample the data and fit the selected model

> Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer eu leo eget nulla volutpat tristique. Ut scelerisque odio et mi tristique, in interdum ligula laoreet. Morbi vel dapibus lectus.

## Chapter 7: Apply the result to the test data and save the result

> Lorem ipsum dolor sit amet, consectetur adipiscing elit. Curabitur feugiat mi eu velit feugiat, vel auctor elit tristique. Vestibulum quis arcu et nulla euismod fringilla id et augue.