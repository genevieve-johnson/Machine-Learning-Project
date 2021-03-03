# Machine-Learning-Project
Harry Potter Sorting pipeline for machine learning course. The goal of this pipeline is to accurately predict the Harry Potter House of an individual based off of their personality scores.

The Harry Potter dataset from Jakob et al. 2019 (https://doi.org/10.17605/OSF.IO/RTF74) consisted of 988 responses from a personality questionnaire. A small group and I constructed a pipeline implementing several different machine learning algorithms, as well as multiple styles of parameter optimization.

Along with demographic information, the personality dataset included the International Personality Item Pool (IPIP 50) to measure an individual's score of the Big Five traits: extraversion, conscientiousness, agreeableness, emotional stability, and openness to experience (Goldberg et al., 2006). The dataset also looked at the Short Dark Triad (SD3) to measure an individual's score for psychopathy, machiavellianism, and narcissism (Jones & Paulhus, 2013). Finally the Portrait Value Questionnaire (PVQ) was used to measure human values of an individual and their life goals (Schwartz et al 2012). 

Logistic regression, random forest, SVM, KNN, perceptron, and adaline SGD models were all implemented in this pipeline to find the best model for house prediction. For each of the algorithms, several hyperparameters were tested in a grid search method to identify the most accurate parameters for each model. Principal component analysis (PCA), with 2 to 30 components, was also conducted on logistic regression, SVM, KNN, perceptron, and adaline SGD models with the best parameters (from the grid search) to see if dimensionality reductions would increase the models' performances.

Overall, it was found that Random Forest trained on the standardized dataset and SVM trained on the normalized dataset performed with best. Random Forest's best parameters were Max_features =  ‘auto’ and N_estimators = 200 and it gave an accuracy of 73.6% and an F1 score of 73.3%. SVM's best parameters were C = 1, Kernel = 'rbf', DFS = 'ovo', and Gamma = 'scale', along with a PCA with 4 components which gave an accuracy of 70.7% and 68.6%.

To compare our results, we used scikit-learn’s DummyClassifier method to build a baseline model using the stratified strategy parameter and it gave an accuracy score of 27%.



This project was collaborated on by Annie Novak, Kathleen Delany, Elyse Geoffrey, and Laura Maskeri.
