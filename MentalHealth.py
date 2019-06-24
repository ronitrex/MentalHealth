import numpy as np
import pandas as pd  # data processing, CSV file I/O
import matplotlib.pyplot as plt
import seaborn as sns

from subprocess import check_output
from sklearn import preprocessing

from sklearn.preprocessing import binarize, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

from mlxtend.classifier import StackingClassifier
from scipy.stats import randint

# models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


# pd.set_option('display.max_columns', None) # or 1000
# pd.set_option('display.max_rows', None)  # or 1000
# pd.set_option('display.max_colwidth', -1)  # or 199
print(check_output(["ls", "datasets/"]).decode("utf8"))
survey2017 = pd.read_csv('datasets/OSMI Mental Health in Tech Survey 2017.csv').assign(Year=2017)
survey2018 = pd.read_csv('datasets/OSMI Mental Health in Tech Survey 2018.csv').assign(Year=2018)


# Pandas: whats the data row count?
# print(survey2017.shape)
# print(survey2018.shape)
survey = pd.concat([survey2017, survey2018], axis=0, sort=True)
# print(survey.shape)

#missing data
total = survey.isnull().sum().sort_values(ascending=False)
nullValues = survey.isnull().sum()
totalValues = survey.isnull().count()
percent = (nullValues/totalValues).sort_values(ascending=False)
missingData = pd.concat([total, percent*100], axis=1, keys=['Total', 'Percent'])
print(missingData.head(20))

survey = survey.drop(['Network ID', 'Start Date (UTC)', 'Submit Date (UTC)'], axis=1)
survey = survey.drop(['#'], axis=1)

survey.rename(columns={'Have you ever sought treatment for a mental health disorder from a mental health professional?': 'Sought treatment for mental health?',
                       'Would you be willing to talk to one of us more extensively about your experiences with mental health issues in the tech industry? (Note that all interview responses would be used <em>anonymously</em> and only with your permission.)': 'Talk anonymously with interviewer? 2',
                       'Would you be willing to talk to one of us more extensively about your experiences with mental health issues in the tech industry? (Note that all interview responses would be used <em>anonymously</em> and only with your permission.)': 'Talk anonymously with interviewer?',
                       'What is your age?': 'Age',
                       'What is your gender?': 'Gender'},
              inplace=True)

for feature in survey:
    if survey[feature].isnull().sum()>1000:
        # print(feature, 'dropped')
        survey = survey.drop([feature], axis=1)

    else:
        print(feature)
        uniqueList = survey[feature].unique()
        # print(uniqueList)
        print(uniqueList.dtype)

        if np.str in uniqueList:
            # print('str cast')
            survey[feature].fillna('Not Available', coerce=True)
            survey[feature] = survey[feature].astype(str)
            survey[feature] = survey[feature].replace(['', ' ', '\n', '   '], 'NaN')

        if survey[feature].dtype == np.int64:
            # print('numeric cast')
            survey[feature] = pd.to_numeric(survey[feature], errors='coerce').astype(int)
            survey[feature].fillna(0, inplace=True)

        if survey[feature].dtype == np.float64:
            # print('float cast')
            survey[feature] = pd.to_numeric(survey[feature], errors='coerce').astype(float)
            survey[feature].fillna(0, inplace=True)

        if survey[feature].dtype == np.object:
            # print('object cast')
            survey[feature] = survey[feature].astype(str)
            survey[feature] = survey[feature].replace(['', ' ', '\n', '   ', '.'], 'NaN')



try:
    survey['Talk anonymously with interviewer?'] = survey['Talk anonymously with interviewer?']+survey['Would you be willing to talk to one of us more extensively about your experiences with mental health issues in the tech industry? (Note that all interview responses would be used <em>anonymously</em> and only with your permission.)']
except: print('KeyError')
survey = survey.drop(['Would you be willing to talk to one of us more extensively about your experiences with mental health issues in the tech industry? (Note that all interview responses would be used <em>anonymously</em> and only with your permission.)'], axis=1)
survey['Age-range'] = pd.cut(survey['Age'], [0, 20, 30, 65, 100], labels=["0-20", "21-30", "31-65", "66-100"], include_lowest=True)



print(survey.shape)
# print(survey.info())
# print(survey.head(5))

# Made gender groups Gender
male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "Cis Male", "cis male"]
trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]
female_str = ["female", "woman", "femake", "female ", "f", "cis-female", "female/femme", "females", "femail"]

for index, row in survey.iterrows():
    if any(female in str.lower(row['Gender']) for female in female_str):
        survey['Gender'].replace(to_replace=row['Gender'], value='female', inplace=True)
    elif any(male in str.lower(row['Gender']) for male in male_str):
        survey['Gender'].replace(to_replace=row['Gender'], value='male', inplace=True)
    elif any(trans in str.lower(row['Gender']) for trans in trans_str):
        survey['Gender'].replace(to_replace=row['Gender'], value='trans', inplace=True)
    else:
        survey['Gender'].replace(to_replace=row['Gender'], value='undecided', inplace=True)

genderDistribution = survey['Gender'].value_counts()
sns.barplot(x=genderDistribution.index, y=genderDistribution)
plt.show()

ageDistribution = survey['Age']
ageDistribution.plot.hist()
plt.xlabel('Age')
plt.show()
sns.distplot(ageDistribution, bins=10)
plt.show()

# correlation matrix
correlationMatrix = survey.corr()
sns.heatmap(correlationMatrix)
plt.show()

# treatment correlation matrix
k = 9   # number of variables for heatmap
colsByTreatmentDesc = correlationMatrix.nlargest(k, 'Sought treatment for mental health?')
colsDesc = colsByTreatmentDesc['Sought treatment for mental health?'].index
# print(survey[cols].values.T)
correlationMatrixDesc = np.corrcoef(survey[colsDesc].values.T)
sns.set(font_scale=1.25)
sns.heatmap(correlationMatrixDesc, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=colsDesc.values, xticklabels=colsDesc.values)
plt.show()

colsByTreatmentAsc = correlationMatrix.nsmallest(k, 'Sought treatment for mental health?')
colsAsc = colsByTreatmentAsc['Sought treatment for mental health?'].index
correlationMatrixCAsc = np.corrcoef(survey[colsAsc].values.T)
sns.set(font_scale=1.25)
sns.heatmap(correlationMatrixCAsc, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=colsAsc.values, xticklabels=colsAsc.values)
plt.show()

# Separate by treatment or not
g = sns.FacetGrid(survey, col='Sought treatment for mental health?', height=10)
g = g.map(sns.countplot, 'Gender')
plt.show()
menTreatment = len(survey[(survey['Gender'] == 'male') & (survey['Sought treatment for mental health?'] == 1)])
womenTreatment = len(survey[(survey['Gender'] == 'female') & (survey['Sought treatment for mental health?'] == 1)])
print(menTreatment, 'Males requiring treatment')
print(womenTreatment, 'Females requiring treatment')


# Cleaning & Encoding the database for data analysis data
labelDict = {}
for feature in survey:
    # print(feature, 'LOOK HERE')
    le = preprocessing.LabelEncoder()
    le.fit(survey[feature])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    survey[feature] = le.transform(survey[feature])
     # Get labels
    labelKey = feature
    labelValue = [*le_name_mapping]
    labelDict[labelKey] = labelValue


# Let see how many people has been treated
g = sns.countplot(x='Sought treatment for mental health?', data=survey)
g.set_xticklabels(labelDict['Gender'])
plt.title('Total Distribuition by treated or not')
plt.show()

# Probability of mental health condition based on gender
g = sns.catplot(x='Age-range', y='Sought treatment for mental health?', hue="Gender", data=survey, kind="bar",  ci=None, height=5, aspect=2, legend_out = True)
g.set_xticklabels(labelDict['Age-range'])
plt.title('Probability of mental health condition')
plt.ylabel('Probability x 100')
plt.xlabel('Age')
# replace legend labels
for t, l in zip(g._legend.texts, labelDict['Gender']):
    t.set_text(l)
# Positioning the legend
g.fig.subplots_adjust(top=0.9, right=0.8)
plt.show()


g = sns.catplot(x='Talk anonymously with interviewer?', y='Sought treatment for mental health?', hue="Gender", data=survey, kind="bar", ci=None, height=5, aspect=2, legend_out=True)
g.set_xticklabels(labelDict['Talk anonymously with interviewer?'])
plt.title('Probability of mental health condition')
plt.ylabel('Probability x 100')
plt.xlabel('Ability to talk anonymously with interviewer?.')
# replace legend labels
for t, l in zip(g._legend.texts, labelDict['Gender']):
    t.set_text(l)
# Positioning the legend
g.fig.subplots_adjust(top=0.9, right=0.8)
plt.show()

g = sns.catplot(x='Have you ever discussed your mental health with coworkers?', y='Sought treatment for mental health?', hue="Gender", data=survey, kind="bar", ci=None, height=5, aspect=2, legend_out=True)
g.set_xticklabels(labelDict['Have you ever discussed your mental health with coworkers?'])
plt.title('Probability of mental health condition')
plt.ylabel('Probability x 100')
plt.xlabel('Have you ever discussed your mental health with coworkers?')
# replace legend labels
for t, l in zip(g._legend.texts, labelDict['Gender']):
    t.set_text(l)
# Positioning the legend
g.fig.subplots_adjust(top=0.9, right=0.8)
plt.show()

g = sns.catplot(x='How willing would you be to share with friends and family that you have a mental illness?', y='Sought treatment for mental health?', hue="Gender", data=survey, kind="bar", ci=None, height=5, aspect=2, legend_out=True)
g.set_xticklabels(labelDict['How willing would you be to share with friends and family that you have a mental illness?'])
plt.title('Probability of mental health condition')
plt.ylabel('Probability x 100')
plt.xlabel('How willing would you be to share with friends and family that you have a mental illness?')
# replace legend labels
for t, l in zip(g._legend.texts, labelDict['Gender']):
    t.set_text(l)
# Positioning the legend
g.fig.subplots_adjust(top=0.9, right=0.8)
plt.show()

# -------------------------------------------------------------------

# Scaling Age
survey['Age'] = MinMaxScaler().fit_transform(survey[['Age']])
# print(survey['Age'], 'MINMAXSCALER')

# Removing the column 'Sought treatment for mental health?'
colsDesc = colsDesc.delete([0])
# feature_cols = colsDesc
feature_cols = colsDesc.union(colsAsc)

# Define X and Y
X = survey[feature_cols]
y = survey['Sought treatment for mental health?']

# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

# Create dictionaries for final graph
# Use: methodDict['Stacking'] = accuracy_score
methodDict = {}
rmseDict = ()

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=500)

forest.fit(X, y)
# Return the feature importances (the higher, the more important the feature).
importance = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importance)[::-1]
# print(std, 'std ', indices, 'indices ', importance, ' importance')


labels = []
for f in indices:
    labels.append(feature_cols[f])

# Plot the feature importances of the forest
plt.figure(figsize=(12, 8))
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importance[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), labels, rotation='vertical')
plt.xlim([-1, X.shape[1]])
plt.show()

# ------------------------------------------------------------------------


def evalClassModel(model, yTest, yPredClass, plot=False):
    # Classification accuracy: percentage of correct predictions
    # calculate accuracy
    print('Accuracy:', metrics.accuracy_score(yTest, yPredClass))

    # Null accuracy: accuracy that could be achieved by always predicting the most frequent class
    # examine the class distribution of the testing set (using a Pandas Series method)
    print('Null accuracy:\n', yTest.value_counts())

    # calculate the percentage of ones
    print('Percentage of ones:', yTest.mean())

    # calculate the percentage of zeros
    print('Percentage of zeros:', 1 - yTest.mean())

    # Comparing the true and predicted response values
    print('True:', yTest.values[0:25])
    print('Pred:', yPredClass[0:25])

    # Conclusion:
    # Classification accuracy is the easiest classification metric to understand
    # But, it does not tell you the underlying distribution of response values
    # And, it does not tell you what "types" of errors your classifier is making

    # Confusion matrix
    # save confusion matrix and slice into four pieces
    confusion = metrics.confusion_matrix(y_test, yPredClass)
    # [row, column]
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    # visualize Confusion Matrix
    sns.heatmap(confusion, annot=True, fmt="d")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Metrics computed from a confusion matrix
    # Classification Accuracy: Overall, how often is the classifier correct?
    accuracy = metrics.accuracy_score(yTest, yPredClass)
    print('Classification Accuracy:', accuracy)

    # Classification Error: Overall, how often is the classifier incorrect?
    print('Classification Error:', 1 - metrics.accuracy_score(yTest, yPredClass))

    # False Positive Rate: When the actual value is negative, how often is the prediction incorrect?
    false_positive_rate = FP / float(TN + FP)
    print('False Positive Rate:', false_positive_rate)

    # Precision: When a positive value is predicted, how often is the prediction correct?
    print('Precision:', metrics.precision_score(yTest, yPredClass))

    # IMPORTANT: first argument is true values, second argument is predicted probabilities
    print('AUC Score:', metrics.roc_auc_score(yTest, yPredClass))

    # calculate cross-validated AUC
    print('Cross-validated AUC:', cross_val_score(model, X, y, cv=10, scoring='roc_auc').mean())

    ##########################################
    # Adjusting the classification threshold
    ##########################################
    # print the first 10 predicted responses
    # 1D array (vector) of binary values (0, 1)
    print('First 10 predicted responses:\n', model.predict(X_test)[0:10])

    # print the first 10 predicted probabilities of class membership
    print('First 10 predicted probabilities of class members:\n', model.predict_proba(X_test)[0:10])

    # print the first 10 predicted probabilities for class 1
    print(model.predict_proba(X_test)[0:10, 1])

    # store the predicted probabilities for class 1
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    if plot:
        # histogram of predicted probabilities
        # adjust the font size
        plt.rcParams['font.size'] = 12
        # 8 bins
        plt.hist(y_pred_prob, bins=8)

        # x-axis limit from 0 to 1
        plt.xlim(0, 1)
        plt.title('Histogram of predicted probabilities')
        plt.xlabel('Predicted probability of treatment')
        plt.ylabel('Frequency')

    # predict treatment if the predicted probability is greater than 0.3
    # it will return 1 for all values above 0.3 and 0 otherwise
    # results are 2D so we slice out the first column
    y_pred_prob = y_pred_prob.reshape(-1, 1)
    y_pred_class = binarize(y_pred_prob, 0.3)[0]

    # print the first 10 predicted probabilities
    print('First 10 predicted probabilities:\n', y_pred_prob[0:10])

    ##########################################
    # ROC Curves and Area Under the Curve (AUC)
    ##########################################

    # Question: Wouldn't it be nice if we could see how sensitivity and specificity are affected by various thresholds, without actually changing the threshold?
    # Answer: Plot the ROC curve!

    # AUC is the percentage of the ROC plot that is underneath the curve
    # Higher value = better classifier
    roc_auc = metrics.roc_auc_score(yTest, y_pred_prob)

    # IMPORTANT: first argument is true values, second argument is predicted probabilities
    # we pass y_test and y_pred_prob
    # we do not use y_pred_class, because it will give incorrect results without generating an error
    # roc_curve returns 3 objects fpr, tpr, thresholds
    # fpr: false positive rate
    # tpr: true positive rate
    fpr, tpr, thresholds = metrics.roc_curve(yTest, y_pred_prob)
    if plot:
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.rcParams['font.size'] = 12
        plt.title('ROC curve for treatment classifier')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.legend(loc="lower right")
        plt.show()

    # define a function that accepts a threshold and prints sensitivity and specificity
    def evaluate_threshold(threshold):
        # Sensitivity: When the actual value is positive, how often is the prediction correct?
        # Specificity: When the actual value is negative, how often is the prediction correct?print('Sensitivity for ' + str(threshold) + ' :', tpr[thresholds > threshold][-1])
        print('Specificity for ' + str(threshold) + ' :', 1 - fpr[thresholds > threshold][-1])

    # One way of setting threshold
    predict_mine = np.where(y_pred_prob > 0.50, 1, 0)
    confusion = metrics.confusion_matrix(y_test, predict_mine)
    print(confusion)

    return accuracy

def logisticRegression():
    # train a logistic regression model on the training set
    logreg = LogisticRegression(solver='liblinear')
    logreg.fit(X_train, y_train)

    # make class predictions for the testing set
    y_pred_class = logreg.predict(X_test)

    print('########### Logistic Regression ###############')

    accuracy_score = evalClassModel(logreg, y_test, y_pred_class, True)

    # Data for final graph
    methodDict['Log. Regres.'] = accuracy_score * 100


tunedParamsknn = {}
randomizedparams = {}


def tuningRandomizedSearchCV(model, param_dist):
    # Searching multiple parameters simultaneously
    # n_iter controls the number of searches
    rand = RandomizedSearchCV(model, param_dist, cv=10, scoring='accuracy', n_iter=10, random_state=5, iid=True)
    rand.fit(X, y)
    # rand.cv_results_

    # examine the best model
    print('Rand. Best Score: ', rand.best_score_)
    print('Rand. Best Params: ', rand.best_params_)

    return rand.best_params_


def tuningMultParam(knn):
    # Searching multiple parameters simultaneously
    # define the parameter values that should be searched
    k_range = list(range(1, 31))
    weight_options = ['uniform', 'distance']

    # create a parameter grid: map the parameter names to the values that should be searched
    param_grid = dict(n_neighbors=k_range, weights=weight_options)
    # print(param_grid)

    # instantiate and fit the grid
    grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', iid=True)
    grid.fit(X, y)

    # view the complete results
    # print(grid.cv_results_)

    # examine the best model
    print('Multiparam. Best Score: ', grid.best_score_)
    print('Multiparam. Best Params: ', grid.best_params_)

    # resultsUniform = []
    # resultsDistance = []

    # for jk in range(len(grid.cv_results_)):
    #     if grid.cv_results[_
    #     if jk['weights'] == 'uniform':
    #         results.append(grid.cv_results_['mean_test_score'][_])
    # print(results)
    # plot the results
    # plt.plot(k_range, results)
    # plt.xlabel('Value of K for KNN')
    # plt.ylabel('Cross-Validated Accuracy')
    # plt.show()
    return grid.best_params_


def kNearestNeighbors():
    # Calculating the best parameters
    knn = KNeighborsClassifier()

    # define the parameter values that should be searched
    k_range = list(range(1, 31))
    weight_options = ['uniform', 'distance']

    # specify "parameter distributions" rather than a "parameter grid"
    param_dist = dict(n_neighbors=k_range, weights=weight_options)
    tuningRandomizedSearchCV(knn, param_dist)


    # From https://github.com/justmarkham/scikit-learn-videos/blob/master/08_grid_search.ipynb
    # tuningCV()
    # tuningGridSerach(knn)

    tunedParamsknn = tuningMultParam(knn)

    # train a KNeighborsClassifier model on the training set
    # knn = KNeighborsClassifier(weights=tunedParams['weights'], n_neighbors=tunedParams['n_neighbors'])
    knn = KNeighborsClassifier(**tunedParamsknn)
    knn.fit(X_train, y_train)

    # make class predictions for the testing set
    y_pred_class = knn.predict(X_test)

    print('########### KNeighborsClassifier ###############')

    accuracy_score = evalClassModel(knn, y_test, y_pred_class, True)

    # Data for final graph
    methodDict['KNN'] = accuracy_score * 100

def treeClassifier():
    # Calculating the best parameters
    tree = DecisionTreeClassifier()
    featuresSize = feature_cols.__len__()
    param_dist = {"max_depth": [3, None],
                  "max_features": randint(1, featuresSize),
                  "min_samples_split": randint(2, 9),
                  "min_samples_leaf": randint(1, 9),
                  "criterion": ["gini", "entropy"]}
    randomizedparams = tuningRandomizedSearchCV(tree, param_dist)


    # train a decision tree model on the training set
    tree = DecisionTreeClassifier(**randomizedparams)
    tree.fit(X_train, y_train)

    # make class predictions for the testing set
    y_pred_class = tree.predict(X_test)

    print('########### Tree classifier ###############')

    accuracy_score = evalClassModel(tree, y_test, y_pred_class, True)

    # Data for final graph
    methodDict['Tree clas.'] = accuracy_score * 100



def randomForest():
    # Calculating the best parameters
    forest = RandomForestClassifier(n_estimators=100)

    featuresSize = feature_cols.__len__()
    param_dist = {"max_depth": [3, None],
                  "max_features": randint(1, featuresSize),
                  "min_samples_split": randint(2, 9),
                  "min_samples_leaf": randint(1, 9),
                  "criterion": ["gini", "entropy"]}
    randomizedparams = tuningRandomizedSearchCV(forest, param_dist)

    # Building and fitting my_forest
    forest = RandomForestClassifier(**randomizedparams)
    my_forest = forest.fit(X_train, y_train)

    # make class predictions for the testing set
    y_pred_class = my_forest.predict(X_test)

    print('########### Random Forests ###############')

    accuracy_score = evalClassModel(my_forest, y_test, y_pred_class, True)

    # Data for final graph
    methodDict['R. Forest'] = accuracy_score * 100




def bagging():
    # Building and fitting
    bag = BaggingClassifier(DecisionTreeClassifier(), max_samples=1.0, max_features=1.0, bootstrap_features=False)
    bag.fit(X_train, y_train)

    # make class predictions for the testing set
    y_pred_class = bag.predict(X_test)

    print('########### Bagging ###############')

    accuracy_score = evalClassModel(bag, y_test, y_pred_class, True)

    # Data for final graph
    methodDict['Bagging'] = accuracy_score * 100




def boosting():
    # Building and fitting
    clfDecision = DecisionTreeClassifier(criterion='entropy', max_depth=1)
    boost = AdaBoostClassifier(base_estimator=clfDecision, n_estimators=500)
    boost.fit(X_train, y_train)

    # make class predictions for the testing set
    y_pred_class = boost.predict(X_test)

    print('########### Boosting ###############')

    accuracy_score = evalClassModel(boost, y_test, y_pred_class, True)

    # Data for final graph
    methodDict['Boosting'] = accuracy_score * 100




def stacking():
    # Building and fitting
    clf1 = KNeighborsClassifier(**tunedParamsknn)
    clf2 = RandomForestClassifier(**randomizedparams, n_estimators=100)
    clf3 = GaussianNB()
    lr = LogisticRegression(solver='liblinear')
    stack = StackingClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr)
    stack.fit(X_train, y_train)

    # make class predictions for the testing set
    y_pred_class = stack.predict(X_test)

    print('########### Stacking ###############')

    accuracy_score = evalClassModel(stack, y_test, y_pred_class, True)

    # Data for final graph
    methodDict['Stacking'] = accuracy_score * 100


def plotSuccess():
    s = pd.Series(methodDict)

    # Colors
    ax = s.plot(kind='bar',
                width=0.4,
                figsize=(12, 8))
    for p in ax.patches:
        ax.annotate(str(round(p.get_height(), 2)), (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.ylim([50.0, 80.0])
    plt.xlabel('Success')
    plt.ylabel('Percentage')
    plt.title('Success of methods')

    plt.show()


kNearestNeighbors()
logisticRegression()
treeClassifier()
randomForest()
bagging()
boosting()
stacking()
plotSuccess()

# Generate predictions with the best method
clfAda = AdaBoostClassifier()
clfAda.fit(X, y)
dfTestPredictions = clfAda.predict(X_test)

# Write predictions to csv file
# We don't have any significative field so we save the index
# results = pd.DataFrame({'Index': X_test.index, 'Treatment': dfTestPredictions})
# Save to file
# This file will be visible after publishing in the output section
# results.to_csv('results.csv', index=False)
# print(results.head(20))



