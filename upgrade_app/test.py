import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib


def score_classifier(dataset, classifier, labels):
    kf = KFold(n_splits=3, random_state=50, shuffle=True)
    confusion_mat = np.zeros((2, 2))
    recall = 0
    for training_ids, test_ids in kf.split(dataset):
        training_set = dataset[training_ids]
        training_labels = labels[training_ids]
        test_set = dataset[test_ids]
        test_labels = labels[test_ids]
        classifier.fit(training_set, training_labels)
        predicted_labels = classifier.predict(test_set)
        confusion_mat += confusion_matrix(test_labels, predicted_labels)
        recall += recall_score(test_labels, predicted_labels)
    recall /= 3
    print(confusion_mat)
    print(recall)
    return recall


# Load dataset
df = pd.read_csv("nba_logreg.csv")

names = df['Name'].values.tolist()
labels = df['TARGET_5Yrs'].values
paramset = df.drop(['TARGET_5Yrs', 'Name'], axis=1).columns.values
df_vals = df.drop(['TARGET_5Yrs', 'Name'], axis=1).values

for x in np.argwhere(np.isnan(df_vals)):
    df_vals[x] = 0.0

X = MinMaxScaler().fit_transform(df_vals)

best_recall = 0
best_classifier = None

# Train and evaluate with RandomForestClassifier
for n_estimators in [50, 100, 150]:
    print(f"Number of Estimators: {n_estimators}")
    classifier_rf = RandomForestClassifier(n_estimators=n_estimators)
    recall = score_classifier(X, classifier_rf, labels)

    if recall > best_recall:
        best_recall = recall
        best_classifier = classifier_rf

# Save the best classifier to a file
joblib.dump(best_classifier, 'best_model.joblib')
