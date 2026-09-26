import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("cleaned_data.csv")

X = df.drop("Survived", axis=1)
y = df["Survived"]

X = pd.get_dummies(X, drop_first=True)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

logistic_model = LogisticRegression(max_iter=1000)
tree_model = DecisionTreeClassifier(random_state=42)

logistic_scores = cross_val_score(logistic_model, X, y, cv=cv, scoring="accuracy")
tree_scores = cross_val_score(tree_model, X, y, cv=cv, scoring="accuracy")

print("Logistic Regression Scores:", logistic_scores)
print("Logistic Regression Mean:", logistic_scores.mean())
print("Logistic Regression Standard Deviation:", logistic_scores.std())

print()

print("Decision Tree Scores:", tree_scores)
print("Decision Tree Mean:", tree_scores.mean())
print("Decision Tree Standard Deviation:", tree_scores.std())