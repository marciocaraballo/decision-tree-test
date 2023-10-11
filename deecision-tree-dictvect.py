import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import DictVectorizer

# Encode all feature values as numbers based on the json key-pair structure
vec = DictVectorizer(sparse=False, sort=True)

df = pd.read_json('./dataset-genre-array.json')

# Ignore some columns
x = df.drop(columns=["like", "summary_text",
            "poster_url", "name"], axis="columns")

# Builds a dataset based on dict format
x = vec.fit_transform(x.to_dict('records'))

# Target ("yes" / "no")
y = df["like"]

# split the dataset
x_train, test_x, y_train, y_test = train_test_split(
    x, y, test_size=0.2)

decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree = decision_tree.fit(x_train, y_train)
r = export_text(decision_tree, feature_names=vec.feature_names_, max_depth=100)

print("\nDecision tree scructure\n")
print(r)

# Predict based on the split test dataset
res_pred = decision_tree.predict(test_x)

# Inverse transform does not produce exact dataset

test_x = pd.DataFrame.from_dict(vec.inverse_transform(test_x)).gt(0)

print("\nTest set prediction\n")
print(test_x)

score = accuracy_score(y_test, res_pred)
print("\nPrediction accuracy: " + str(score))

# Original dataset with predicted value
df_original = df.drop(columns=["summary_text", "poster_url",
                               "cast"], axis="columns").iloc[y_test.index]
df_original["predicted_like"] = res_pred

print("\nOriginal dataset with prediction\n")
print(df_original[["origin", "language", "name", "year", "certificate", "runtime",
      "genre", "ratingValue", "ratingCount", "director", "like", "predicted_like"]])
