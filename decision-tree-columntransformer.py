import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer

# For this example, genre is divided in 3 columns, as a
# movie can have multiple genres
df = pd.read_json('./dataset-multiple-genre.json')

# Ignore some columns
x = df.drop(columns=["like", "summary_text",
            "poster_url", "cast"], axis="columns")

numeric_features = ["year", "runtime", "ratingValue", "ratingCount"]
categorical_features = ["origin", "language",
                        "certificate", "director", "genre1", "genre2", "genre3"]

ct = ColumnTransformer([("num", MinMaxScaler(), numeric_features),
                       ("cat", OrdinalEncoder(), categorical_features)], verbose_feature_names_out=False)

# Fit column transformer pre processor
x = ct.fit_transform(x)

# Target ("yes" / "no")
y = df["like"]

# Split the dataset
x_train, test_x, y_train, y_test = train_test_split(
    x, y, test_size=0.2)

# Build decision tree classifier
decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree = decision_tree.fit(x_train, y_train)
r = export_text(decision_tree, feature_names=ct.get_feature_names_out())

print("\nDecision tree scructure\n")
print(r)

# Predict based on the split test dataset
res_pred = decision_tree.predict(test_x)

test_x_final = pd.DataFrame(test_x, columns=ct.get_feature_names_out())
test_x_final['predictited_like'] = res_pred

print("\nTest set prediction\n")
print(test_x_final)

# Evaluate accuracy of the model
score = accuracy_score(y_test, res_pred)
print("\nPrediction accuracy: " + str(round(score*100, 2)) + "%")

# Original dataset with predicted value
df_original = df.drop(columns=["summary_text", "poster_url",
                               "cast"], axis="columns").iloc[y_test.index]
df_original["predicted_like"] = res_pred

print("\nOriginal dataset with prediction\n")
print(df_original[["origin", "language", "name", "year", "certificate", "runtime",
      "genre1", "genre2", "genre3", "ratingValue", "ratingCount", "director", "like", "predicted_like"]])
