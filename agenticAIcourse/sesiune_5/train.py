import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
 
data = load_iris()
X = data.data      # features: (150, 4) — sepal/petal length & width
y = data.target    # labels: 0, 1, 2 (3 specii)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

#model = LinearRegression()  
model = LogisticRegression()
#model = RandomForestClassifier()
model.fit(X_train, y_train)


#Salvare model
dump(model, "model.joblib")

#Incarcare model
loaded_model = load("model.joblib")

prediction = loaded_model.predict([[5.1, 3.5, 1.4, 0.2]])
score = model.score(X_test, y_test)
print(f"R² score: {score:.3f}")

print(data.feature_names)  # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print("X train ", X_train.shape)
print("X test", X_test.shape)
print("y train ", y_train.shape)
print("y test", y_test.shape)
print("prediction", prediction)