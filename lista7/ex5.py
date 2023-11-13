import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from yellowbrick.classifier import ConfusionMatrix
from sklearn.compose import ColumnTransformer
from scipy import stats
import numpy as np

data = pd.read_csv("breast-cancer.csv")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

z_scores = np.abs(stats.zscore(X.select_dtypes(include=[np.number])))
outlier_rows = np.where(z_scores > 3)[0]
X = X.drop(outlier_rows)
y = y.drop(outlier_rows)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), X.select_dtypes(include=[np.number]).columns),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), X.select_dtypes(include=['object']).columns)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', MLPClassifier(max_iter=1000, verbose=True, tol=1e-14,
     solver='adam', activation='relu', hidden_layer_sizes=(9)))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Acurácia:", accuracy)

cm = ConfusionMatrix(pipeline)
cm.fit(X_train, y_train)
cm.score(X_test, y_test)

param_grid = {
    'classifier__hidden_layer_sizes': [(10,), (20,), (30,), (10, 5), (20, 10), (30, 15)],
    'classifier__activation': ['relu', 'tanh', 'logistic'],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5,
                           scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Melhores parametros:", best_params)

best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Acurácia do melhor modelo:", accuracy)
