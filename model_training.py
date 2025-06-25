import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from assets_data_prep import prepare_data

# שלב 1: קריאת הנתונים
df = pd.read_csv("train.csv")

# שלב 2: עיבוד הנתונים
processed_df = prepare_data(df, mode="train")

# שלב 3: הפרדה ל־X ו־y
X = processed_df.drop("price", axis=1)
y = processed_df["price"]

# שלב 4: חלוקה ל־Train/Test (לצורך הערכה בלבד)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# שלב 5: הגדרת Grid Search
param_grid = {
    'alpha': [0.1, 1, 10],
    'l1_ratio': [0.1, 0.5, 1.0],
}

grid_search = GridSearchCV(
    ElasticNet(max_iter=10000),
    param_grid,
    cv=10,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# שלב 6: המודל האופטימלי
best_model = grid_search.best_estimator_

# שלב 7: שמירת המודל המאומן
with open("trained_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
