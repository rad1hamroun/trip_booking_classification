import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Model:
    def __init__(
        self,
        data,
        targets=["booked_flight", "booked_hotel", "booked_rental"],
        categorical_features=["user_device", "user_osName", "user_browserName"],
        numeric_features=["distance", "user_lat", "user_lng", "dest_lat", "dest_lng"],
    ):
        self.data = data
        self.targets = targets
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.y = self.data[targets]
        self.X = self.data.drop(targets, axis=1)
        self.pipeline = None

    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def create_pipeline(self):
        categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")

        numerical_preprocessor = StandardScaler()

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", categorical_preprocessor, self.categorical_features),
                ("num", numerical_preprocessor, self.numeric_features),
            ]
        )

        model = RandomForestClassifier()

        self.pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", model)]
        )

    def fit(self):
        self.pipeline.fit(self.X_train, self.y_train)
        return self.pipeline

    def evaluate(self):
        y_pred = self.pipeline.predict(self.X_test)
        print("Accuracy:", accuracy_score(self.y_test, y_pred))

    def predict(self, X=None):
        if X is None:
            if self.X_test is None:
                X = self.X
            else:
                X = self.X_test
        arr = self.pipeline.predict(X)
        df = pd.DataFrame(arr, columns=self.targets)
        return df

    def build_model_and_evaluate(self):
        self.train_test_split()
        self.create_pipeline()
        self.fit()
        self.evaluate()
