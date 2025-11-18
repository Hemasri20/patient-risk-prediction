import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def train_model():
    df = pd.read_csv("data/patients.csv")
    X = df.drop("risk", axis=1)
    y = df["risk"]

    model = Sequential([
        Dense(16, activation="relu", input_shape=(X.shape[1],)),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=10, batch_size=4, verbose=0)

    return model
