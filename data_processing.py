import pandas as pd

#using the wisconsin breast cancer dataset

def prep_data():
    df = pd.read_csv('data.csv')

    df = df.iloc[:545]

    y_labels = df["diagnosis"]

    x_input = df.drop(["id", "Unnamed: 32", "diagnosis"], axis=1)

    x_input = (x_input - x_input.min()) / (x_input.max() - x_input.min())

    y_labels = y_labels.map({"M": 0, "B": 1})

    return x_input.to_numpy().T, y_labels.to_numpy().reshape(1, -1)

print (prep_data())