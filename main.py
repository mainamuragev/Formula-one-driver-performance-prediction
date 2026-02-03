import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def load_data():
    # Placeholder dataset: driver stats
    data = pd.DataFrame({
        "Driver": ["Hamilton", "Verstappen", "Leclerc", "Norris"],
        "Wins": [11, 12, 3, 2],
        "Podiums": [17, 18, 10, 9],
        "Points": [387, 395, 280, 260]
    })
    return data

def train_model(data):
    # Simple regression: predict Points from Wins + Podiums
    X = data[["Wins", "Podiums"]]
    y = data["Points"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    return model, mse

if __name__ == "__main__":
    df = load_data()
    model, mse = train_model(df)
    print("Model trained. MSE:", mse)
