import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import traceback

class TRAINING:
    def __init__(self, data):
        try:
            # Check if the input is a DataFrame or a file path
            if isinstance(data, pd.DataFrame):
                self.df = data
            else:
                self.df = pd.read_csv(data)

            # Map categorical columns to numeric values
            self.df['sex'] = self.df['sex'].map({'Female': 0, 'Male': 1})
            self.df['smoker'] = self.df['smoker'].map({'No': 0, 'Yes': 1})
            self.df["day"] = self.df['day'].map({"Sun": 0, "Thur": 1, "Fri": 2, "Sat": 3})
            self.df['time'] = self.df['time'].map({'Dinner': 0, 'Lunch': 1})

            # Split the data into X (independent) and y (dependent)
            self.x = self.df.drop(columns=['total_bill'])  # independent variables
            self.y = self.df['total_bill']  # dependent variable

            # Perform train-test split
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.x, self.y, test_size=0.20, random_state=12
            )
        except Exception:
            print("An error occurred during initialization:")
            traceback.print_exc()  # Display the full error traceback

    def checking_training_performance(self):
        try:
            lr = LinearRegression()
            lr.fit(self.x_train, self.y_train)
            self.y_pred = lr.predict(self.x_train)
            self.acc = r2_score(self.y_train, self.y_pred)
            print(f"Training R2 Score: {self.acc:.2f}")
        except Exception:
            print("An error occurred during training performance evaluation:")
            traceback.print_exc()  # Display the full error traceback
    def checking_testing_performance(self):
        try:
            lr = LinearRegression()
            lr.fit(self.x_test, self.y_test)
            self.y_pred_1 = lr.predict(self.x_test)
            self.acc_1 = r2_score(self.y_test, self.y_pred_1)
            print(f"Training R2 Score: {self.acc_1:.2f}")
        except Exception:
            print("An error occurred during training performance evaluation:")
            traceback.print_exc()  # Display the full error traceback        
                

if __name__ == "__main__":
    try:
        # Load the tips dataset from seaborn
        data = sns.load_dataset('tips')
        # Pass the dataset to the class
        obj = TRAINING(data)  # Constructor will be called
        obj.checking_training_performance()
        obj.checking_testing_performance()
    except Exception:
        print("An error occurred in the main block:")
        traceback.print_exc()  # Display the full error traceback
