
import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Order_ID": [101, 102, 103, 104, 105, 106],
    "Distance_KM": [10, 25, 15, 40, 20, 35],
    "Expected_Delivery_Hours": [2, 4, 3, 6, 3, 5],
    "Actual_Delivery_Hours": [2, 5, 2, 8, 4, 5]
}

df = pd.DataFrame(data)

df["Delay_Hours"] = (
    df["Actual_Delivery_Hours"]
    - df["Expected_Delivery_Hours"]
)

print(df)

print(df["Actual_Delivery_Hours"].mean(), "hours")

print(df[df["Delay_Hours"] > 0])

print(df.loc[df["Delay_Hours"].idxmax()])

plt.bar(df["Order_ID"].astype(str),df["Actual_Delivery_Hours"])

plt.xlabel("Order ID")
plt.ylabel("Actual Delivery Time (Hours)")
plt.title("Logistics Delivery Performance")
plt.tight_layout()
plt.show()
