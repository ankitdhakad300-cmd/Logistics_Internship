
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("week3_output", exist_ok=True)

np.random.seed(42)
n = 200

df = pd.DataFrame({
    "Shipment_ID": range(1, n + 1),
    "Region": np.random.choice(
        ["North", "South", "East", "West"], n
    ),
    "Transport_Mode": np.random.choice(
        ["Road", "Rail", "Air"], n, p=[0.6, 0.25, 0.15]
    ),
    "Distance_km": np.random.randint(50, 1501, n),
    "Shipment_Weight_kg": np.random.randint(10, 1001, n),
    "Shipment_Volume": np.random.randint(1, 101, n)
})

mode_time = df["Transport_Mode"].map({
    "Road": 0.055,
    "Rail": 0.035,
    "Air": 0.012
})

mode_cost = df["Transport_Mode"].map({
    "Road": 8,
    "Rail": 12,
    "Air": 30
})

df["Delivery_Time_hours"] = (
    df["Distance_km"] * mode_time
    + df["Shipment_Weight_kg"] * 0.008
    + np.random.normal(8, 5, n)
).clip(lower=1).round(2)

df["Planned_Delivery_Time_hours"] = (
    df["Distance_km"] * 0.04 + 10
).round(2)

df["Transportation_Cost"] = (
    df["Distance_km"] * mode_cost
    + df["Shipment_Weight_kg"] * 2
    + np.random.normal(100, 200, n)
).clip(lower=50).round(2)

df["Delay_Hours"] = (
    df["Delivery_Time_hours"]
    - df["Planned_Delivery_Time_hours"]
).clip(lower=0).round(2)

df["Delivery_Status"] = np.where(
    df["Delivery_Time_hours"]
    <= df["Planned_Delivery_Time_hours"],
    "On Time",
    "Delayed"
)

df.to_csv("week3_output/logistics_dataset.csv", index=False)

print(df.head())

print(df.shape)

df.info()


print(df.isnull().sum())

print(df.duplicated().sum())

print(df.describe().round(2))

metrics = [
    "Distance_km",
    "Shipment_Weight_kg",
    "Delivery_Time_hours",
    "Transportation_Cost",
    "Delay_Hours"
]

print(df[metrics].agg(["mean", "median", "std"]).round(2))


print(df["Delivery_Status"].value_counts())

print(df.groupby("Transport_Mode")[
    "Transportation_Cost"
].mean().round(2))

print(df.groupby("Region")[
    "Delivery_Time_hours"
].mean().round(2))

correlation = df[metrics].corr().round(2)
print(correlation)


plt.figure(figsize=(8, 5))
plt.hist(df["Delivery_Time_hours"], bins=15, edgecolor="black")
plt.title("Distribution of Delivery Times")
plt.xlabel("Delivery Time (hours)")
plt.ylabel("Number of Shipments")
plt.tight_layout()
plt.savefig("week3_output/1_delivery_time_distribution.png")
plt.show()

avg_cost = df.groupby("Transport_Mode")["Transportation_Cost"].mean().sort_values()

plt.figure(figsize=(8, 5))
avg_cost.plot(kind="bar", edgecolor="black")
plt.title("Average Transportation Cost by Mode")
plt.xlabel("Transport Mode")
plt.ylabel("Average Cost")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("week3_output/2_average_cost_by_mode.png")
plt.show()

status_counts = df["Delivery_Status"].value_counts()

plt.figure(figsize=(7, 5))
status_counts.plot(kind="bar", edgecolor="black")
plt.title("On-Time vs Delayed Shipments")
plt.xlabel("Delivery Status")
plt.ylabel("Number of Shipments")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("week3_output/3_delivery_status.png")
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(df["Distance_km"],df["Delivery_Time_hours"],alpha=0.7)
plt.title("Distance vs Delivery Time")
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (hours)")
plt.tight_layout()
plt.savefig("week3_output/4_distance_vs_delivery_time.png")
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(len(metrics)), metrics, rotation=45, ha="right")
plt.yticks(range(len(metrics)), metrics)

for i in range(len(metrics)):
    for j in range(len(metrics)):
        plt.text(
            j, i, f"{correlation.iloc[i, j]:.2f}",
            ha="center", va="center"
        )

plt.title("Correlation Between Logistics Metrics")
plt.tight_layout()
plt.savefig("week3_output/5_correlation_heatmap.png")
plt.show()

avg_time = df.groupby("Region")["Delivery_Time_hours"].mean().sort_values()

plt.figure(figsize=(8, 5))
avg_time.plot(kind="bar", edgecolor="black")
plt.title("Average Delivery Time by Region")
plt.xlabel("Region")
plt.ylabel("Average Delivery Time (hours)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("week3_output/6_delivery_time_by_region.png")
plt.show()

print("Dataset and six charts saved in the week3_output folder.")
