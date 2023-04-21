import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")
df = df[df["lable"] != "rest"]

acc_r = df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2
gyr_r = df["gyr_x"] ** 2 + df["gyr_y"] ** 2 + df["gyr_z"] ** 2
df["acc_r"] = np.sqrt(acc_r)
df["gyr_r"] = np.sqrt(gyr_r)

# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------

bench_df = df[df["lable"] == "bench"]
squat_df = df[df["lable"] == "squat"]
row_df = df[df["lable"] == "row"]
ohp_df = df[df["lable"] == "ohp"]
dead_df = df[df["lable"] == "dead"]

# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------

plot_df = bench_df
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_x"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_y"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_z"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_r"].plot()

plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_x"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_y"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_z"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_r"].plot()

# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------

bench_set = bench_df[bench_df["set"] == bench_df["set"].unique()[0]]
squat_set = squat_df[squat_df["set"] == squat_df["set"].unique()[0]]
row_set = row_df[row_df["set"] == row_df["set"].unique()[0]]
ohp_set = ohp_df[ohp_df["set"] == ohp_df["set"].unique()[0]]
dead_set = dead_df[dead_df["set"] == dead_df["set"].unique()[0]]

bench_set["acc_r"].plot()

LowPass = LowPassFilter()

col = "acc_r"
fs = 1000/200
cutoff = 0.4
order = 10
LowPass.low_pass_filter(
    bench_set, col, fs, cutoff, order
)[col + "_lowpass"].plot()

# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------

def count_repetitions(data, col = "acc_r", fs = 5, cutoff = 0.4, order = 10):
    data = LowPass.low_pass_filter(
        data, col, fs, cutoff, order
    )
    index = argrelextrema(data[col + "_lowpass"].values, np.greater)
    peaks = data.iloc[index]

    fig, ax = plt.subplots()
    plt.plot(data[col + "_lowpass"])
    plt.plot(peaks[col + "_lowpass"], "o", color="red")
    ax.set_ylabel(col + "_lowpass")
    exercise = data["lable"].iloc[0].title()
    category = data["category"].iloc[0].title()
    plt.title(f"{exercise} {category}: {len(peaks)} repetitions")
    plt.show()

    return len(peaks)

count_repetitions(bench_set, cutoff = 0.4)
count_repetitions(squat_set, cutoff = 0.35)
count_repetitions(row_set, cutoff = 0.65, col = "gyr_x")
count_repetitions(ohp_set, cutoff = 0.35)
count_repetitions(dead_set, cutoff = 0.4)

# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------






# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------

df["reps"] = df["category"].apply(lambda x: 5 if x == "heavy" else 10)
rep_df = df.groupby(["lable", "category", "set"])["reps"].max().reset_index()

for s in df["set"].unique():
    subset = df[df["set"] == s]

    col = "acc_r"
    cutoff = 0.4

    if subset["lable"].iloc[0] == "row":
        cutoff = 0.65
        col = "gyr_x"

    if subset["lable"].iloc[0] == "squat":
        cutoff = 0.35

    if subset["lable"].iloc[0] == "ohp":
        cutoff = 0.35

    reps = count_repetitions(subset, col = col, cutoff = cutoff)
    rep_df.loc[rep_df["set"] == s, "reps_pred"] = reps

# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------

error = mean_absolute_error(rep_df["reps"], rep_df["reps_pred"]).round(2)
rep_df.groupby(["lable", "category"])["reps", "reps_pred"].mean().plot.bar()