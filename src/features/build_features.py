import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction


# -------------------------------------------------------------- 
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/02_removed_outliers_chauvenets_df.pkl")

predictor_columns = list(df.columns[:6])

# plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 7)
plt.rcParams["figure.dpi"] = 200
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

df.info()

df[df["set"] == 2]["acc_x"].plot()

for col in predictor_columns:
    df[col] = df[col].interpolate(method="linear")

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

df[df["set"] == 25]["acc_y"].plot()
df[df["set"] == 50]["acc_y"].plot()

duration = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]
duration = duration.seconds

for s in df["set"].unique():

    start = df[df["set"] == s].index[0]
    end = df[df["set"] == s].index[-1]
    
    duration = end - start

    df.loc[df["set"] == s, "duration"] = duration.seconds

duration_df = df.groupby(["category"])["duration"].mean()

duration_df.iloc[0]/5
duration_df.iloc[1]/10

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()
LowPass = LowPassFilter()

fs = 1000/200
cutoff = 1.2

df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff)

subset = df_lowpass[df_lowpass["set"] == 45]

fig, ax = plt.subplots(nrows = 2, sharex = True, figsize = (20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), fancybox=True, shadow=True)

for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]
    
for col in predictor_columns:
    df_lowpass[df_lowpass["set"] == 45][col].reset_index(drop = True).plot()
    plt.show()

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

plt.figure(figsize = (20, 10))
plt.plot(range(1, len(pc_values) + 1), pc_values)
plt.xlabel("Number of principal components")
plt.ylabel("Explained variance")
plt.show()

df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

subset = df_pca[df_pca["set"] == 35]
subset[["pca_1", "pca_2", "pca_3"]].reset_index(drop=True).plot()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_squared = df_pca.copy()

acc_r = np.sqrt(df_squared["acc_x"]**2 + df_squared["acc_y"]**2 + df_squared["acc_z"]**2)
gyr_r = np.sqrt(df_squared["gyr_x"]**2 + df_squared["gyr_y"]**2 + df_squared["gyr_z"]**2)

df_squared["acc_r"] = acc_r
df_squared["gyr_r"] = gyr_r

df_squared[df_squared["set"] == 35][["acc_r", "gyr_r"]].reset_index(drop=True).plot()
# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------



# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------



# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------



# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------



# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------