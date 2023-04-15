import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans 

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

df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()

predictor_columns = list(df_temporal.columns[:6]) + ["acc_r", "gyr_r"]

ws = int(1000/200)

for col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "mean")
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "std")

df_temporal_list = []
for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy()
    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], ws, "std")
    df_temporal_list.append(subset)

df_temporal = pd.concat(df_temporal_list)

df_temporal.info()

subset[["acc_x", "acc_x_temp_mean_ws_5", "acc_x_temp_std_ws_5"]].plot()
subset[["gyr_x", "gyr_x_temp_mean_ws_5", "gyr_x_temp_std_ws_5"]].plot()


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

df_freq = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()

fs = int(1000/200)
ws = int(2800/200)

df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], ws, fs)
df_freq.info()

# visualize results

subset = df_freq[df_freq["set"] == 15]
subset[["acc_y"]].plot()
subset[
    [
        "acc_y_freq_0.0_Hz_ws_14",
        "acc_y_freq_0.357_Hz_ws_14",
        "acc_y_freq_0.714_Hz_ws_14",
        "acc_y_freq_1.071_Hz_ws_14",
        "acc_y_freq_1.429_Hz_ws_14",
    ]
].plot()

# apply frequency abstraction to all sets
df_freq = df_temporal.copy().reset_index()
df_freq_list = []
for s in df_freq["set"].unique():
    print(f"applying frequency abstraction to set {s}")
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list)
df_freq.set_index("epoch (ms)", drop = True)

df_freq.info()

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

df_freq = df_freq.dropna()
df_freq = df_freq.iloc[::2].set_index("epoch (ms)", drop = True)

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

df_cluster = df_freq.copy()

cluster_columns = ["acc_x", "acc_y", "acc_z"]
k_values = range(2, 10)
inertias = []

for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init = 20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)

plt.figure(figsize = (10, 10))
plt.plot(k_values, inertias)
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.show()

kmeans = KMeans(n_clusters=5, n_init = 20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

fig = plt.figure(figsize = (15, 15))
ax = fig.add_subplot(projection='3d')
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label = c)
ax.set_xlabel("acc_x")
ax.set_ylabel("acc_y")
ax.set_zlabel("acc_z")
plt.legend()
plt.show()

fig = plt.figure(figsize = (15, 15))
ax = fig.add_subplot(projection='3d')
for l in df_cluster["lable"].unique():
    subset = df_cluster[df_cluster["lable"] == l]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label = l)
ax.set_xlabel("acc_x")
ax.set_ylabel("acc_y")
ax.set_zlabel("acc_z")
plt.legend()
plt.show()



# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle("../../data/interim/03_data_features.pkl")