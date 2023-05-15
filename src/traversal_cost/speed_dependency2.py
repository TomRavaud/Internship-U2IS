import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Load the features
road_features = np.load("src/traversal_cost/road.npy")
grass_features = np.load("src/traversal_cost/grass.npy")
sand_features = np.load("src/traversal_cost/sand.npy")

# Concatenate the features
features = np.concatenate((road_features, grass_features, sand_features))

# Array of velocities
velocities = np.array(([0.2]*4 + [0.4]*4 + [0.6]*4 + [0.8]*4 + [1.0]*4)*3)
velocities_unique = list(set(velocities))

# List of labels
labels = np.array(["road"]*20 + ["grass"]*20 + ["sand"]*20)
labels_unique = list(set(labels))

# labels = ["road0.2"]*20 + ["grass"]*20 + ["sand"]*20
# labels_unique = list(set(labels))
# labels = np.array(labels)

# Scale the dataset
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply PCA
method = "pca"

if method == "pca":
    pca = PCA(n_components=2)
    costs = pca.fit_transform(features_scaled)
    
    # Display the coefficients of the first principal component
    plt.matshow(pca.components_, cmap="viridis")
    plt.colorbar()
    plt.xticks(range(12),
               [
                   "var approx [roll]",
                   "var lvl 1 [roll]",
                   "var lvl 2 [roll]",
                   "var lvl 3 [roll]",
                   "var approx [pitch]",
                   "var lvl 1 [pitch]",
                   "var lvl 2 [pitch]",
                   "var lvl 3 [pitch]",
                   "var approx [z acc]",
                   "var lvl 1 [z acc]",
                   "var lvl 2 [z acc]",
                   "var lvl 3 [z acc]",
                ],
               rotation=60,
               ha="left")
    plt.xlabel("Feature")
    plt.ylabel("Principal component 1")
    plt.title("First principal component coefficients")


    
elif method == "tsne":
    tsne = TSNE(random_state=42)
    costs = tsne.fit_transform(features_scaled)

dataframe = pd.DataFrame(costs, columns=["pc1", "pc2"])
dataframe["velocity"] = velocities

plt.figure()

# Terrain classes
plt.subplot(1, 2, 1)
for label in labels_unique:
    indexes_label = labels == label
    
    plt.scatter(dataframe.loc[indexes_label, "pc1"],
                dataframe.loc[indexes_label, "pc2"],
                label=label,
                # c=dataframe.loc[indexes_label, "velocity"],
                )

plt.legend()
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")

# Velocities
plt.subplot(1, 2, 2)
for velocity in velocities_unique:
    indexes_velocity = velocities == velocity
    
    plt.scatter(dataframe.loc[indexes_velocity, "pc1"],
                dataframe.loc[indexes_velocity, "pc2"],
                label=velocity,
                )

plt.legend()
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")

# Traversal cost

# Cartesian
# alpha = 0.5
# costs = alpha*costs[:, 0]/np.max(np.abs(costs[:, 0])) + (1-alpha)*costs[:, 1]/np.max(np.abs(costs[:, 1]))

# Polar
x = costs[:, 0]
x = x - np.min(x)
y = costs[:, 1]
r = np.sqrt(x**2 + y**2)
theta = np.arctan(y/(x+1e-3))
costs = r*np.sin((theta + np.pi/2)/2)


# Transform the costs to make the distribution closer
# to a Gaussian distribution
# normalized_costs = np.log(costs - np.min(costs) + 1)
normalized_costs = costs

# print(normalized_costs.shape)

colors = np.array(["black"]*20 + ["green"]*20 + ["orange"]*20)

plt.figure()
for label in labels_unique:
    indexes_label = labels == label
    plt.scatter(velocities[indexes_label],
                normalized_costs[indexes_label],
                label=label,
                color=colors[indexes_label])

plt.legend()

plt.xlabel("Velocity [m/s]")
plt.ylabel("Traversal cost")

plt.show()
