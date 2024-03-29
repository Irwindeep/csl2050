import time
from data import fetch_lfw_deep_people
import matplotlib.pyplot as plt
from features import compute_hog, calcLBP
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
DATA_DIR = "../Dataset/lfw-deepfunneled/lfw-deepfunneled"

start_time = time.time()

faces, target, target_names, paths = fetch_lfw_deep_people(DATA_DIR, resize=0.4, min_faces_per_person=20)

print(faces.shape, target.shape, target_names.shape)
h = faces.shape[1]
w = faces.shape[2]

plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(faces[i])
    plt.title(target_names[target[i]])
    plt.axis('off')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
for i in range(10):
    hog_f, hog_i = compute_hog(paths[i])
    plt.subplot(2, 5, i + 1)
    plt.imshow(hog_i)
    plt.title(target_names[target[i]])
    plt.axis('off')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
for i in range(10):
    hist_lbp = calcLBP(paths[i])
    plt.subplot(2, 5, i + 1)
    plt.plot(hist_lbp)
    plt.title(target_names[target[i]])
    plt.xlabel("Pixel Value")
plt.tight_layout()
plt.show()

X = faces.reshape(len(faces), -1)
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

pca = PCA()
pca.fit(X_train)

cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

target_variance = 0.98
n_comp = np.argmax(cumulative_variance_ratio >= target_variance) + 1

plt.plot(cumulative_variance_ratio)
plt.axvline(x=n_comp, color='red', linestyle='--', label='n_components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio vs. Number of Components')
plt.grid(True)
plt.show()

pca = PCA(n_comp)
pca.fit(X_train)
X_train_t = pca.transform(X_train)
X_test_t = pca.transform(X_test)

n_rows = 3
n_cols = 4
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape((h, w)), cmap='gray')
    ax.set_title(f"Eigenface {i+1}")
    ax.axis('off')

plt.tight_layout()
plt.show()

lda = LinearDiscriminantAnalysis()
lda.fit(X_train_t, y_train)

X_train_t = lda.transform(X_train_t)
X_test_t = lda.transform(X_test_t)

with open("../test.csv", "w") as f:
    for i in range(X_test_t.shape[0]):
        for j in range(X_test_t.shape[1]):
            f.write(f"{X_test_t[i][j]},")
        f.write(f"{y_test[i]}\n")

with open("../train.csv", "w") as f:
    for i in range(X_train_t.shape[0]):
        for j in range(X_train_t.shape[1]):
            f.write(f"{X_train_t[i][j]},")
        f.write(f"{y_train[i]}\n")
