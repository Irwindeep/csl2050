import matplotlib.pyplot as plt
from data import fetch_lfw_deep_people
from features import compute_hog, calcLBP, extract_cnn_features, resnet
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
DATA_DIR = "../Dataset/lfw-deepfunneled/lfw-deepfunneled"   # Directory containing all the images

# Calling the function to fetch and load images
faces, target, target_names, paths = fetch_lfw_deep_people(DATA_DIR, resize=0.4, min_faces_per_person=40)

print(faces.shape, target.shape, target_names.shape)
h = faces.shape[1]
w = faces.shape[2]

# Plotting the faces
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(faces[i])
    plt.title(target_names[target[i]])
    plt.axis('off')
plt.tight_layout()
plt.show()

# Plotting HoG features
plt.figure(figsize=(10, 4))
for i in range(10):
    hog_f, hog_i = compute_hog(paths[i])
    plt.subplot(2, 5, i + 1)
    plt.imshow(hog_i)
    plt.title(target_names[target[i]])
    plt.axis('off')
plt.tight_layout()
plt.show()

# Plotting the histograms for LBP Features
plt.figure(figsize=(10, 4))
for i in range(10):
    hist_lbp = calcLBP(paths[i])
    plt.subplot(2, 5, i + 1)
    plt.plot(hist_lbp)
    plt.title(target_names[target[i]])
    plt.xlabel("Pixel Value")
plt.tight_layout()
plt.show()

feats = extract_cnn_features(paths[0], resnet)
print(f"CNN features for {paths[0]}:")
print(feats)

# Storing faces as 1-D array
X = faces.reshape(len(faces), -1)
y = target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Fitting data to PCA model
pca = PCA()
pca.fit(X_train)

# Cummulative Variance Ratio
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# Setting the total variance and n_components
target_variance = 0.98
n_comp = np.argmax(cumulative_variance_ratio >= target_variance) + 1

print(f"n_components = {n_comp}")

# Plotting the cummulative variance ratio with n_components
plt.plot(cumulative_variance_ratio)
plt.axvline(x=n_comp, color='red', linestyle='--', label='n_components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio vs. Number of Components')
plt.grid(True)
plt.show()

# Calling another instance of PCA with n_components = n_com
pca = PCA(n_comp)
pca.fit(X_train)

# Transforming the dataset to n_comp dimensions
X_train_t = pca.transform(X_train)
X_test_t = pca.transform(X_test)

fig, axes = plt.subplots(3, 4, figsize=(6, 6))

# plotting first 12 eigenfaces
for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape((h, w)), cmap='gray')
    ax.set_title(f"Eigenface {i+1}")
    ax.axis('off')

plt.tight_layout()
plt.show()

# Fitting the dataset to LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_t, y_train)

# Printing the LDA projection vector
print(f"LDA Projection Vector: {lda.coef_[0]}")

# Transforming the dataset along the LDA projection vector
X_train_t = lda.transform(X_train_t)
X_test_t = lda.transform(X_test_t)

# Writing the transformed dataset into csv files for easy accessibility
with open("./test.csv", "w") as f:
    for i in range(X_test_t.shape[0]):
        for j in range(X_test_t.shape[1]):
            f.write(f"{X_test_t[i][j]},")
        f.write(f"{y_test[i]}\n")

with open("./train.csv", "w") as f:
    for i in range(X_train_t.shape[0]):
        for j in range(X_train_t.shape[1]):
            f.write(f"{X_train_t[i][j]},")
        f.write(f"{y_train[i]}\n")

print(f"Dimensions of transformed data: {X_train_t.shape[1]}")