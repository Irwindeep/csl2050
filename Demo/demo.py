from joblib import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

pca = load('pca.joblib')
lda = load('lda.joblib')
gnb = load('gnb.joblib')
knn = load('knn.joblib')
forest = load('random_forest.joblib')
svm = load('svm.joblib')

target_names = np.load('target_names.npy')

img_path = input("Provide path to the image: ")

pil_img = Image.open(img_path)
pil_img = pil_img.resize((100, 100))
face = np.asarray(pil_img, dtype=np.float32)

face /= 255.0
plt.imshow(face)
plt.show()
face = face.mean(axis=2)

face = face.reshape(-1)

face_pca = pca.transform([face])
face_lda = lda.transform(face_pca)

print(f"Naive Bayes Prediction: {target_names[gnb.predict(face_lda)][0]}")
print(f"KNN Prediction: {target_names[knn.predict(face_lda)][0]}")
print(f"Random Forest Prediction: {target_names[forest.predict(face_lda)][0]}")
print(f"SVC Prediction: {target_names[svm.predict(face_lda)][0]}")

