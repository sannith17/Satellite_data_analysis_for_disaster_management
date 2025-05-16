import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from tensorflow.keras import models, layers

def apply_pca(X, n_components=100):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return pca, X_pca

def train_svm(X, y):
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(X, y)
    return clf

def build_cnn(input_shape=(64, 64, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
