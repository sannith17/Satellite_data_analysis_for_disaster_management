import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from PIL import Image
import io

st.set_page_config(page_title="Satellite Water Detection Dashboard", layout="wide")
st.title("🌍 Satellite Water Detection Dashboard")

st.sidebar.header("Upload Dataset or Image")
data_option = st.sidebar.radio("Choose Input Type:", ("Upload CSV Dataset", "Upload Satellite Image"))

if data_option == "Upload CSV Dataset":
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Dataset")
        st.dataframe(df.head())

        if st.checkbox("Run PCA & SVM Model"):
            target_column = st.selectbox("Select Target Column", df.columns)
            X = df.drop(columns=[target_column])
            y = df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            st.write("#### PCA Visualization")
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_train)
            pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
            pca_df['target'] = y_train.values
            fig, ax = plt.subplots()
            sns.scatterplot(x='PC1', y='PC2', hue='target', data=pca_df, ax=ax)
            st.pyplot(fig)

            st.write("#### SVM Classification")
            clf = SVC()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            st.text("Classification Report")
            st.text(classification_report(y_test, y_pred))
            st.text("Confusion Matrix")
            st.write(confusion_matrix(y_test, y_pred))

elif data_option == "Upload Satellite Image":
    uploaded_image = st.sidebar.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Satellite Image", use_column_width=True)

        if st.checkbox("Run CNN Model on Image"):
            st.write("### CNN-Based Water Detection")
            image = image.resize((64, 64))
            image_array = np.array(image) / 255.0
            if len(image_array.shape) == 2:
                image_array = np.stack((image_array,) * 3, axis=-1)
            image_array = np.expand_dims(image_array, axis=0)

            # Dummy CNN model for demonstration
            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
                MaxPooling2D(2, 2),
                Flatten(),
                Dense(64, activation='relu'),
                Dense(2, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            # Mock prediction (you can load your trained model instead)
            prediction = model.predict(image_array)
            st.write("Predicted Class:", "Water" if np.argmax(prediction) == 1 else "Non-Water")

st.sidebar.markdown("---")
st.sidebar.write("Built with ❤️ using Streamlit")
