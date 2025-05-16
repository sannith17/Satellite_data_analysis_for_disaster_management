import streamlit as st
import numpy as np
from PIL import Image
from model_utils import apply_pca, train_svm, build_cnn

st.set_page_config(layout="wide")
st.title("Satellite Image Classifier (PCA + SVM + CNN)")

uploaded_file = st.file_uploader("Upload a satellite image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).resize((64, 64))
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    img_array = np.array(image).astype('float32') / 255.0
    img_flat = img_array.flatten().reshape(1, -1)

    with st.spinner("Running PCA + SVM + CNN..."):
        pca, img_pca = apply_pca(img_flat)
        clf = train_svm(img_pca, [0])  # Dummy training
        svm_pred = clf.predict(img_pca)
        svm_prob = clf.predict_proba(img_pca)[0]

        cnn_model = build_cnn()
        cnn_pred = cnn_model.predict(img_array.reshape(1, 64, 64, 3))[0][0]

    st.subheader("Predictions")
    st.metric("SVM Prediction", int(svm_pred[0]))
    st.metric("SVM Probability", f"{svm_prob[int(svm_pred[0])]*100:.2f}%")
    st.metric("CNN Probability", f"{cnn_pred*100:.2f}%")
