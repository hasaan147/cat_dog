import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input
import tensorflow as tf
import streamlit as st

# Streamlit configurations
st.title("Cat and Dog Image Classification Using CNN")

# Function to extract the zip file
def extract_zip(zip_file_path, extract_to_path):
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_path)
    except zipfile.BadZipFile:
        st.error("Error: The file is not a zip file or it is corrupted.")

# Function to read images and labels from the specified folder
def read_images_and_labels_from_folder(folder_path, num_images=150):
    image_paths = []
    labels = []
    images = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                file_path = os.path.join(root, file)
                try:
                    image = Image.open(file_path)
                    images.append(image)
                    
                    label = 0 if 'cat' in file.lower() else 1
                    
                    image_paths.append(file_path)
                    labels.append(label)
                    
                    if len(images) >= num_images:
                        break
                except Exception as e:
                    st.error(f"Error processing file {file_path}: {e}")
        if len(images) >= num_images:
            break
    
    return images, image_paths, labels

# Function to extract features from images
def extract_features(images):
    features = []
    for image in tqdm(images):
        img = image.resize((128, 128), Image.LANCZOS)
        img = img_to_array(img)
        features.append(img)
        
    features = np.array(features)
    features = features.reshape(len(features), 128, 128, 3)
    return features

# File uploader
uploaded_file = st.file_uploader("Choose a zip file containing the dataset", type="zip")

if uploaded_file is not None:
    with st.spinner('Extracting zip file...'):
        extract_to_path = 'extracted'
        os.makedirs(extract_to_path, exist_ok=True)
        extract_zip(uploaded_file, extract_to_path)
        st.success('Zip file extracted successfully')

    dataset_folder_path = extract_to_path  # Adjust the path if necessary
    if os.path.exists(dataset_folder_path):
        with st.spinner('Reading images and labels...'):
            images, image_paths, labels = read_images_and_labels_from_folder(dataset_folder_path)
            st.success('Images and labels read successfully')

        df = pd.DataFrame()
        df['image'], df['label'] = image_paths, labels
        label_dict = {0: 'Cat', 1: 'Dog'}

        if st.checkbox('Show Label Distribution'):
            sns.countplot(df['label'])
            plt.xlabel('Label')
            plt.ylabel('Count')
            plt.title('Label Distribution')
            st.pyplot()

        with st.spinner('Extracting features...'):
            X = extract_features(images)
            X = X / 255.0
            y = np.array(df['label'])
            st.success('Features extracted successfully')

        # Define the model
        input_shape = (128, 128, 3)
        inputs = Input((input_shape))
        conv_1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
        maxp_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
        conv_2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(maxp_1)
        maxp_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
        conv_3 = Conv2D(128, kernel_size=(3, 3), activation='relu')(maxp_2)
        maxp_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)
        conv_4 = Conv2D(256, kernel_size=(3, 3), activation='relu')(maxp_3)
        maxp_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)

        flatten = Flatten()(maxp_4)
        dense_1 = Dense(256, activation='relu')(flatten)
        dropout_1 = Dropout(0.4)(dense_1)
        output_1 = Dense(1, activation='sigmoid', name='label_out')(dropout_1)

        model = Model(inputs=[inputs], outputs=[output_1])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        if st.checkbox('Train Model'):
            with st.spinner('Training model...'):
                history = model.fit(x=X, y=y, batch_size=32, epochs=10, validation_split=0.2)
                st.success('Model trained successfully')

            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            epochs = range(len(acc))

            plt.plot(epochs, acc, 'b', label='Training Accuracy')
            plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
            plt.title('Accuracy Graph')
            plt.legend()
            st.pyplot()

            loss = history.history['loss']
            val_loss = history.history['val_loss']

            plt.plot(epochs, loss, 'b', label='Training Loss')
            plt.plot(epochs, val_loss, 'r', label='Validation Loss')
            plt.title('Loss Graph')
            plt.legend()
            st.pyplot()

        image_index = st.slider('Select Image Index for Prediction', 0, len(df)-1, 0)
        if st.button('Predict'):
            st.write("Original Label:", label_dict[y[image_index]])
            pred = model.predict(X[image_index].reshape(1, 128, 128, 3))
            pred_label = label_dict[round(pred[0][0])]
            st.write("Predicted Label:", pred_label)
            if pred_label == 'Cat':
                st.write("The predicted image is of a cat.")
            else:
                st.write("The predicted image is of a dog.")
            plt.axis('off')
            plt.imshow(X[image_index].reshape(128, 128, 3))
            st.pyplot()
