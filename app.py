import streamlit as st
import numpy as np
import os
from PIL import Image
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns

def download_images(category, num_images, st_progress, st_text):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    url = f"https://commons.wikimedia.org/wiki/Category:{category}"
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    if not os.path.exists(category):
        os.makedirs(category)

    image_elements = soup.find_all("img")
    image_elements = [img for img in image_elements if int(img.get('width', 0)) > 50 and int(img.get('height', 0)) > 50]

    for i, img in enumerate(image_elements[:num_images]):
        img_url = urljoin(url, img['src'])
        img_response = requests.get(img_url, headers=headers)

        if img_response.status_code == 200:
            img_data = img_response.content
            img_name = os.path.join(category, f"{category}_{i+1}.jpg")

            with open(img_name, 'wb') as f:
                f.write(img_data)

            st_text.text(f"Image {i+1} downloaded and saved as {img_name}")
        else:
            st_text.text(f"Failed to fetch image {i+1}: HTTP Status Code {img_response.status_code}")

    st_text.text(f"{num_images} images from category {category} downloaded successfully!")

def load_images_from_directory(base_path, categories, target_size=(224, 224)):
    images = []
    labels = []

    for category in categories:
        category_path = os.path.join(base_path, category)
        if os.path.isdir(category_path):
            for img_name in os.listdir(category_path):
                if img_name.endswith(".jpg"):
                    img_path = os.path.join(category_path, img_name)
                    img = Image.open(img_path)
                    img = img.resize(target_size)
                    img_array = np.array(img)
                    images.append(img_array)
                    labels.append(category)

    return images, labels

def main():
    st.title("Image Classification App")
    st.sidebar.header('Settings')

    # Image Downloading Section
    st.header("Image Downloading")
    categories_input = st.text_input("Enter categories (comma separated):")
    num_images = st.number_input("Number of images per category:", min_value=1, value=5)
    
    if st.button("Download Images"):
        categories = [category.strip() for category in categories_input.split(",") if category]
        if categories:
            with st.spinner('Downloading images...'):
                st_progress = st.progress(0)
                st_text = st.empty()
                for i, category in enumerate(categories, start=1):
                    st_text.text(f"Downloading images for category: {category}")
                    download_images(category, num_images, st_progress, st_text)
                    st_progress.progress(i / len(categories))
                st_text.text("Downloading completed!")
        else:
            st.warning("Please enter at least one category.")

    # Image Loading Section
    st.header("Load Images")
    base_path = st.text_input("Enter the base directory path:")
    categories_to_load = st.multiselect("Select categories to load:", os.listdir(base_path) if base_path else [])
    loaded_images, loaded_labels = load_images_from_directory(base_path, categories_to_load) if base_path and categories_to_load else ([], [])

    st.text(f"Total images loaded: {len(loaded_images)}")
    if loaded_images:
        st.image(loaded_images[0], caption="Sample Image", width=300)

    # EDA and Model Training Section (Placeholder)
    st.header("EDA and Model Training")
    st.text("This section is a placeholder for EDA and model training.")

if __name__ == "__main__":
    main()
