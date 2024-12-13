import os
from skimage.feature import graycomatrix, graycoprops
from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np
from shearch_with_image_vgg16 import *



def extract_glcm_features(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    if len(image.shape) == 3:
        image = rgb2gray(image)
    image = (image * 255).astype(np.uint8)
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    features = {
        'contrast': graycoprops(glcm, 'contrast').mean(),
        'dissimilarity': graycoprops(glcm, 'dissimilarity').mean(),
        'homogeneity': graycoprops(glcm, 'homogeneity').mean(),
        'energy': graycoprops(glcm, 'energy').mean(),
        'correlation': graycoprops(glcm, 'correlation').mean()
    }
    return features


def calculate_similarity(query_features, dataset_features):
    similarities = []
    for image_path, features in dataset_features.items():
        distance = np.linalg.norm(np.array(list(query_features.values())) - np.array(list(features.values())))
        similarities.append((image_path, distance))
    similarities.sort(key=lambda x: x[1])
    return similarities

def query_image_data(image_path,database_file="books.json"):
    main_folder = 'images'  
    if os.path.exists(database_file):
        database = load_image_database(database_file)
    else:
        print("Building image database...")
        database = build_image_database(main_folder, save_path=database_file)
    query_hog_features = extract_hog_features(image_path)
    query_vgg_features = extract_vgg_features(image_path)

    similarities = []
    for entry in database:
        image_path = entry['image_path']
        titel = entry['titel']
        author = entry['Author']
        description = entry['Description']
        hog_features = entry['hog_features']
        vgg_features = entry['vgg_features']
        hog_similarity = calculate_similarity_2(query_hog_features, hog_features)
        vgg_similarity = calculate_similarity_2(query_vgg_features, vgg_features)
        combined_similarity = 0.5 * hog_similarity + 0.5 * vgg_similarity
        similarities.append((image_path, author, titel, description, combined_similarity))
        print(titel)
    similarities = sorted(similarities, key=lambda x: x[4], reverse=True)
    result_data = [[x[0],x[1],x[2],x[3]] for x in similarities]
    return result_data[:1]

