import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet152
from sklearn.metrics.pairwise import cosine_similarity
import os
import csv

FEATURES_FILE = "database_features.csv"

def load_images(directory):
    images = []
    image_paths = []

    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            images.append(image)
            image_paths.append(image_path)

    return images, image_paths

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    preprocessed_image = transform(image)
    preprocessed_image = preprocessed_image.unsqueeze(0)

    return preprocessed_image

def extract_features(images):
    features = []
    model = resnet152(pretrained=True)
    model = model.eval()

    with torch.no_grad():
        for image in images:
            preprocessed_image = preprocess_image(image)
            features.append(model(preprocessed_image).squeeze().numpy())

    return features

def save_features(features, image_paths):
    with open(FEATURES_FILE, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "feature_1", "feature_2", ...])  # Replace feature_1, feature_2, ... with actual feature names

        for image_path, feature in zip(image_paths, features):
            writer.writerow([image_path] + list(feature))

def load_features():
    features = []
    image_paths = []

    with open(FEATURES_FILE, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip the header row

        for row in reader:
            image_path = row[0]
            feature = [float(x) for x in row[1:]]
            features.append(feature)
            image_paths.append(image_path)

    return features, image_paths

def search_similar_images(query_image, database_features, image_paths, top_n=5):
    query_feature = extract_features([query_image])[0]
    similarities = cosine_similarity([query_feature], database_features)[0]
    indices = similarities.argsort()[::-1][:top_n]
    similar_images = [image_paths[i] for i in indices]

    return similar_images

# Check if features file exists, otherwise extract features and save them
if os.path.exists(FEATURES_FILE):
    database_features, database_image_paths = load_features()
else:
    # Load database images
    database_images, database_image_paths = load_images('database_directory')

    # Extract features from database images using ResNet-152
    database_features = extract_features(database_images)

    # Save the extracted features
    save_features(database_features, database_image_paths)

# Load and preprocess the query image
query_image = cv2.imread('query_image.jpg')

# Search for similar images
similar_images = search_similar_images(query_image, database_features, database_image_paths, top_n=5)

# Print the similar images
print(similar_images)
