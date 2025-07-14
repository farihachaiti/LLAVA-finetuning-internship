import numpy as np
import pandas as pd
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import requests
from io import BytesIO
import clip

# --- 1. Load Models ---
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Text model (LLM embedding) - Use a compatible sentence-transformers model
text_model = SentenceTransformer('all-MiniLM-L6-v2')  # Free, fast, good quality

# Image model (Vision embedding)
vision_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
#model, preprocess = clip.load('ViT-B/32', device)
vision_model.eval()
feature_extractor = torch.nn.Sequential(*list(vision_model.children())[:-1])  # Remove final classification layer

## Image preprocessing
#preprocess = transforms.Compose([
#    transforms.Resize(256),
#    transforms.CenterCrop(224),
#    transforms.ToTensor(),
#    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#])

# --- 2. Feature Extraction Functions ---

def get_text_embedding(text):
    return text_model.encode(text)

def get_image_embedding(image_path):
    # Construct the full path to the image
    full_image_path = os.path.join("images", image_path)
    img = Image.open(full_image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        features = feature_extractor(img_tensor).squeeze().numpy()  # shape: (2048,)
    return features

# --- 3. Prepare Dataset ---

# Example: Load your data
df = pd.read_csv("complaints_with_images.csv")[:10000]
label_map = {"Complaint": 1, "Non-Complaint": 0}
df["label"] = df["annotation"].map(label_map)

# Check for NaN values after label mapping
print(f"NaN values in annotation: {df['annotation'].isna().sum()}")
print(f"NaN values in label: {df['label'].isna().sum()}")
print(f"Unique annotation values: {df['annotation'].unique()}")

df = df.dropna(subset=['image_url', 'review_body', 'annotation', 'label'])
def clean_text(text):
    return str(text).strip().lower()


df["text"] = df["review_body"].apply(clean_text)

os.makedirs("images", exist_ok=True)
os.makedirs("result_complaint_detector", exist_ok=True)
def download_image(url, review_id):
    try:
        response = requests.get(url, timeout=5)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        filename = f"{review_id.replace(' ', '')[:20]}.jpg"
        path = os.path.join("images", filename)
        img.save(path)
        return filename
    except Exception as e:
        print(f"Failed for {url}: {e}")
        return None
def is_valid_image(path):
    try:
        img = Image.open(path)
        img.verify()
        return True
    except Exception:
        return False


# Step 1: Download images
df["image_path"] = df.apply(lambda row: download_image(row["image_url"], row["review_id"]), axis=1)

# Step 2: Drop rows where download failed
df = df[df["image_path"].notna()]

# Step 3: Keep only rows with valid image files
def safe_is_valid_image(p):
    if p is None:
        return False
    full_path = os.path.join("images", p)
    return is_valid_image(full_path)

df = df[df["image_path"].apply(safe_is_valid_image)].reset_index(drop=True)

def get_image_size(image_path):
    try:
        with Image.open(os.path.join("images", image_path)) as img:
            return img.size[::-1]  # (height, width)
    except Exception:
        return None


df["image_sizes"] = df["image_path"].apply(get_image_size)
df = df[df["image_sizes"].notna()].reset_index(drop=True)
# For demonstration, let's assume df has columns: 'review_body', 'image_path', 'label'
# label: 1 for complaint, 0 for non-complaint

X = []
y = []
#for idx, row in df.iterrows():
#    text_feat = get_text_embedding(row['text'])  # shape: (384,)
#    img_feat = get_image_embedding(row['image_path'])   # shape: (2048,)
#    combined = np.concatenate([text_feat, img_feat])    # shape: (2432,)
#    X.append(combined)
#    y.append(row['label'])
with torch.no_grad():
    for idx, row in df.iterrows():
        # Load image from file path
        full_image_path = os.path.join("images", row['image_path'])
        image = Image.open(full_image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Tokenize text properly
        #text_inputs = clip.tokenize(row['text']).to(device)
        
        # Get CLIP features
        image_features = get_image_embedding(row['image_path'])
        text_features = get_text_embedding(row['text'])
        
        # Convert to numpy and concatenate
        image_features_np = image_features.flatten()
        text_features_np = text_features.flatten()
        combined = np.concatenate([text_features_np, image_features_np])
        
        X.append(combined)
        y.append(row['label'])

X = np.stack(X)
y = np.array(y)

# --- 4. Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 5. Train Classifier ---
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# --- 6. Evaluate ---
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))