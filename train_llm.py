import numpy as np
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer
from torchvision import models, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- 1. Load Models ---

# Text model (LLM embedding)
text_model = SentenceTransformer('all-MiniLM-L6-v2')  # Free, fast, good quality

# Image model (Vision embedding)
vision_model = models.resnet50(pretrained=True)
vision_model.eval()
feature_extractor = torch.nn.Sequential(*list(vision_model.children())[:-1])  # Remove final classification layer

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 2. Feature Extraction Functions ---

def get_text_embedding(text):
    return text_model.encode(text)

def get_image_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        features = feature_extractor(img_tensor).squeeze().numpy()  # shape: (2048,)
    return features

# --- 3. Prepare Dataset ---

# Example: Load your data
# df = pd.read_csv("complaints_with_images.csv")
# For demonstration, let's assume df has columns: 'review_body', 'image_path', 'label'
# label: 1 for complaint, 0 for non-complaint

X = []
y = []

for idx, row in df.iterrows():
    text_feat = get_text_embedding(row['review_body'])  # shape: (384,)
    img_feat = get_image_embedding(row['image_path'])   # shape: (2048,)
    combined = np.concatenate([text_feat, img_feat])    # shape: (2432,)
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