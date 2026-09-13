from pathlib import Path
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "Crop_recommendation.csv"
)

MODEL_DIR = PROJECT_ROOT / "model"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Load dataset
data = pd.read_csv(DATA_PATH)


# Create the SAME LabelEncoder
# used during training
label_encoder = LabelEncoder()

label_encoder.fit(data["label"])


# Save encoder
encoder_path = MODEL_DIR / "label_encoder.pkl"

joblib.dump(
    label_encoder,
    encoder_path
)


print(
    f"Label encoder saved to: {encoder_path}"
)

print(
    "Crop classes:"
)

print(
    list(label_encoder.classes_)
)