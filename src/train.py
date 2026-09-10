from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "dataset" / "Crop_recommendation.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "crop_recommendation_model.pkl"

FEATURES = [
	"N",
	"P",
	"K",
	"temperature",
	"humidity",
	"ph",
	"rainfall",
]


def train_and_save_model() -> None:
	data = pd.read_csv(DATA_PATH)
	model = RandomForestClassifier(
		n_estimators=200,
		random_state=42,
	)
	model.fit(data[FEATURES], data["label"])

	MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(model, MODEL_PATH)
	print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
	train_and_save_model()
