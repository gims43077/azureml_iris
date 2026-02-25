import argparse
from pathlib import Path
import joblib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    model_path = model_dir / "model.pkl"
    scaler_path = model_dir / "scaler.pkl"

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 데모용 샘플 1개 (iris: sepal_len, sepal_wid, petal_len, petal_wid)
    x = [[5.1, 3.5, 1.4, 0.2]]

    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)[0]

    label_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    print("=== Iris inference demo ===")
    print("input:", x[0])
    print("pred_class:", int(pred))
    print("pred_label:", label_map.get(int(pred), "unknown"))

if __name__ == "__main__":
    main()