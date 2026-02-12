# import
import argparse
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Mlflow 추적 옵션
try:
    import mlflow
    mlflow.autolog()
except Exception:
    mlflow = None


def resolve_output_dir(model_output: str) -> Path:
    """
    model_output이 폴더면: 그 폴더 반환
    model_output이 파일 경로면: 그 파일의 parent 폴더 반환
    """
    p = Path(model_output)

    if str(model_output).endswith(("/", "\\")):
        p.mkdir(parents=True, exist_ok=True)
        return p
   
    if p.suffix:
        p.parent.mkdir(parents=True, exist_ok=True)
        return p.parent

    p.mkdir(parents=True, exist_ok=True)
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--max_iter", type=int, default=500)

    # v2 outputs 경로는 job.yml에서 항상 넘기므로 required사용
    parser.add_argument("--model_output", type=str, required=True)
    args = parser.parse_args()

    # data load, split
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Scaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # fitting
    model = LogisticRegression(max_iter=args.max_iter, random_state=args.random_state)
    model.fit(X_train, y_train)

    # prediction
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"accuracy: {acc:.4f}")

    # save file
    out_dir = resolve_output_dir(args.model_output)
    model_path = out_dir / "model.pkl"
    scaler_path = out_dir / "scaler.pkl"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Saved model to: {model_path.resolve()}")
    print(f"Saved scaler to: {scaler_path.resolve()}")


if __name__ == "__main__":
    main()
