import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from models import CNN1D, CNNLSTM
from data_utils import load_ut_har, standardize_data, create_dataloaders


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "UT_HAR"
RESULTS_ROOT = ROOT / "results"

CLASS_NAMES = [
    "bed",
    "fall",
    "pickup",
    "run",
    "sitdown",
    "standup",
    "walk"
]


def evaluate_model(model_name, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    X_train, X_val, X_test, y_train, y_val, y_test = load_ut_har(DATA_ROOT)

    X_train, X_val, X_test = standardize_data(
        X_train,
        X_val,
        X_test
    )

    _, _, test_loader = create_dataloaders(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        batch_size=batch_size
    )

    if model_name == "cnn1d":
        model = CNN1D(num_classes=7)
        checkpoint_path = RESULTS_ROOT / "cnn1d_best.pt"
        figure_path = RESULTS_ROOT / "cnn1d_confusion_matrix.png"
        title = "Confusion Matrix - 1D-CNN"

    elif model_name == "cnnlstm":
        model = CNNLSTM(num_classes=7)
        checkpoint_path = RESULTS_ROOT / "cnnlstm_best.pt"
        figure_path = RESULTS_ROOT / "cnnlstm_confusion_matrix.png"
        title = "Confusion Matrix - CNN-LSTM"

    else:
        raise ValueError("Model harus 'cnn1d' atau 'cnnlstm'.")

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True
        )
    )

    model = model.to(device)
    model.eval()

    predictions_all = []
    labels_all = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            predictions = outputs.argmax(dim=1)

            predictions_all.extend(predictions.cpu().numpy())
            labels_all.extend(y_batch.cpu().numpy())

    accuracy = accuracy_score(
        labels_all,
        predictions_all
    )

    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            labels_all,
            predictions_all,
            target_names=CLASS_NAMES,
            digits=4
        )
    )

    cm = confusion_matrix(
        labels_all,
        predictions_all
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )

    fig, ax = plt.subplots(figsize=(9, 8))

    disp.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=False
    )

    plt.title(title)
    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Confusion matrix tersimpan di: {figure_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["cnn1d", "cnnlstm"],
        required=True
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64
    )

    args = parser.parse_args()

    evaluate_model(
        model_name=args.model,
        batch_size=args.batch_size
    )