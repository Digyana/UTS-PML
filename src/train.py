import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import CNN1D, CNNLSTM
from data_utils import load_ut_har, standardize_data, create_dataloaders


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "UT_HAR"
RESULTS_ROOT = ROOT / "results"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    model_name,
    epochs=20,
    batch_size=64,
    learning_rate=0.001
):
    set_seed(42)

    RESULTS_ROOT.mkdir(exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # =========================
    # LOAD DATASET
    # =========================

    X_train, X_val, X_test, y_train, y_val, y_test = load_ut_har(
        DATA_ROOT
    )

    # =========================
    # STANDARDIZATION
    # =========================

    X_train, X_val, X_test = standardize_data(
        X_train,
        X_val,
        X_test
    )

    train_loader, val_loader, _ = create_dataloaders(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        batch_size=batch_size
    )

    # =========================
    # MODEL SELECTION
    # =========================

    if model_name == "cnn1d":
        model = CNN1D(num_classes=7)

        checkpoint_path = (
            RESULTS_ROOT / "cnn1d_best.pt"
        )

        accuracy_curve_path = (
            RESULTS_ROOT / "cnn1d_accuracy_curve.png"
        )

        loss_curve_path = (
            RESULTS_ROOT / "cnn1d_loss_curve.png"
        )

        plot_name = "1D-CNN"

    elif model_name == "cnnlstm":
        model = CNNLSTM(num_classes=7)

        checkpoint_path = (
            RESULTS_ROOT / "cnnlstm_best.pt"
        )

        accuracy_curve_path = (
            RESULTS_ROOT / "cnnlstm_accuracy_curve.png"
        )

        loss_curve_path = (
            RESULTS_ROOT / "cnnlstm_loss_curve.png"
        )

        plot_name = "CNN-LSTM"

    else:
        raise ValueError(
            "Model harus 'cnn1d' atau 'cnnlstm'."
        )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    best_val_acc = 0.0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    print(f"\nTraining model: {model_name}")
    print("=" * 70)

    # =========================
    # TRAINING
    # =========================

    for epoch in range(epochs):

        # TRAIN
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            train_loss += (
                loss.item() * X_batch.size(0)
            )

            predictions = outputs.argmax(dim=1)

            train_correct += (
                predictions == y_batch
            ).sum().item()

            train_total += y_batch.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # VALIDATION
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(
                    outputs,
                    y_batch
                )

                val_loss += (
                    loss.item() * X_batch.size(0)
                )

                predictions = outputs.argmax(dim=1)

                val_correct += (
                    predictions == y_batch
                ).sum().item()

                val_total += y_batch.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        # SAVE HISTORY
        history["train_loss"].append(
            train_loss
        )

        history["train_acc"].append(
            train_acc
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_acc"].append(
            val_acc
        )

        # SAVE BEST MODEL
        marker = ""

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

            marker = " <-- BEST"

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}%"
            f"{marker}"
        )

    # =========================
    # SAVE ACCURACY CURVE
    # =========================

    epoch_numbers = range(
        1,
        epochs + 1
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        epoch_numbers,
        history["train_acc"],
        label="Train Accuracy"
    )

    plt.plot(
        epoch_numbers,
        history["val_acc"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title(
        f"{plot_name} Training and Validation Accuracy"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        accuracy_curve_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # =========================
    # SAVE LOSS CURVE
    # =========================

    plt.figure(figsize=(8, 5))

    plt.plot(
        epoch_numbers,
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        epoch_numbers,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        f"{plot_name} Training and Validation Loss"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        loss_curve_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # =========================
    # SUMMARY
    # =========================

    print("\nTraining selesai.")

    print(
        f"Best Validation Accuracy: "
        f"{best_val_acc * 100:.2f}%"
    )

    print(
        f"Model tersimpan di: "
        f"{checkpoint_path}"
    )

    print(
        f"Accuracy curve tersimpan di: "
        f"{accuracy_curve_path}"
    )

    print(
        f"Loss curve tersimpan di: "
        f"{loss_curve_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=[
            "cnn1d",
            "cnnlstm"
        ],
        required=True
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001
    )

    args = parser.parse_args()

    train_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )