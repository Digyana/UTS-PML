import numpy as np
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader


def load_ut_har(data_root):
    data_root = Path(data_root)

    X_train = np.load(data_root / "data/X_train.csv", allow_pickle=True)
    X_val = np.load(data_root / "data/X_val.csv", allow_pickle=True)
    X_test = np.load(data_root / "data/X_test.csv", allow_pickle=True)

    y_train = np.load(data_root / "label/y_train.csv", allow_pickle=True)
    y_val = np.load(data_root / "label/y_val.csv", allow_pickle=True)
    y_test = np.load(data_root / "label/y_test.csv", allow_pickle=True)

    return X_train, X_val, X_test, y_train, y_val, y_test


def standardize_data(X_train, X_val, X_test):
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)

    feature_mean = X_train.mean(axis=(0, 1), keepdims=True)
    feature_std = X_train.std(axis=(0, 1), keepdims=True)

    feature_std[feature_std == 0] = 1.0

    X_train = (X_train - feature_mean) / feature_std
    X_val = (X_val - feature_mean) / feature_std
    X_test = (X_test - feature_mean) / feature_std

    return X_train, X_val, X_test


def create_dataloaders(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    batch_size=64
):
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train).long()
    )

    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val).long()
    )

    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test).long()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader