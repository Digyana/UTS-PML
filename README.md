# UTS PML - WiFi CSI Human Activity Recognition

Project pengganti UTS mata kuliah Pembelajaran Mesin Lanjut.

Project ini melakukan klasifikasi aktivitas manusia berbasis WiFi Channel State Information (CSI) menggunakan dataset UT-HAR.

## Dataset

Dataset yang digunakan adalah UT-HAR dengan 7 kelas aktivitas:

- bed
- fall
- pickup
- run
- sitdown
- standup
- walk

Dataset terdiri dari:

- Training: 3977 sampel
- Validation: 496 sampel
- Testing: 500 sampel

Setiap sampel memiliki dimensi 250 × 90.

## Models

Dua arsitektur Deep Learning dibandingkan:

1. 1D-CNN
2. CNN-LSTM

## Results

| Model | Best Validation Accuracy | Test Accuracy |
|---|---:|---:|
| 1D-CNN | 98.99% | 97.60% |
| CNN-LSTM | 94.15% | 94.80% |

Berdasarkan hasil eksperimen, 1D-CNN memperoleh performa lebih baik dibandingkan CNN-LSTM pada dataset UT-HAR.

## Project Structure

```text
UTS-PML/
├── data/
│   └── UT_HAR/
├── notebooks/
│   └── UT_HAR.ipynb
├── results/
├── src/
├── .gitignore
├── README.md
└── requirements.txt
```

Dataset tidak disertakan di repository karena ukuran file yang besar.

## Installation

Buat virtual environment:

```bash
python -m venv .venv
```

Aktifkan virtual environment pada Git Bash:

```bash
source .venv/Scripts/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running

Buka file:

```text
notebooks/UT_HAR.ipynb
```

Kemudian jalankan notebook menggunakan Python environment dari `.venv`.

## Hardware Used

Eksperimen dijalankan menggunakan NVIDIA GeForce RTX 4060 Laptop GPU dengan CUDA acceleration.