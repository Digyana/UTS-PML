# UTS PML - WiFi CSI Human Activity Recognition

Project pengganti UTS mata kuliah Pembelajaran Mesin Lanjut.

Project ini melakukan klasifikasi aktivitas manusia berbasis WiFi Channel State Information (CSI) menggunakan dataset UT-HAR.

## Kelompok 1

1. Zaviandra Chalil (2311511009)
2. Alfin Pratama Putra (2311511011)
3. Aulia Digyana Irwan (2311512001)
4. Sulthan Zhafransyah Kuswara (2311512027)

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

Dataset tidak disertakan di repository karena ukuran file yang besar.

## Models

Dua arsitektur Deep Learning dibandingkan:

1. 1D-CNN
2. CNN-LSTM

## Results

| Model | Best Validation Accuracy | Test Accuracy |
|---|---:|---:|
| 1D-CNN | 99.19% | 98.80% |
| CNN-LSTM | 95.77% | 94.80% |

Berdasarkan hasil eksperimen dengan random seed 42, 1D-CNN memperoleh performa terbaik dengan test accuracy 98.80%, sedangkan CNN-LSTM memperoleh test accuracy 94.80%.

## Project Structure

```text
UTS-PML/
├── data/
│   └── UT_HAR/
├── notebooks/
│   └── UT_HAR.ipynb
├── results/
│   ├── cnn1d_accuracy_curve.png
│   ├── cnn1d_confusion_matrix.png
│   ├── cnn1d_loss_curve.png
│   ├── cnnlstm_accuracy_curve.png
│   ├── cnnlstm_confusion_matrix.png
│   └── cnnlstm_loss_curve.png
├── src/
│   ├── data_utils.py
│   ├── evaluate.py
│   ├── models.py
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

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

## Running the Notebook

Buka file:

```text
notebooks/UT_HAR.ipynb
```

Kemudian jalankan notebook menggunakan Python environment dari `.venv`.

## Training

Training 1D-CNN:

```bash
python src/train.py --model cnn1d --epochs 20
```

Training CNN-LSTM:

```bash
python src/train.py --model cnnlstm --epochs 20
```

Checkpoint model terbaik akan disimpan di folder `results/`. File checkpoint `.pt` tidak disertakan di repository.

## Evaluation

Evaluasi 1D-CNN:

```bash
python src/evaluate.py --model cnn1d
```

Evaluasi CNN-LSTM:

```bash
python src/evaluate.py --model cnnlstm
```

Proses evaluasi menampilkan test accuracy dan classification report serta menyimpan confusion matrix ke folder `results/`.

## Hardware Used

Eksperimen dijalankan menggunakan NVIDIA GeForce RTX 4060 Laptop GPU dengan CUDA acceleration.
