# Prediksi Konversi Pengunjung E-Commerce Berdasarkan Perilaku Browsing Customer

Final Project Lab Based Expo — prediksi apakah sesi kunjungan e-commerce akan berakhir dengan transaksi (`Revenue`) berdasarkan perilaku browsing pengunjung (durasi halaman, jumlah halaman, bounce/exit rate, dll).

## 1. Dataset

**Online Shoppers Purchasing Intention Dataset**
- Sumber: UCI Machine Learning Repository
- Link: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- Sitasi: Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q
- Jumlah data: 12.330 sesi, 18 kolom (10 fitur numerik, 8 kategorikal)
- Target: `Revenue` (True/False) — imbalanced, 84,5% False vs 15,5% True

## 2. Struktur Repo

```
.
├── README.md
├── requirements.txt
├── online_shoppers_intention.csv                 # dataset mentah
├── Prediksi_Konversi_...ipynb                     # notebook utama (EDA -> model -> evaluasi)
├── sample_input.csv                               # sample data untuk demo (5 baris dari X_test, belum pernah dilihat model saat training)
├── final_model.pkl                                # model terbaik yang sudah di-training (dipakai saat demo, tidak retrain)
└── preprocessor_fe.pkl                            # ColumnTransformer (scaling + encoding) yang sudah di-fit
```

## 3. Cara Menjalankan

### 3.1 Setup environment

```bash
pip install -r requirements.txt
```

### 3.2 Menjalankan notebook dari awal (EDA sampai evaluasi)

1. Pastikan `online_shoppers_intention.csv` ada satu folder dengan notebook.
2. Buka `Prediksi_Konversi_Pengunjung_E-Commerce_Berdasarkan_Perilaku_Browsing_Customer.ipynb`.
3. Run All — notebook akan otomatis:
   - Load & EDA dataset
   - Preprocessing (train-test split → scaling & encoding → feature engineering)
   - Baseline comparison (Logistic Regression, Decision Tree, KNN)
   - Hyperparameter tuning + penanganan imbalance (Random Oversampling & SMOTE via `ImbPipeline`, resampling di-fit ulang tiap fold cross-validation supaya tidak bocor ke data validasi)
   - Perbandingan seluruh eksperimen & pemilihan model terbaik (berdasarkan F1-Score di test set)
   - Menyimpan `final_model.pkl`, `preprocessor_fe.pkl`, dan `sample_input.csv`

### 3.3 Menjalankan demo (tanpa training ulang)

Model final sudah di-training dan disimpan — untuk demo, load langsung dan prediksi:

```python
import joblib
import pandas as pd

model = joblib.load("final_model.pkl")
preprocessor = joblib.load("preprocessor_fe.pkl")

def add_features(df):
    df = df.copy()
    df["Total_Duration"] = (df["Administrative_Duration"] +
                             df["Informational_Duration"] +
                             df["ProductRelated_Duration"])
    df["Pages_Visited"] = (df["Administrative"] +
                            df["Informational"] +
                            df["ProductRelated"])
    df["Avg_Duration_Per_Page"] = df["Total_Duration"] / (df["Pages_Visited"] + 1)
    return df

data = pd.read_csv("sample_input.csv")
data_fe = add_features(data)
data_proc = preprocessor.transform(data_fe)

prediction = model.predict(data_proc)
print(prediction)
```

## 4. Metode

- **Model**: Machine learning klasik — Logistic Regression, Decision Tree, K-Nearest Neighbors (supervised).
- **Feature Engineering**: 3 fitur tambahan (`Total_Duration`, `Pages_Visited`, `Avg_Duration_Per_Page`) merepresentasikan engagement pengunjung.
- **Preprocessing**: `StandardScaler` untuk fitur numerik, `OneHotEncoder` untuk fitur kategorikal — di-fit hanya pada data training.
- **Penanganan Imbalance**: Random Oversampling dan SMOTE, diterapkan lewat `imblearn.pipeline.Pipeline` sehingga resampling hanya terjadi pada fold training saat cross-validation (test set asli tidak pernah di-resample).
- **Evaluasi**: Accuracy, Precision, Recall, F1-Score, ROC AUC — dihitung dari test set (20% data, stratified split), bukan data latih. F1-Score dipakai sebagai kriteria utama karena dataset imbalanced.

## 5. Hasil Eksperimen (ringkasan)

| Eksperimen | F1-Score (Test) |
|---|---|
| Decision Tree (Baseline) | 0,587 |
| Decision Tree (Tuned) | 0,669 |
| Decision Tree Tuned + Random Oversampling | 0,665 |
| **Decision Tree Tuned + SMOTE** | **0,673** (terbaik) |

Detail lengkap (termasuk confusion matrix dan perbandingan semua model) ada di notebook, Section 7–8.

## 6. Batasan

- Dataset imbalanced (84,4% : 15,6%); walau sudah ditangani SMOTE, recall kelas minoritas masih terbatas.
- Hanya fitur perilaku browsing — tidak ada data harga, histori pembelian, atau demografi.
- Hanya 3 model klasik yang dicoba (sesuai ketentuan FP — tanpa ensemble/deep learning untuk prediksi akhir).
- Threshold klasifikasi memakai default 0,5, belum dioptimasi.

Detail lengkap limitation & saran pengembangan ada di notebook, Section 10.
