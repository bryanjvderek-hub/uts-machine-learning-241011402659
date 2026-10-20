# UTS Machine Learning — Linear Regression & Logistic Regression

## Identitas Mahasiswa

- **Nama:** Bryan Justin Valentino Derek
- **NIM:** 241011402659
- **Kelas:** Reg B
- **Random state:** `59`

## Tujuan

Tugas ini menerapkan Linear Regression untuk memodelkan hubungan biaya iklan dengan penjualan, serta Logistic Regression untuk mengklasifikasikan email sebagai spam atau bukan spam. Implementasi utama model, pembagian data, dan metrik evaluasi dibuat sendiri menggunakan NumPy sesuai instruksi notebook. Scikit-learn hanya digunakan pada bagian bonus sebagai pembanding.

## Dataset

- **`Ads_Data.csv`** — 40 baris, dengan kolom `BiayaIklan` (juta rupiah per bulan) dan `Penjualan` (juta rupiah per bulan).
- **`Email_Data.csv`** — 60 baris, dengan kolom `JumlahLink`, `KataPromo`, dan `Spam` (1 = spam, 0 = bukan spam). Dataset terdiri dari 25 email spam dan 35 bukan spam.

## Bagian A — Linear Regression

Model Linear Regression dilatih menggunakan Normal Equation pada data training, dengan pembagian data 80% training dan 20% testing (`random_state=59`). Parameter yang diperoleh:

- Intercept: **27,741367**
- Koefisien biaya iklan: **9,096725**
- Persamaan: **ŷ = 27,741367 + 9,096725 × BiayaIklan**

Artinya, di dalam rentang dan pola data ini, kenaikan biaya iklan sebesar Rp1 juta berasosiasi dengan kenaikan prediksi penjualan sekitar Rp9,10 juta. Ini adalah hubungan yang dipelajari dari data, bukan bukti bahwa iklan menjadi satu-satunya penyebab penjualan naik.

| Metrik | Training | Testing |
|---|---:|---:|
| MAE | 4,5471 | 4,7274 |
| MSE | 37,6723 | 30,9590 |
| RMSE | 6,1378 | 5,5641 |
| R² | 0,9423 | 0,9231 |

R² testing sebesar 0,9231 menunjukkan model menjelaskan sebagian besar variasi penjualan pada sampel testing ini. RMSE testing sekitar Rp5,56 juta menunjukkan besaran kesalahan prediksi tipikal dalam satuan penjualan. Hasil tetap perlu dibaca dengan mempertimbangkan ukuran dataset yang terbatas dan kemungkinan faktor lain yang memengaruhi penjualan.

### Eksperimen Gradient Descent

Eksperimen menggunakan 5.000 iterasi dan tiga learning rate:

- **0,001:** masih relatif lambat; pada iterasi ke-5.000, intercept sekitar 19,919930, koefisien 10,337532, dan MSE 52,568697.
- **0,02:** paling sesuai pada eksperimen ini; intercept sekitar 27,741367, koefisien 9,096725, dan MSE 37,672289, mendekati hasil Normal Equation.
- **0,1:** divergen; parameter menjadi tidak finite pada iterasi ke-462. Nilai ini terlalu besar untuk skala fitur dan target pada data yang digunakan.

Notebook menyertakan grafik loss terhadap iterasi dan menandai divergensi tanpa menyamarkan hasil eksperimen.

Prediksi dari model Normal Equation:
- Biaya iklan Rp4,5 juta → penjualan diprediksi **Rp68,677 juta**.
- Biaya iklan Rp7 juta → penjualan diprediksi **Rp91,418 juta**.
- Biaya iklan Rp12 juta → penjualan diprediksi **Rp136,902 juta**.

Prediksi pada Rp12 juta merupakan ekstrapolasi di luar rentang data yang tersedia, sehingga lebih tidak pasti dibandingkan prediksi dalam rentang data.

## Bagian B — Logistic Regression

Model Logistic Regression dilatih dengan Gradient Descent (`learning_rate=0.1`, `n_iterations=10000`) menggunakan dua fitur: jumlah link dan jumlah kata promo. Pembagian data adalah 45 baris training dan 15 baris testing.

Parameter yang dipelajari:
- Intercept: **-11,567036**
- Koefisien `JumlahLink`: **1,053244**
- Koefisien `KataPromo`: **0,942154**

Persamaan probabilitas:

`P(Spam) = sigmoid(-11,567036 + 1,053244 × JumlahLink + 0,942154 × KataPromo)`

Log loss turun dari sekitar **0,681384** pada awal pelatihan menjadi **0,264458** pada akhir pelatihan.

### Evaluasi pada data testing

Confusion matrix dengan urutan `[[TN, FP], [FN, TP]]`:

```text
[[11, 0],
 [ 1, 3]]
```

- **Accuracy:** 0,9333 (93,33%)
- **Precision:** 1,0000 (100%)
- **Recall:** 0,7500 (75%)
- **F1-score:** 0,8571 (85,71%)

Pada sampel testing, tidak ada email bukan spam yang salah ditandai sebagai spam (FP = 0), tetapi satu email spam lolos sebagai bukan spam (FN = 1). Karena data testing hanya 15 email, metrik ini belum cukup untuk menyimpulkan kinerja pada semua email di dunia nyata.

### Eksperimen threshold

| Threshold | TN | FP | FN | TP | Precision | Recall | F1-score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,3 | 10 | 1 | 1 | 3 | 0,75 | 0,75 | 0,75 |
| 0,5 | 11 | 0 | 1 | 3 | 1,00 | 0,75 | 0,8571 |
| 0,7 | 11 | 0 | 1 | 3 | 1,00 | 0,75 | 0,8571 |

Threshold **0,5** dipilih sebagai kompromi awal karena pada sampel testing menghasilkan precision 100% dan recall 75%. Threshold 0,7 tidak mengubah klasifikasi pada sampel ini. Pilihan threshold perlu dievaluasi lagi dengan data yang lebih banyak sebelum digunakan dalam sistem nyata.

### Prediksi email baru

Dengan threshold 0,5, notebook memprediksi:
- **E1** (`JumlahLink=1`, `KataPromo=2`): bukan spam, probabilitas sekitar 0,0002.
- **E2** (`JumlahLink=5`, `KataPromo=6`): bukan spam, probabilitas sekitar 0,3435.
- **E3** (`JumlahLink=9`, `KataPromo=10`): spam, probabilitas sekitar 0,9993.
- **E4** (`JumlahLink=2`, `KataPromo=11`): spam, probabilitas sekitar 0,7116.

## Bonus — Perbandingan dengan scikit-learn

Linear Regression dari implementasi NumPy dan `sklearn.linear_model.LinearRegression` menghasilkan parameter dan metrik testing yang sama pada presisi yang ditampilkan: intercept 27,741367, koefisien 9,096725, MAE 4,727423, MSE 30,958974, RMSE 5,564079, dan R² 0,923130.

Untuk Logistic Regression, scikit-learn menggunakan `C=1e6` dan `max_iter=10000` sebagai pembanding dengan regularisasi yang sangat lemah. Akurasi, precision, recall, dan F1 pada testing sama dengan implementasi NumPy pada konfigurasi ini, tetapi koefisiennya berbeda karena metode optimasi dan pengaturan regularisasi tidak identik.

## Kesimpulan

Pada dataset iklan ini, Linear Regression menunjukkan kecocokan yang baik terhadap data testing, tetapi prediksi di luar rentang data perlu diperlakukan dengan hati-hati. Logistic Regression mendapatkan accuracy 93,33% pada sampel testing, tetapi masih melewatkan satu email spam. Hasil eksperimen berguna untuk memahami proses model dan metrik; ukuran data yang kecil membuat evaluasi tambahan tetap diperlukan.

## Struktur Repository

```text
uts-machine-learning-241011402659/
├── README.md
├── tugas_UTS.ipynb
├── Ads_Data.csv
└── Email_Data.csv
```

## Dependensi

Notebook diuji menggunakan Python **3.13.5** dengan paket berikut tersedia:

- NumPy
- Pandas
- Matplotlib
- scikit-learn (hanya bagian bonus)
- Jupyter/IPython kernel (`ipykernel`, `nbformat`, `nbclient` untuk menjalankan notebook)

Jika diperlukan, instal paket dengan:

```bash
python -m pip install numpy pandas matplotlib scikit-learn ipykernel nbformat nbclient
```

Buka notebook dari folder repository dan pilih kernel Python yang sesuai. Jalankan **Restart Kernel and Run All** untuk mengulang seluruh analisis dari awal.
