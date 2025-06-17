# Penjelasan Chapter 03 - Classification

Bab ini membahas tentang masalah klasifikasi dalam machine learning, di mana tujuannya adalah untuk **memprediksi kategori atau kelas dari suatu data**.

## 1. MNIST Dataset
Dataset MNIST terdiri dari gambar-gambar tulisan tangan digit **0 hingga 9**, yang sering digunakan sebagai dataset contoh dalam pembelajaran klasifikasi.
- **Penggunaan**: Melatih model machine learning agar dapat mengenali digit tulisan tangan.

## 2. Training a Binary Classifier
Klasifikasi biner adalah masalah **klasifikasi dengan hanya dua kelas** (misalnya, "ya" atau "tidak", "spam" atau "bukan spam").
- **Contoh**: Mengklasifikasikan apakah suatu angka adalah angka **5 atau bukan angka 5**.

## 3. Performance Measures
- **Akurasi (Accuracy)**: Proporsi prediksi yang benar.
  - *Rumus*: `Akurasi = (Jumlah prediksi benar) / (Jumlah total prediksi)`
- **Cross-Validation**: Teknik evaluasi model dengan membagi data menjadi beberapa **folds** untuk pelatihan dan validasi.

## 4. Confusion Matrix
Confusion matrix adalah **tabel yang menunjukkan kinerja model klasifikasi** dengan membandingkan prediksi dengan nilai aktual:
- **True Positives (TP)**: Prediksi positif yang benar.
- **True Negatives (TN)**: Prediksi negatif yang benar.
- **False Positives (FP)**: Prediksi positif yang salah (*Type I error*).
- **False Negatives (FN)**: Prediksi negatif yang salah (*Type II error*).

## 5. Precision and Recall
- **Precision**: Proporsi prediksi positif yang benar.
  - *Rumus*: `Precision = TP / (TP + FP)`
- **Recall (Sensitivity atau True Positive Rate)**: Proporsi positif aktual yang diprediksi dengan benar.
  - *Rumus*: `Recall = TP / (TP + FN)`

## 6. Precision/Recall Trade-off
Ada **trade-off** antara precision dan recall:
- **Meningkatkan precision** biasanya **menurunkan recall**, dan sebaliknya.
- **Kurva Precision-Recall** menunjukkan hubungan antara precision dan recall pada berbagai ambang batas (*threshold*).

## 7. The ROC Curve
Grafik **Receiver Operating Characteristic (ROC)** menunjukkan kinerja model klasifikasi berdasarkan ambang batas yang berbeda:
- **False Positive Rate (FPR)**: Proporsi negatif aktual yang diprediksi salah.
  - *Rumus*: `FPR = FP / (FP + TN)`
- **True Positive Rate (TPR) atau Recall**: Proporsi positif aktual yang diprediksi benar.
- **Area Under the Curve (AUC)**: Mengukur kemampuan model dalam **membedakan antara kelas-kelas**.

## 8. Multiclass Classification
Klasifikasi dengan **lebih dari dua kelas**, menggunakan dua strategi utama:
- **One-versus-All (OvA) atau One-versus-Rest (OvR)**: Melatih satu **classifier biner** untuk setiap kelas.
- **One-versus-One (OvO)**: Melatih **classifier biner untuk setiap pasangan kelas**.

## 9. Error Analysis
Menganalisis jenis **kesalahan yang dibuat oleh model** untuk meningkatkan kinerjanya.

## 10. Multilabel Classification
Klasifikasi di mana setiap **instance bisa diklasifikasikan ke dalam lebih dari satu kelas**.
- **Contoh**: Artikel berita yang bisa dikategorikan ke dalam beberapa **topik sekaligus**.

## 11. Multioutput Classification
Klasifikasi di mana setiap **kelas dapat memiliki beberapa nilai** untuk satu instance.