# Penjelasan Chapter 08 - Dimensionality Reduction 

Bab ini membahas berbagai pendekatan untuk mengurangi dimensi data, mengatasi tantangan dalam ruang berdimensi tinggi, dan menggunakan berbagai teknik seperti PCA, Kernel PCA, dan LLE.

---

## Curse of Dimensionality & Pendekatan Utama

- **Curse of Dimensionality**: Di ruang berdimensi tinggi, data menjadi jarang, membuat model Machine Learning sulit belajar dan rawan overfitting.
- **Proyeksi**: Data diputar dan diproyeksikan ke ruang berdimensi lebih rendah.
- **Manifold Learning**: Asumsikan data berdimensi tinggi berada pada manifold berdimensi lebih rendah, dan teknik reduksi bertugas mengekstraknya.

---

## Principal Component Analysis (PCA)

- **Tujuan**: Mempertahankan varians maksimum saat memproyeksikan data.
- **Komponen Utama**: Vektor unit yang mewakili arah varians maksimum.
  
### Rumus SVD
$$X = U \Sigma V^T$$

- $V$ berisi komponen utama (principal components).

### Proyeksi ke d-Dimensi
$$X_{d-proj} = X W_d$$

- $W_d$: matriks dari d komponen utama pertama.

### Scikit-Learn PCA
- Gunakan kelas `PCA`.
- `explained_variance_ratio_`: menunjukkan rasio varians yang dijelaskan oleh tiap komponen.

### Memilih Dimensi
- Targetkan 95% varians untuk ditangkap.
- Gunakan elbow plot dari `explained_variance_ratio_`.

### Kompresi dengan PCA
- Mengurangi ukuran data untuk penyimpanan/komputasi lebih efisien.
- Dekompresi = rekonstruksi dengan kehilangan informasi minimal.

### Randomized PCA
- Alternatif cepat untuk dataset besar.
- Cocok jika jumlah dimensi target jauh lebih kecil dari ukuran dataset.

### Incremental PCA
- Proses batch kecil dari dataset besar.
- Cocok untuk data yang tidak muat dalam memori sekaligus.

---

## Kernel PCA & LLE

### Kernel PCA
- Melakukan transformasi nonlinier dengan kernel (RBF, polynomial, sigmoid, dll).
- Meningkatkan representasi linier dari data kompleks.

### Tuning Hyperparameters
- Gunakan `GridSearchCV` dengan validasi silang.
- Evaluasi kinerja melalui tugas hilir seperti klasifikasi.

### LLE (Locally Linear Embedding)
- Mencari representasi berdimensi rendah tanpa proyeksi.
- Setiap data dimodelkan sebagai kombinasi linier dari tetangga terdekat.
- Optimalkan representasi agar mempertahankan bobot hubungan lokal.

---

## Teknik Reduksi Dimensi Lainnya

- **MDS (Multi-Dimensional Scaling)**: Pertahankan jarak antar instance dalam proyeksi.
- **Isomap**: Pertahankan jarak geodetik pada manifold.
- **t-SNE**: Visualisasi berdimensi rendah yang mempertahankan struktur lokal.
- **LDA (Linear Discriminant Analysis)**: Teknik supervisi untuk klasifikasi & reduksi dimensi.

---