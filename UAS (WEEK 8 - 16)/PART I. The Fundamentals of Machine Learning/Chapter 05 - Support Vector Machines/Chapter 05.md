# Teknik Pelatihan Model: Regresi dan Klasifikasi

Dokumen ini membahas berbagai teknik pelatihan model dalam konteks **Regresi** dan **Klasifikasi**. Berikut adalah poin-poin utama serta rumus yang digunakan dalam metode ini.

---

## 1. Regresi Linear (Linear Regression)
**Tujuan:** Memodelkan hubungan linear antara variabel input (fitur) dan variabel output (target).

### Rumus:
- **Persamaan Garis Lurus:**  
  
    $$
    \hat{y} = \theta_0 + \theta_1x_1 + \theta_2x_2 + ... + \theta_nx_n
    $$
  
  - **ŷ**: Nilai prediksi.
  - **θ**: Parameter model (bias dan bobot fitur).
  - **x**: Nilai fitur.

- **Persamaan Normal (The Normal Equation):**  
  Digunakan untuk menemukan nilai **θ** yang meminimalkan fungsi biaya (misalnya, Mean Squared Error), dengan perhitungan matriks yang kompleks.

### Metode Optimalisasi:
- **Gradient Descent:** Algoritma iteratif untuk menemukan nilai **θ** dengan bergerak menuju arah penurunan fungsi biaya.
  - **Batch Gradient Descent**
  - **Stochastic Gradient Descent (SGD)**
  - **Mini-batch Gradient Descent**

---

## 2. Regresi Polinomial (Polynomial Regression)
**Tujuan:** Memodelkan hubungan non-linear dengan menambahkan fitur polinomial.

**Proses:**
- Data asli diubah untuk menyertakan fitur polinomial (misalnya, **x², x³, dst.**).
- Model **Regresi Linear** kemudian dilatih pada data yang telah diperluas.

---

## 3. Kurva Pembelajaran (Learning Curves)
- Grafik yang menunjukkan **kinerja model** pada data pelatihan dan validasi sebagai fungsi dari **ukuran set pelatihan**.
- Membantu mendeteksi **overfitting** atau **underfitting**.

---

## 4. Model Regresi Regularisasi (Regularized Linear Models)
**Tujuan:** Mencegah **overfitting** dengan menambahkan istilah regularisasi ke fungsi biaya.

### Jenis Regularisasi:
- **Ridge Regression:** Menambahkan istilah **L2 regularization**.
- **Lasso Regression:** Menambahkan istilah **L1 regularization**.
- **Elastic Net:** Kombinasi **L1 dan L2 regularization**.
- **Early Stopping:** Menghentikan pelatihan lebih awal berdasarkan kinerja pada set validasi.

---

## 5. Regresi Logistik (Logistic Regression)
**Tujuan:**  
Meskipun disebut **regresi**, ini adalah algoritma klasifikasi untuk **masalah klasifikasi biner**.

### Konsep Utama:
- **Estimasi Probabilitas:** Menghasilkan probabilitas bahwa sebuah instance termasuk dalam kelas tertentu.
- **Fungsi Biaya:**  
  - **Logistic Cost Function** digunakan untuk pelatihan.
- **Batas Keputusan (Decision Boundaries):**  
  - Memisahkan instance dari kelas yang berbeda.

---

## 6. Regresi Softmax (Softmax Regression)
**Tujuan:**  
Merupakan generalisasi dari **Regresi Logistik** untuk **masalah klasifikasi multi-kelas**.

### Karakteristik:
- Menghasilkan **probabilitas** untuk setiap kelas.
- Digunakan untuk **klasifikasi multikelas**.

---

## 7. Rumus Tambahan
Dokumen ini juga mencakup:
- **Fungsi Biaya (Cost Functions)** untuk berbagai jenis regresi.
- **Perhitungan gradien dalam Gradient Descent**.
- **Istilah regularisasi dalam Ridge, Lasso, dan Elastic Net**.
- **Fungsi Logistik (Sigmoid Function)**.
- **Fungsi Softmax**.

---