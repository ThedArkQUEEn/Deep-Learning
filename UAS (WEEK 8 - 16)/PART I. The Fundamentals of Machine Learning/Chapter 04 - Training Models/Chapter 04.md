# Penjelasan Chapter 04 - Training Models

Bab ini membahas berbagai teknik pelatihan model dalam konteks **Regresi** dan **Klasifikasi**. Berikut adalah poin-poin utama serta rumus yang digunakan dalam metode ini.

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

    - $\theta = (X^TX)^{-1}X^Ty$

    - Pendekatan ini efektif untuk dataset berukuran kecil hingga menengah karena memberikan **solusi langsung** tanpa perlu iterasi.

### Metode Optimalisasi:
- **Gradient Descent (Iteratif)**

    - Metode optimasi berbasis iterasi untuk memperbarui **θ** berdasarkan gradien fungsi biaya:

    - $\theta := \theta - \alpha \nabla J(\theta)$

    - di mana:
        - **$\alpha$** adalah **learning rate** yang mengontrol kecepatan perubahan **θ**.
        - **$\nabla J(\theta)$** adalah **gradien dari fungsi biaya**, menunjukkan arah perubahan parameter untuk mengurangi kesalahan.

    - **Pendekatan ini lebih efisien** untuk dataset besar karena tidak memerlukan inversi matriks seperti Normal Equation.


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
- **Transformasi fitur:**  

  $
  x \rightarrow (1, x, x^2, ..., x^d)
  $

  di mana:
  - **$x$** adalah fitur asli,
  - **$x^d$** adalah fitur polinomial dengan derajat **$d$**.

### Kesimpulan:
- **Regresi Polinomial efektif** dalam memodelkan **data non-linier**.
- Dengan menambahkan **fitur pangkat**, regresi linier dapat menangkap **pola yang lebih kompleks** dalam data.


---

## 3. Kurva Pembelajaran (Learning Curves)
### Konsep:
- Plot **training error** dan **validation error** terhadap **ukuran data**.
- Membantu mendiagnosis **underfitting** dan **overfitting**.

### Pola yang Terlihat:
- **Overfitting:**  
  - **Training error rendah**, tetapi **validation error tinggi**.  
  - Model terlalu kompleks dan menghafal data latih.
- **Underfitting:**  
  - **Baik training error maupun validation error tinggi**.  
  - Model terlalu sederhana dan tidak menangkap pola dengan baik.
### Kesimpulan:
- **Kurva pembelajaran membantu memilih kompleksitas model yang optimal**.
- Model yang seimbang akan memiliki **error rendah dan konvergen** antara training dan validation set.



---

## 4. Model Regresi Regularisasi (Regularized Linear Models)
- **Regularisasi** menambahkan **penalty ke fungsi biaya** untuk **mencegah overfitting**.
- Memastikan model tidak terlalu kompleks dan lebih mampu **menggeneralisasi data baru**.

### Jenis Regularisasi:
- **Ridge Regression (L2 Regularization)**  

  $
  J(\theta) = \text{Loss} + \alpha \sum \theta^2
  $

  - **Menekan bobot fitur agar tidak terlalu besar**.
  - **Menghindari overfitting** dengan memberikan penalti **L2 norm** pada parameter.

- **Lasso Regression (L1 Regularization)**  

  $
  J(\theta) = \text{Loss} + \alpha \sum |\theta|
  $

  - **Dapat membuat beberapa parameter menjadi nol**, sehingga berguna untuk **seleksi fitur**.
  - **Lebih agresif dalam mencegah overfitting** dibandingkan Ridge.


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

### Rumus:
- **Prediksi probabilitas**  

  $
  h_{\theta}(z) = \sigma (\theta^T z)
  $

  di mana **$\sigma(x)$** adalah fungsi sigmoid:

  $
  \sigma(x) = \frac{1}{1 + e^{-x}}
  $

### Fungsi Loss:
- **Log Loss (Binary Cross-Entropy)** digunakan sebagai fungsi biaya:

  $
  J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \Big[ y^{(i)} \log h_{\theta}(x^{(i)}) + (1 - y^{(i)}) \log (1 - h_{\theta}(x^{(i)})) \Big]
  $

### Kesimpulan:
- **Regresi Logistik tidak digunakan untuk prediksi angka**, tetapi untuk **klasifikasi probabilistik**.
- Model ini efektif dalam **menentukan batas keputusan** untuk dua kelas dalam masalah klasifikasi biner.

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