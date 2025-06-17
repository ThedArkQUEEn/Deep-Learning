# Penjelasan Chapter 05 - Support Vector Machines (SVM)

Support Vector Machines (**SVMs**) adalah model pembelajaran mesin yang kuat dan serbaguna yang dapat digunakan untuk **klasifikasi linier**, **nonlinier**, **regresi**, dan **deteksi outlier**.

---

## 🔹 1. Klasifikasi SVM Linier (Linear SVM Classification)

### **Konsep Utama**:
- Mencari **hyperplane** (garis dalam 2D, bidang dalam 3D, dst.) yang **memisahkan kelas-kelas dengan margin terbesar**.
- **Margin** adalah jarak antara hyperplane dan **instance pelatihan terdekat** (disebut *support vector*).
- **Soft Margin Classification** memungkinkan beberapa instance melanggar margin untuk keseimbangan antara **margin terbesar dan pelanggaran margin minimal**.

### **Rumus**:
- **Fungsi Keputusan** (Hyperplane):

  $w^T x + b = 0$

  di mana:
  - **$w$** adalah vektor bobot,
  - **$x$** adalah vektor instance,
  - **$b$** adalah bias.

- **Optimasi Tujuan**:
  
  $\min \frac{1}{2} ||w||^2$

  Dengan kendala:

  $t_i (w^T x_i + b) \geq 1$

  di mana **$t_i$** adalah label target **(+1 atau -1)**.

---

## 🔹 2. Klasifikasi SVM Nonlinier (Nonlinear SVM Classification)

### **Konsep Utama**:
- **Kernel Trick** digunakan untuk **memetakan instance ke ruang berdimensi lebih tinggi**, memungkinkan pemisahan linier di ruang yang lebih kompleks.

### **Jenis Kernel**:
- **Polynomial Kernel**:

  $K(a, b) = (a^T b + r)^d$

  di mana **$d$** adalah **derajat polinomial**, **$r$** adalah **koefisien**.

- **Gaussian RBF Kernel**:

  $K(a, b) = \exp(-\gamma ||a - b||^2)$

  di mana **$\gamma$** mengontrol **lebar Gaussian**.

### **Fungsi Keputusan di Ruang Fitur yang Dipetakan**:

$f(x) = \sum \alpha_i t_i K(x_i, x) + b$

di mana **$\alpha_i$** adalah **bobot Lagrange**.

---

## 🔹 3. Regresi SVM (SVM Regression)

### **Konsep Utama**:
- **Alih-alih memisahkan dua kelas**, regresi **SVM** menyesuaikan sebanyak mungkin instance **dalam margin** yang ditentukan oleh **$\epsilon$**.
- **Epsilon-insensitive Tube**: Semua prediksi dalam **tabung epsilon** dianggap benar.

### **Rumus**:
Mirip dengan **klasifikasi SVM**, tetapi dengan tujuan berbeda:

$\min \frac{1}{2} ||w||^2 + C \sum (|\xi_i| + |\xi_i^*|)$

Dengan kendala:

$|y_i - (w^T x_i + b)| \leq \epsilon + |\xi_i|$

di mana **$\xi_i$** adalah **variabel slack**.

---

## 🔹 4. Di Balik Layar (Under the Hood)

### **Fungsi Keputusan dan Prediksi**:
- **Fungsi keputusan menghitung skor** berdasarkan **vektor bobot** dan instance.
- **Tanda skor** menentukan kelas **(+ atau -)**.

### **Tujuan Pelatihan (Training Objective)**:
- Menemukan **$w$** dan **$b$** yang **meminimalkan fungsi biaya**.

### **Masalah Dual (The Dual Problem)**:
- **Memecahkan masalah dual sering lebih efisien**, terutama dengan **kernel trick**.
- **Hanya support vector** yang digunakan dalam solusi, sehingga meningkatkan efisiensi komputasi.

---

## 🔹 5. SVM Online (Online SVMs)

### **Algoritma**:
- **SVM dapat dilatih secara online** menggunakan **Stochastic Gradient Descent (SGD)**.
- Implementasi dengan **`SGDClassifier`** di **Scikit-Learn**.

### **Manfaat**:
- Dapat menangani **data streaming** atau **dataset besar** yang tidak muat dalam memori.

---

##  **Kesimpulan**
- **SVM dapat digunakan untuk berbagai tugas**, termasuk **klasifikasi**, **regresi**, dan **deteksi outlier**.
- **Kernel Trick memungkinkan klasifikasi nonlinier** dengan memetakan data ke dimensi lebih tinggi.
- **Regresi SVM menggunakan pendekatan epsilon-insensitive** untuk menemukan model yang tetap dalam margin.
- **SVM online dapat menangani dataset besar** dengan metode **SGD**.

---