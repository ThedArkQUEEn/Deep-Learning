#  Penjelasan Chapter 06 - Decision Trees

Bab ini membahas **Decision Trees**, salah satu algoritma **Machine Learning** yang digunakan untuk **klasifikasi dan regresi**.

---

##  Konsep Utama

### **Struktur Pohon**:
- **Decision Tree** adalah model berbentuk pohon di mana:
  - **Setiap node internal** mewakili **tes pada atribut (fitur)**.
  - **Setiap cabang** mewakili **hasil dari tes**.
  - **Setiap node daun** (leaf) memberikan **label kelas (klasifikasi)** atau **nilai prediksi (regresi)**.

### **Pembuatan Keputusan**:
- Klasifikasi atau regresi dilakukan dengan **menelusuri pohon** dari **akar (root) ke daun**, mengikuti **tes pada setiap node**.

### **Algoritma CART**:
- **Classification And Regression Tree (CART)** adalah algoritma umum untuk **membangun Decision Trees**.
- Digunakan baik untuk **klasifikasi** maupun **regresi**.

---

##  Rumus dan Konsep Penting

### 🔹 **Gini Impurity (untuk Klasifikasi)**
- **Mengukur ketidakmurnian node** → **Semakin rendah Gini, semakin "murni" suatu node**.
- **Rumus**:

  $\text{Gini}(p) = 1 - \sum (p_i)^2$

  di mana:
  - **$p_i$** adalah **rasio instance** dalam kelas **$i$** di node tersebut.
  - **$\sum$** menunjukkan **penjumlahan** untuk semua kelas.

### 🔹 **Entropy (untuk Klasifikasi)**
- **Alternatif dari Gini**, juga mengukur **ketidakmurnian node**.
- **Rumus**:

  $H(p) = - \sum (p_i \log_2 p_i)$

  di mana:
  - **$p_i$** adalah **rasio instance dalam kelas $i$**.
  - **$\sum$** menunjukkan **penjumlahan** untuk semua kelas.

### 🔹 **Memilih Split Terbaik**
- **CART** mencari **split (tes)** pada setiap node yang menghasilkan **penurunan ketidakmurnian (Gini atau Entropy) terbesar**.

### 🔹 **Regularisasi Hyperparameters**
- **Decision Trees rentan terhadap overfitting**.
- Gunakan **hyperparameter** untuk **membatasi pertumbuhan pohon** dan **meningkatkan generalisasi**:
  - **`max_depth`** → Batasi **kedalaman maksimum pohon**.
  - **`min_samples_split`** → Tentukan **jumlah minimum sampel** untuk membagi node.
  - **`min_samples_leaf`** → Tentukan **jumlah minimum sampel dalam node daun**.

---

##  Regresi dengan Decision Trees

### 🔹 **Konsep Regresi**
- Dalam tugas **regresi**, **Decision Trees memprediksi nilai numerik**.
- Prediksi pada **node daun** adalah **rata-rata nilai target dari instance pelatihan** di node tersebut.

### 🔹 **Mean Squared Error (MSE)**
- **MSE sering digunakan sebagai kriteria** untuk memilih split terbaik dalam **regresi Decision Trees**.
- **Rumus**:

  $MSE = \frac{1}{m} \sum (y_i - \hat{y}_i)^2$

  di mana:
  - **$y_i$** adalah **nilai asli** dari instance.
  - **$\hat{y}_i$** adalah **nilai prediksi**.
  - **$m$** adalah **jumlah sampel**.

---

##  Ringkasan Singkat

✔ **Decision Trees adalah algoritma Machine Learning yang kuat dan serbaguna**.  
✔ **Algoritma CART secara rekursif membagi dataset menjadi subset** untuk meminimalkan **Gini Impurity atau Entropy** (klasifikasi) atau **varians (regresi)**.  
✔ **Gini Impurity dan Entropy mengukur ketidakmurnian node dalam klasifikasi**.  
✔ **Hyperparameter digunakan untuk mengontrol kompleksitas pohon dan mencegah overfitting**.