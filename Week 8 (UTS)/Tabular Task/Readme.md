# **Hasil dan penjelasan regresi:**

**1. RegresiUTSTelkom - Tensorflow MLP - (256-128-64) - No Batch Norm No dropout**
- **MSE: 104.6609**
- **RMSE: 10.2304**
- **MAE: 7.1375**
- **R² Score: 0.1286**

**2. RegresiUTSTelkom - Tensorflow MLP - (256-128-64)**
- **MSE: 89.7439**
- **RMSE: 9.4733**
- **MAE: 6.7766**
- **R² Score: 0.2528**

**3. RegresiUTSTelkom - Tensorflow MLP - (512-256-128-64)**
- **MSE: 84.2023**
- **RMSE: 9.1762**
- **MAE: 6.4898**
- **R² Score: 0.2989**

Dari hasil ketiganya, model regresi tanpa Batch Normalization dan tanpa Dropout hasilnya lebih buruk dibandingkan yang lain secara score R². Yang terbaik adalah dengan tambahan layer 512 namun masih sangat jauh dari angka 1.

# **Hasil dan penjelasan klasifikasi 4 class:**

**1. RegresiUTSTelkom - Tensorflow MLP - (256-128-64) - No Batch Norm No dropout**
- **Accuracy: 0.4249**
- **Precision: 0.4155**
- **Recall: 0.4249**
- **F1-Score: 0.4140**
- **Class 0 (AUC = 0.8139)**
- **Class 1 (AUC = 0.6246)**
- **Class 2 (AUC = 0.6540)**
- **Class 3 (AUC = 0.7140)**

**2. RegresiUTSTelkom - Tensorflow MLP - (256-128-64)**
- **Accuracy: 0.4250**
- **Precision: 0.4127**
- **Recall: 0.4250**
- **F1-Score: 0.4129**
- **Class 0 (AUC = 0.8144)**
- **Class 1 (AUC = 0.6237)**
- **Class 2 (AUC = 0.6533)**
- **Class 3 (AUC = 0.7138)**

**3. RegresiUTSTelkom - Tensorflow MLP - (512-256-128-64)**
- **Accuracy: 0.4461**
- **Precision: 0.4344**
- **Recall: 0.4461**
- **F1-Score: 0.4346**
- **Class 0 (AUC = 0.8406)**
- **Class 1 (AUC = 0.6428)**
- **Class 2 (AUC = 0.6667)**
- **Class 3 (AUC = 0.7315)**

Dari hasil ketiganya, model klasifikasi semuanya memiliki hasil yang mirip dan masih jauh dari angka 1 untuk model yang bagus, namun sedikit lumayan untuk hasil AUC score setiap class. Kemungkinan hasilnya tidak bagus karena pembagian 4 class dan bukan 2 class yang lebih mudah di training.

---

# **Penjelasan untuk Kode:**
# **Analisis Statistik dan Preprocessing Data**

## **1. Analisis Tipe Data dan Data Hilang**
Kode ini mengevaluasi tipe data dan jumlah nilai yang hilang dalam dataset: **data.dtypes.value_counts() data.isnull().sum()**

Menghitung total nilai yang hilang: Total Missing Values = **data.isnull().sum().sum()**

Jika ada data yang hilang, bisa dilakukan imputasi atau penghapusan.

## **2. Ekstraksi Target dan Fitur**
Variabel target dipilih sebagai kolom pertama: 

**y = data.iloc[:, 0].values** 

**X = data.drop(data.columns[0], axis=1)**


Target `y` adalah variabel yang akan diprediksi, sedangkan `X` berisi fitur yang digunakan dalam pelatihan model.

## **3. Statistik Dasar Target**
Rumus rata-rata (mean), nilai minimum, maksimum, dan standar deviasi: $Mean (μ) = Σy / N Min = min(y) Max = max(y) Std Dev (σ) = sqrt(Σ(y - μ)² / N)$

Visualisasi distribusi target dilakukan dengan histogram.

## **4. Deteksi Outlier dengan Interquartile Range (IQR)**
Outlier dihitung berdasarkan rentang interkuartil: **IQR = Q3 - Q1**, **Lower Bound = Q1 - 1.5 * IQR**, **Upper Bound = Q3 + 1.5 * IQR**

Data di luar batas ini dianggap sebagai outlier.

## **5. Identifikasi Tipe Fitur**
Fitur numerik dan kategorikal diidentifikasi: **Numeric Features = X.select_dtypes(include=['int64', 'float64']).columns**, **Categorical Features = X.select_dtypes(include=['object', 'category']).columns**


## **6. Korelasi dengan Target**
Korelasi antara fitur dan target dihitung menggunakan **Pearson Correlation: Corr(x, y) = Cov(x, y) / (Std(x) * Std(y))**

Fitur dengan korelasi tertinggi dengan target dapat diprioritaskan dalam model.

## **7. Preprocessing dan Normalisasi**
Numeric features diproses dengan median imputasi dan normalisasi: **Pipeline([ ('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()) ])**

StandardScaler memastikan semua fitur memiliki rata-rata 0 dan standar deviasi 1.

## **8. Seleksi Fitur dengan f-regression**
Fitur dipilih berdasarkan f-statistic: **F-statistic = (Explained Variance / Unexplained Variance)**

Memilih 50 fitur terbaik menggunakan **SelectKBest(f_regression, k=min(50, len(X.columns)))**.

## **9. Reduksi Dimensi dengan PCA**
Principal Component Analysis (PCA) digunakan untuk mengurangi dimensi sambil mempertahankan 95% varians: **X_pca = PCA(n_components=0.95).fit_transform(X_transformed)**
Grafik varians kumulatif digunakan untuk menentukan jumlah komponen PCA optimal.

## **10. Visualisasi PCA**
Plot scatter dari dua komponen pertama PCA dilakukan:

**plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=30, alpha=0.7)**

Membantu memahami struktur data setelah reduksi dimensi.

---

# **Pembuatan Label Klasifikasi & Pembagian Data**

## **1. Pembuatan Label Klasifikasi Berdasarkan Persentil**
Kode ini membagi data target `y` ke dalam beberapa kategori berdasarkan persentil: **percentiles = [0, 25, 50, 75, 100] bin_edges = np.percentile(y, percentiles)**

Persentil ini membagi data ke dalam 4 kelompok:
- **0-25%** (Kategori 0)
- **25-50%** (Kategori 1)
- **50-75%** (Kategori 2)
- **75-100%** (Kategori 3)

Sehingga setiap nilai dalam `y` diklasifikasikan menggunakan: **y_class = pd.cut(y, bins=bin_edges, labels=[0, 1, 2, 3])**



## **2. Filter Data Valid**
Jika ada nilai NaN dalam `y_class`, maka kita menghapusnya: **valid_idx = ~np.isnan(y_class)**


Menghasilkan subset data yang valid untuk analisis.

## **3. Distribusi Kelas**
Setelah klasifikasi selesai, kita menghitung distribusi kelas: **class_counts = pd.Series(y_class_valid).value_counts().sort_index()**

Ini memberi kita frekuensi setiap kategori.

## **4. Visualisasi Distribusi Label**
Histogram distribusi label dibuat menggunakan: **sns.countplot(x=y_class_valid, palette='viridis')**

Menampilkan jumlah sampel dalam setiap kategori.

## **5. Encoding Label Klasifikasi**
Kategori numerik dikonversi menjadi format yang bisa diproses oleh model machine learning:
- **Label Encoding** mengubah kategori menjadi angka: **label_encoder = LabelEncoder() y_class_encoded = label_encoder.fit_transform(y_class_valid)**

- **One-Hot Encoding** mengubah label ke bentuk vektor: **onehot_encoder = OneHotEncoder(sparse_output=False)**, **y_class_onehot = onehot_encoder.fit_transform(y_class_encoded.reshape(-1, 1)**


## **6. Pembagian Data Train-Test**
Data dibagi menjadi **training (80%)** dan **testing (20%)**: **X_train, X_test, y_train, y_test = train_test_split(X_pca_valid, y_valid, test_size=0.2, random_state=42, stratify=y_class_valid)**

Pembagian ini dilakukan dengan **stratifikasi**, memastikan setiap kategori kelas memiliki proporsi yang sama di training dan testing.

Hal yang sama dilakukan untuk klasifikasi: **X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_pca_valid, y_class_encoded, test_size=0.2, random_state=42, stratify=y_class_valid)**

Target data dikonversi ke bentuk one-hot sebelum digunakan dalam model klasifikasi.

## **7. Final Dataset Shapes**
Dimensi akhir dari dataset setelah preprocessing diperiksa: **print(f"X_train: {X_train.shape}, y_train: {y_train.shape}") print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")**

---

# **Penjelasan Matematika: Model Regresi**

## **1. Definisi Model Regresi**
- Model regresi menggunakan beberapa lapisan **Dense**: **Dense(256, input_shape=(X_train.shape[1],)), Dense(128), Dense(64), Dense(1)**

- Setiap lapisan memiliki **LeakyReLU** sebagai fungsi aktivasi: **LeakyReLU(negative_slope=0.2)**

LeakyReLU mengatasi masalah **vanishing gradient** dengan: $f(x) = x jika x > 0 f(x) = αx jika x < 0 (α kecil)$


## **2. Fungsi Loss: Mean Squared Error (MSE)**
MSE mengukur kesalahan prediksi dengan: $MSE = Σ (y_true - y_pred)² / N$


Digunakan karena menangkap perbedaan antara nilai sebenarnya dan prediksi.

## **3. Optimizer Adam**
Adam optimizer memperbarui bobot model dengan momentum dan adaptasi learning rate: $m_t = β1 m_{t-1} + (1-β1) g_t v_t = β2 v_{t-1} + (1-β2) g_t² θ_t = θ_{t-1} - (α / sqrt(v_t) + ε) * m_t$


Ini mempercepat konvergensi dengan **adaptive learning rate**.

## **4. Callback: Early Stopping dan Reduce LR**
- **Early Stopping** berhenti pelatihan jika validasi loss tidak meningkat setelah beberapa epoch.
- **Reduce LR on Plateau** mengurangi learning rate jika model stagnan.

## **5. Evaluasi Model Regresi**
Model membuat prediksi dengan: **y_pred = regression_model.predict(X_test).flatten()**

Beberapa metrik evaluasi:
- **Mean Absolute Error (MAE)**:
$MAE = Σ y_true - y_pred / N$

- **Root Mean Squared Error (RMSE)**:
$RMSE = sqrt(MSE)$

- **R² Score (Koefisien Determinasi)**:
$R² = 1 - (Σ (y_true - y_pred)² / Σ (y_true - mean(y_true))²)$

Menunjukkan seberapa baik model memprediksi target.

## **6. Visualisasi Evaluasi Model**
- **Scatter Plot** untuk melihat hubungan antara nilai asli vs prediksi.
- **Histogram Kesalahan Prediksi** untuk melihat distribusi error.
- **Grafik Training History** untuk melihat perubahan loss dan MAE selama pelatihan.


## **Evaluasi Model Klasifikasi**:
### 1. **Akurasi**: Rasio prediksi benar terhadap total data.
   $$ \text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}} $$

### 2. **Presisi**: Rasio data positif yang benar terhadap total prediksi positif.
   $$ \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} $$

### 3. **Recall**: Rasio data positif yang benar terhadap total data positif sebenarnya.
   $$ \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} $$

### 4. **F1 Score**: Harmonik antara presisi dan recall.
   $$ \text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} $$

### 5. **AUC (Area Under Curve)**: Area di bawah ROC Curve, mengukur kemampuan model membedakan kelas.
   Nilai ini dihitung berdasarkan integral dari ROC Curve:
   $$ \text{AUC} = \int_0^1 TPR \, dFPR $$

### 6. **ROC Curve**:
   - **True Positive Rate (TPR)**:
     $$ TPR = \frac{\text{TP}}{\text{TP} + \text{FN}} $$
   - **False Positive Rate (FPR)**:
     $$ FPR = \frac{\text{FP}}{\text{FP} + \text{TN}} $$

## **Visualisasi**:
   - Grafik **akurasi** dan **loss** terhadap epoch untuk melihat proses pelatihan.
   - **Confusion Matrix**: Matriks untuk mengukur kesalahan prediksi antara kelas aktual dan prediksi.
     Contoh struktur:
     | Actual \ Predicted | Not Positive | Positive |
     |---------------------|--------------|----------|
     | **Not Positive**    | TN           | FP       |
     | **Positive**        | FN           | TP       |
   - **ROC Curve**: Grafik trade-off antara **True Positive Rate (TPR)** dan **False Positive Rate (FPR)**.
     $$ \text{TPR} = \frac{TP}{TP + FN} $$
     $$ \text{FPR} = \frac{FP}{FP + TN} $$