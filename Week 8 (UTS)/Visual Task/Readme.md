# **Hasil dan penjelasan klasifikasi 4 class:**

**1. Klasifikasi Ikan - Tensorflow CNN with Batch Norm**
- **Accuracy: 0.9176**
- **Precision: 0.9197**
- **Recall: 0.9176**
- **F1-Score: 0.9171**
- **Score AUC rata-rata seluruh class semuanya hampir 1.00**

**2. Klasifikasi Ikan - Tensorflow CNN without Batch Norm**
- **Accuracy: 0.8852**
- **Precision: 0.8900**
- **Recall: 0.8852**
- **F1-Score: 0.8859**
- **Score AUC rata-rata seluruh class semuanya hampir 1.00**

Hasil validasi dengan Batch Normalization lebih bagus dibanding tanpa Batch Normalization. Preprocessing dan model yang digunakan juga sangat bagus sehingga hasil validasi dapat di atas 0.91 untuk dengan Batch Normalization dan di atas 0.88 untuk tanpa Batch Normalization, walaupun adanya val_accuracy yang turun saat train di epoch 12 dan 14.

---

# **Penjelasan Matematika dari Kode**

## **1. Pembersihan Label**
Kode ini membersihkan label dengan menghapus spasi di awal dan akhir, serta mengubah huruf menjadi kecil:

\text{df['label']} = \text{df['label']}.str.strip().str.lower()

Operasi ini tidak mengubah nilai numerik tetapi hanya mengubah format teks agar lebih seragam.

## **2. Label Encoding**
Label encoding mengubah kategori string menjadi nilai numerik:

\text{label\_encoded} = \text{LabelEncoder().fit\_transform(df['label'])}

Jika ada tiga kelas: `['fish', 'shark', 'whale']`, maka encoding bisa menjadi `{'fish': 0, 'shark': 1, 'whale': 2}`.

## **3. One-hot Encoding**
One-hot encoding mengubah label kategorikal menjadi vektor biner:

\text{one\_hot} = \text{pd.get\_dummies(df['label'])}


Ini penting untuk model pembelajaran mesin karena menghindari interpretasi ordinal dari label.

## **4. Resize dengan Padding**
Fungsi `resize_with_padding(img, target_size=(150, 150))` mengubah ukuran gambar dengan mempertahankan aspek rasio:
\text{ratio} = \min \left(\frac{\text{target\_height}}{\text{old\_height}}, \frac{\text{target\_width}}{\text{old\_width}}\right)
Kemudian, gambar diubah ukuran dan ditambahkan padding agar sesuai dengan ukuran target.

## **5. Custom Directory Iterator**
Iterator khusus ini memproses batch gambar:
- Membaca gambar dengan OpenCV
- Mengonversi ke RGB
- Melakukan normalisasi dengan:
\frac{\text{pixel\_value}}{255.0}
- Jika `class_mode='categorical'`, label dikonversi ke one-hot encoding.

## **6. Augmentasi Data**
Augmentasi diterapkan dengan:
\text{ImageDataGenerator}(zoom\_range=0.2, horizontal\_flip=True, rotation\_range=20, ...)
Ini membantu meningkatkan variasi data untuk model pembelajaran.

---

# **Penjelasan Matematika dari Arsitektur Model CNN**

## **1. Konvolusi dan Pooling**
Model menggunakan beberapa lapisan **konvolusi** dan **max pooling**:

- **Konvolusi 2D**:
\text{Output Shape} = (H_o, W_o, C_o)
dengan:
H_o = \frac{H_i - F}{S} + 1, \quad W_o = \frac{W_i - F}{S} + 1, \quad C_o = N
dimana:
  - H_i, W_i = tinggi dan lebar input
  - F = ukuran filter (kernel)
  - S = stride
  - N = jumlah filter (depth output)

- **MaxPooling 2D**:
H_o = \frac{H_i}{P}, \quad W_o = \frac{W_i}{P}
dengan:
  -  P  = ukuran pooling (contoh:  P = 2  berarti reduksi ukuran separuh)

## **2. Normalisasi Batch**
Batch Normalization:
\hat{x_i} = \frac{x_i - \mu}{\sigma}
memastikan distribusi nilai tetap stabil untuk mempercepat konvergensi.

## **3. Lapisan Dense dan Aktivasi ReLU**
Lapisan fully connected mengubah tensor menjadi vektor:
\mathbf{y} = \sigma (W\mathbf{x} + \mathbf{b})
dengan fungsi aktivasi **ReLU**:
f(x) = \max(0, x)
yang membantu menangani eksploding/vanishing gradient.

## **4. Dropout untuk Regularisasi**
Dropout mengabaikan beberapa neuron selama pelatihan:
h_i = 
\begin{cases}
    0, & \text{dengan probabilitas } p \\
    \frac{h_i}{1 - p}, & \text{selainnya}
\end{cases}
untuk mengurangi overfitting.

## **5. Softmax untuk Klasifikasi**
Lapisan terakhir menggunakan **softmax** untuk klasifikasi multi-kelas:
\hat{y_i} = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
di mana  K  adalah jumlah kelas, memastikan hasil probabilitas.

## **6. Optimasi dengan Adam**
Adam Optimizer mengupdate parameter dengan:
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{v_t} + \epsilon} m_t
dengan **momentum dan adaptasi learning rate**.

## **7. Loss Function: Categorical Crossentropy**
Fungsi loss untuk klasifikasi multi-kelas:
L = - \sum_{i=1}^{K} y_i \log(\hat{y_i})
dengan  K  sebagai jumlah kelas dan  y_i  sebagai label sebenarnya.

## **8. Pelatihan Model**
Model dilatih dengan:
- **Epoch = 15**, iterasi penuh pada dataset
- **Batch size = 128**, jumlah sampel dalam satu iterasi
- **Validation Data**, untuk mengukur performa terhadap data yang tidak dilihat model

---

# **Evaluasi Model**
### 1. **Akurasi**: Rasio prediksi benar terhadap total data.
   \text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}

### 2. **Presisi**: Rasio data positif yang benar terhadap total prediksi positif.
   \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}

### 3. **Recall**: Rasio data positif yang benar terhadap total data positif sebenarnya.
   \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}

### 4. **F1 Score**: Harmonik antara presisi dan recall.
   \text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}

### 5. **AUC (Area Under Curve)**: Area di bawah ROC Curve, mengukur kemampuan model membedakan kelas.
   Nilai ini dihitung berdasarkan integral dari ROC Curve:
   \text{AUC} = \int_0^1 TPR \, dFPR

### 6. **ROC Curve**:
   - **True Positive Rate (TPR)**:
     TPR = \frac{\text{TP}}{\text{TP} + \text{FN}}
   - **False Positive Rate (FPR)**:
     FPR = \frac{\text{FP}}{\text{FP} + \text{TN}}

## **Visualisasi**:
   - Grafik **akurasi** dan **loss** terhadap epoch untuk melihat proses pelatihan.
   - **Confusion Matrix**: Matriks untuk mengukur kesalahan prediksi antara kelas aktual dan prediksi.
     Contoh struktur:
     | Actual \ Predicted | Not Positive | Positive |
     |---------------------|--------------|----------|
     | **Not Positive**    | TN           | FP       |
     | **Positive**        | FN           | TP       |
   - **ROC Curve**: Grafik trade-off antara **True Positive Rate (TPR)** dan **False Positive Rate (FPR)**.
     \text{TPR} = \frac{TP}{TP + FN}
     \text{FPR} = \frac{FP}{FP + TN}