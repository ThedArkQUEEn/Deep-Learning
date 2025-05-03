# Hasil dan Penjelasan Klasifikasi 4 Kelas

## 1. Klasifikasi Ikan - TensorFlow CNN dengan Batch Normalization

**Metrik Evaluasi:**
- **Akurasi (Accuracy):** 0.9176
- **Presisi (Precision):** 0.9197
- **Recall:** 0.9176
- **F1-Score:** 0.9171
- **AUC Score (Rata-rata seluruh kelas):** Hampir 1.00

## 2. Klasifikasi Ikan - TensorFlow CNN tanpa Batch Normalization

**Metrik Evaluasi:**
- **Akurasi (Accuracy):** 0.8852
- **Presisi (Precision):** 0.8900
- **Recall:** 0.8852
- **F1-Score:** 0.8859
- **AUC Score (Rata-rata seluruh kelas):** Hampir 1.00

**Analisis Hasil Validasi:**

Hasil validasi menunjukkan bahwa model dengan *Batch Normalization* memberikan performa yang lebih baik dibandingkan dengan model tanpa *Batch Normalization*. Penggunaan *preprocessing* data yang efektif dan arsitektur model yang baik menghasilkan skor validasi di atas 0.91 untuk model dengan *Batch Normalization* dan di atas 0.88 untuk model tanpa *Batch Normalization*.

Meskipun demikian, perlu diperhatikan adanya penurunan pada `val_accuracy` selama pelatihan pada *epoch* 12 dan 14. Hal ini mungkin mengindikasikan adanya fluktuasi dalam proses pembelajaran yang perlu diinvestigasi lebih lanjut.

---

# Penjelasan Matematika dari Kode

## 1. Pembersihan Label

Operasi pembersihan label dilakukan untuk menyeragamkan format label dengan menghilangkan spasi di awal dan akhir string serta mengubah seluruh karakter menjadi huruf kecil. Persamaan berikut merepresentasikan operasi ini:

$\text{df['label']} = \text{df['label']}.str.strip().str.lower()$

Operasi ini memastikan konsistensi data teks tanpa mengubah nilai intrinsik dari label.

## 2. Label Encoding

*Label encoding* adalah proses konversi label kategorikal (string) menjadi representasi numerik. Proses ini diimplementasikan menggunakan `LabelEncoder` dari scikit-learn:

$\text{label\_encoded} = \text{LabelEncoder().fit\_transform(df['label'])}$

Contoh, jika kita memiliki kelas `['ikan', 'hiu', 'paus']`, *label encoding* dapat menghasilkan pemetaan seperti `{'ikan': 0, 'hiu': 1, 'paus': 2}`.

## 3. One-hot Encoding

*One-hot encoding* mengubah label kategorikal menjadi vektor biner. Setiap kategori diubah menjadi kolom baru, dan keberadaan kategori tersebut ditandai dengan nilai 1, sementara kategori lainnya ditandai dengan 0. Operasi ini dilakukan menggunakan `pd.get_dummies()` dari pandas:

$\text{one\_hot} = \text{pd.get\_dummies(df['label'])}$

Sebagai contoh, jika kita memiliki tiga kelas, label 'ikan' akan direpresentasikan sebagai `[1, 0, 0]`, 'hiu' sebagai `[0, 1, 0]`, dan 'paus' sebagai `[0, 0, 1]`.

## 4. Resize dengan Padding

Fungsi `resize_with_padding(img, target_size=(150, 150))` bertujuan untuk mengubah ukuran gambar ke dimensi target sambil mempertahankan rasio aspek aslinya. Padding ditambahkan pada sisi gambar yang lebih pendek untuk mencapai ukuran target. Rasio skala dihitung sebagai:

$
\text{ratio} = \min \left(\frac{\text{target\_height}}{\text{old\_height}}, \frac{\text{target\_width}}{\text{old\_width}}\right)
$

Setelah gambar diubah ukurannya berdasarkan rasio ini, padding (biasanya dengan nilai 0) ditambahkan di sekeliling gambar untuk mencapai `target_size`.

## 5. Custom Directory Iterator

Iterator khusus ini bertugas untuk menghasilkan batch gambar dan label yang siap untuk pelatihan model. Langkah-langkah utama dalam iterator ini adalah:
- Membaca gambar dari direktori menggunakan OpenCV.
- Mengonversi format warna gambar menjadi RGB.
- Melakukan normalisasi nilai piksel ke rentang [0, 1] dengan membagi setiap nilai piksel dengan 255.0:
  $$
  \text{normalized\_pixel\_value} = \frac{\text{pixel\_value}}{255.0}
  $$
- Jika `class_mode` diatur sebagai 'categorical', label akan diubah menjadi representasi *one-hot encoding*.

## 6. Augmentasi Data

*Augmentasi data* adalah teknik untuk meningkatkan variasi dalam dataset pelatihan dengan menerapkan transformasi acak pada gambar. `ImageDataGenerator` dari Keras digunakan untuk tujuan ini:

$
\text{ImageDataGenerator}(zoom\_range=0.2, horizontal\_flip=True, rotation\_range=20, ...)
$

Parameter seperti `zoom_range`, `horizontal_flip`, dan `rotation_range` menentukan jenis dan besaran transformasi yang diterapkan secara acak pada setiap batch gambar selama pelatihan. Ini membantu model untuk menjadi lebih robust terhadap variasi dalam data input.

---

# Penjelasan Matematika dari Arsitektur Model CNN

## 1. Konvolusi dan Pooling

Arsitektur model CNN yang digunakan terdiri dari beberapa lapisan **konvolusi** dan **max pooling**.

- **Lapisan Konvolusi 2D (Conv2D)**: Lapisan ini menerapkan serangkaian filter (kernel) untuk mengekstrak fitur spasial dari gambar input. Bentuk output dari lapisan konvolusi dapat dihitung sebagai:

  $
  H_o = \frac{H_i - F}{S} + 1, \quad W_o = \frac{W_i - F}{S} + 1, \quad C_o = N
  $

  dengan:
  - $H_i, W_i$ = tinggi dan lebar input
  - $F$ = ukuran filter (kernel)
  - $S$ = *stride* (langkah pergerakan filter)
  - $N$ = jumlah filter (kedalaman output)

- **Lapisan MaxPooling 2D**: Lapisan ini mengurangi dimensi spasial dari peta fitur yang dihasilkan oleh lapisan konvolusi. Operasi *max pooling* mengambil nilai maksimum dari jendela piksel dan menghasilkan peta fitur yang lebih kecil:

  $
  H_o = \frac{H_i}{P}, \quad W_o = \frac{W_i}{P}
  $

  dengan:
  - $P$ = ukuran jendela *pooling* (misalnya, $P=2$ berarti reduksi ukuran menjadi setengahnya).

## 2. Normalisasi Batch (Batch Normalization)

*Batch Normalization* adalah teknik untuk menstabilkan proses pembelajaran dengan menormalkan output dari lapisan sebelumnya. Untuk setiap fitur dalam batch, rata-rata ($\mu$) dan standar deviasi ($\sigma$) dihitung, dan kemudian nilai fitur dinormalisasi:

$
\hat{x_i} = \frac{x_i - \mu}{\sigma}
$

Lapisan ini membantu dalam mempercepat konvergensi dan membuat pelatihan model lebih stabil.

## 3. Lapisan Dense dan Aktivasi ReLU

- **Lapisan Dense (Fully Connected)**: Lapisan ini menghubungkan setiap neuron di lapisan sebelumnya dengan setiap neuron di lapisan saat ini. Operasi dalam lapisan dense dapat direpresentasikan sebagai:

  $
  \mathbf{y} = \sigma (W\mathbf{x} + \mathbf{b})
  $

  dengan:
  - $\mathbf{x}$ = vektor input
  - $W$ = matriks bobot
  - $\mathbf{b}$ = vektor bias
  - $\sigma$ = fungsi aktivasi

- **Fungsi Aktivasi ReLU (Rectified Linear Unit)**: ReLU adalah fungsi aktivasi non-linear yang umum digunakan dalam jaringan saraf tiruan:

  $
  f(x) = \max(0, x)
  $

  ReLU membantu dalam mengatasi masalah *vanishing gradient* dan mempercepat pelatihan.

## 4. Dropout untuk Regularisasi

*Dropout* adalah teknik regularisasi yang secara acak mengatur sebagian neuron menjadi 0 selama pelatihan. Ini mencegah neuron untuk terlalu beradaptasi pada fitur tertentu dari data pelatihan dan membantu mengurangi *overfitting*. Jika suatu neuron dengan output $h_i$ memiliki probabilitas $p$ untuk di-*dropout*, maka outputnya selama pelatihan adalah:

$
h_i =
\begin{cases}
    0, & \text{dengan probabilitas } p \\
    \frac{h_i}{1 - p}, & \text{selainnya}
\end{cases}
$

Faktor $\frac{1}{1-p}$ diterapkan selama pelatihan untuk mempertahankan skala output yang diharapkan selama inferensi.

## 5. Softmax untuk Klasifikasi

Lapisan terakhir dalam model klasifikasi multi-kelas biasanya menggunakan fungsi aktivasi *softmax*. Fungsi ini mengubah vektor skor mentah menjadi distribusi probabilitas atas semua kelas:

$
\hat{y_i} = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$

di mana:
- $\hat{y_i}$ adalah probabilitas kelas ke-$i$.
- $z_i$ adalah skor keluaran (logits) untuk kelas ke-$i$.
- $K$ adalah jumlah total kelas.

Fungsi *softmax* memastikan bahwa output dari lapisan terakhir adalah vektor probabilitas di mana semua elemen non-negatif dan berjumlah 1.

## 6. Optimasi dengan Adam

*Adam (Adaptive Moment Estimation)* adalah algoritma optimasi yang umum digunakan untuk melatih jaringan saraf tiruan. Adam menggabungkan ide dari *momentum* dan *RMSProp* untuk mengadaptasi *learning rate* untuk setiap parameter model secara individual. Pembaruan parameter dalam algoritma Adam melibatkan perhitungan *first moment* (rata-rata eksponensial dari gradien) dan *second moment* (rata-rata eksponensial dari gradien kuadrat):

$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \\
\hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$

di mana:
- $g_t$ adalah gradien pada langkah $t$.
- $\beta_1$ dan $\beta_2$ adalah koefisien peluruhan untuk estimasi momen.
- $\alpha$ adalah *learning rate*.
- $\epsilon$ adalah konstanta kecil untuk mencegah pembagian dengan nol.

## 7. Loss Function: Categorical Crossentropy

*Categorical Crossentropy* adalah fungsi *loss* yang umum digunakan untuk masalah klasifikasi multi-kelas dengan label dalam format *one-hot encoding*. Fungsi ini mengukur perbedaan antara distribusi probabilitas yang diprediksi ($\hat{y}$) dan distribusi probabilitas sebenarnya ($y$):

$
L = - \sum_{i=1}^{K} y_i \log(\hat{y_i})
$

di mana:
- $K$ adalah jumlah kelas.
- $y_i$ adalah 1 jika sampel termasuk dalam kelas $i$ dan 0 sebaliknya (untuk label sebenarnya).
- $\hat{y_i}$ adalah probabilitas yang diprediksi untuk kelas $i$.

Tujuan dari pelatihan model adalah untuk meminimalkan nilai *loss* ini.

## 8. Pelatihan Model

Proses pelatihan model melibatkan beberapa parameter penting:
- **Epoch**: Satu *epoch* adalah satu iterasi lengkap melalui seluruh dataset pelatihan. Dalam kasus ini, model dilatih selama 15 *epoch*.
- **Batch Size**: *Batch size* menentukan jumlah sampel yang digunakan dalam satu iterasi pelatihan untuk menghitung gradien dan memperbarui bobot model. Di sini, *batch size* adalah 128.
- **Validation Data**: Selama pelatihan, sebagian data (yang tidak digunakan untuk pelatihan) digunakan sebagai data validasi untuk memantau performa model pada data yang belum pernah dilihat sebelumnya. Ini membantu dalam mendeteksi *overfitting*.

---

# Evaluasi Model

Evaluasi model dilakukan menggunakan berbagai metrik untuk mengukur performanya dalam tugas klasifikasi.

### 1. Akurasi (Accuracy)

Akurasi adalah rasio prediksi yang benar terhadap total jumlah prediksi.

$$
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
$$

di mana:
- TP (True Positive) = Jumlah prediksi positif yang benar.
- TN (True Negative) = Jumlah prediksi negatif yang benar.
- FP (False Positive) = Jumlah prediksi positif yang salah.
- FN (False Negative) = Jumlah prediksi negatif yang salah.

### 2. Presisi (Precision)

Presisi adalah rasio prediksi positif yang benar terhadap total jumlah prediksi positif. Ini mengukur seberapa akurat model dalam mengidentifikasi kelas positif.

$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$

### 3. Recall (Sensitivity atau True Positive Rate)

Recall adalah rasio prediksi positif yang benar terhadap total jumlah data positif sebenarnya. Ini mengukur kemampuan model untuk menemukan semua contoh positif.

$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

### 4. F1 Score

F1 Score adalah rata-rata harmonik antara presisi dan recall. Ini memberikan ukuran tunggal yang menyeimbangkan presisi dan recall.

$$
\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

### 5. AUC (Area Under the ROC Curve)

AUC mengukur kemampuan model untuk membedakan antara kelas-kelas. Ini adalah area di bawah Kurva ROC (Receiver Operating Characteristic). Nilai AUC berkisar antara 0 dan 1, di mana nilai yang lebih tinggi menunjukkan performa model yang lebih baik.

$$
\text{AUC} = \int_0^1 TPR \, dFPR
$$

### 6. ROC Curve (Receiver Operating Characteristic Curve)

Kurva ROC adalah grafik yang memplot True Positive Rate (TPR) terhadap False Positive Rate (FPR) pada berbagai ambang batas klasifikasi.

- **True Positive Rate (TPR) atau Recall**:

  $$
  TPR = \frac{\text{TP}}{\text{TP} + \text{FN}}
  $$

- **False Positive Rate (FPR)**:

  $$
  FPR = \frac{\text{FP}}{\text{FP} + \text{TN}}
  $$

## Visualisasi

Visualisasi memainkan peran penting dalam memahami dan mengevaluasi performa model. Beberapa visualisasi umum meliputi:

- **Grafik Akurasi dan Loss terhadap Epoch**: Grafik ini menunjukkan bagaimana akurasi dan fungsi loss model berubah selama proses pelatihan. Ini membantu dalam mengidentifikasi apakah model sedang *overfitting* atau *underfitting*.

- **Confusion Matrix**: Matriks ini menunjukkan jumlah prediksi yang benar dan salah untuk setiap kelas. Baris matriks mewakili kelas aktual, sedangkan kolom mewakili kelas yang diprediksi.

  Contoh Struktur Confusion Matrix:

  | Aktual \ Prediksi | Bukan Positif | Positif |
  |-----------------|---------------|---------|
  | **Bukan Positif** | TN            | FP      |
  | **Positif** | FN            | TP      |

  Untuk klasifikasi multi-kelas, matriks ini diperluas untuk mencakup semua kelas.

- **ROC Curve**: Grafik ini memvisualisasikan trade-off antara TPR dan FPR untuk berbagai ambang batas klasifikasi. Area di bawah kurva (AUC) memberikan ukuran agregat dari performa model. Untuk klasifikasi multi-kelas, kurva ROC dapat diplot untuk setiap kelas (One-vs-Rest).