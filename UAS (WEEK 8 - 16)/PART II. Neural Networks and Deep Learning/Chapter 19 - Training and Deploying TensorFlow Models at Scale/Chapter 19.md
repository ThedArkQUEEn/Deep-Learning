# Penjelasan Chapter 19 - Training and Deploying TensorFlow Models at Scale

## Menyajikan Model TensorFlow

### Menggunakan TensorFlow Serving

TensorFlow Serving adalah sistem yang fleksibel dan berkinerja tinggi untuk menyajikan model *machine learning* yang telah dilatih. Sistem ini memungkinkan Anda untuk dengan mudah menyebarkan algoritma dan model baru, sambil tetap menjaga arsitektur server dan API yang sama.

**Konsep Utama:**
*   **Servable:** Objek dasar dalam TensorFlow Serving. Setiap *servable* dapat berupa satu atau lebih model yang akan digunakan untuk inferensi. Contohnya adalah `SavedModel`.
*   **Loader:** Mengelola siklus hidup *servable*, termasuk memuat, membongkar, dan menyediakan akses ke *servable*.
*   **Source:** Menemukan dan menyediakan *servable*.
*   **Manager:** Mengelola *servable* dari *Loader*.
*   **Version Policy:** Menentukan kapan versi baru *servable* akan dimuat dan kapan versi lama akan dibongkar.

**Alur Kerja Umum:**
1.  Model dilatih dan diekspor dalam format `SavedModel`.
2.  `SavedModel` ditempatkan di direktori yang dapat diakses oleh TensorFlow Serving.
3.  TensorFlow Serving memuat model dan menyediakannya melalui API gRPC atau REST.
4.  Klien dapat mengirim permintaan inferensi ke TensorFlow Serving.

### Membuat Layanan Prediksi di GCP AI Platform

Google Cloud AI Platform menyediakan layanan terkelola untuk melatih, menyebarkan, dan mengelola model *machine learning*.

**Langkah-langkah umum:**
1.  Ekspor model TensorFlow sebagai `SavedModel`.
2.  Unggah model ke Cloud Storage.
3.  Buat versi model di AI Platform, menunjuk ke lokasi `SavedModel` di Cloud Storage.
4.  Gunakan AI Platform Prediction API untuk mengirim permintaan inferensi.

### Menyebarkan Model ke Perangkat Seluler atau Tertanam

TensorFlow Lite adalah pustaka TensorFlow yang ringan yang dirancang untuk perangkat seluler dan tertanam.

**Proses Konversi dan Optimasi:**
1.  Latih model TensorFlow standar.
2.  Konversi model ke format TensorFlow Lite (`.tflite`) menggunakan TensorFlow Lite Converter.
3.  Optimasi model dapat dilakukan, seperti kuantisasi (mengurangi presisi bobot model untuk ukuran yang lebih kecil dan inferensi yang lebih cepat).

## Menggunakan GPU untuk Mempercepat Komputasi

GPU (Graphics Processing Unit) dapat secara signifikan mempercepat pelatihan model *neural network* karena arsitektur paralelnya yang sangat cocok untuk operasi matriks besar.

### Mendapatkan GPU Sendiri

*   Pembelian kartu grafis diskrit dengan kemampuan CUDA (untuk TensorFlow).

### Menggunakan Mesin Virtual yang Dilengkapi GPU

*   Penyedia *cloud* seperti Google Cloud Platform, AWS, dan Azure menawarkan VM dengan GPU.

### Colaboratory

Google Colaboratory adalah lingkungan notebook Jupyter berbasis *cloud* gratis yang menyediakan akses ke GPU dan TPU.

### Mengelola RAM GPU

Secara *default*, TensorFlow akan mengalokasikan hampir semua memori GPU yang tersedia segera setelah dimulai. Ini dapat mencegah proses lain menggunakan GPU.

**Cara Mengelola Alokasi Memori:**
*   **Pertumbuhan Memori (Memory Growth):** Meminta TensorFlow untuk mengalokasikan memori GPU secara bertahap sesuai kebutuhan.
*   **Alokasi Memori Terbatas:** Membatasi jumlah memori GPU yang dapat digunakan TensorFlow.

### Menempatkan Operasi dan Variabel pada Perangkat

Secara *default*, TensorFlow akan mencoba menempatkan operasi pada perangkat GPU jika tersedia. Anda dapat secara eksplisit menentukan perangkat untuk operasi atau variabel menggunakan `tf.device()`.

### Eksekusi Paralel Lintas Beberapa Perangkat

TensorFlow dapat mengeksekusi operasi secara paralel di beberapa perangkat (CPU dan GPU).

### Melatih Model Lintas Beberapa Perangkat

#### Paralelisme Model (Model Parallelism)

Membagi model menjadi beberapa bagian dan menempatkan setiap bagian pada perangkat yang berbeda. Berguna untuk model yang terlalu besar untuk muat dalam satu memori perangkat.

#### Paralelisme Data (Data Parallelism)

Menduplikasi model di setiap perangkat dan membagi *mini-batch* pelatihan di antara perangkat-perangkat tersebut. Setiap replika model melatih pada bagian datanya sendiri, dan kemudian gradien atau bobot model digabungkan (misalnya, dirata-ratakan). Ini adalah pendekatan yang lebih umum dan efisien untuk pelatihan terdistribusi.

**Konsep:**

*   **Synchronous Training:** Semua replika melatih secara bersamaan, dan gradien digabungkan setelah setiap *mini-batch*.
*   **Asynchronous Training:** Replika melatih secara independen, dan gradien diperbarui secara asinkron (misalnya, *parameter server*).

### Melatih dalam Skala Besar Menggunakan API Strategi Distribusi

TensorFlow menyediakan API Strategi Distribusi (`tf.distribute.Strategy`) untuk memudahkan pelatihan terdistribusi. API ini mendukung berbagai strategi, termasuk:

*   **`tf.distribute.MirroredStrategy`:** Untuk pelatihan paralelisme data sinkron di satu host dengan banyak GPU. Setiap GPU memiliki salinan lengkap dari model, dan gradien dijumlahkan di antara mereka.
*   **`tf.distribute.MultiWorkerMirroredStrategy`:** Untuk pelatihan paralelisme data sinkron di beberapa host, masing-masing dengan satu atau lebih GPU.
*   **`tf.distribute.TPUStrategy`:** Untuk pelatihan di Cloud TPU.

**Langkah-langkah Umum:**

1.  Inisialisasi strategi distribusi yang sesuai.
2.  Buat model dalam konteks strategi (`strategy.scope()`).
3.  Kompilasi model.
4.  Latih model dengan `model.fit()`.

### Melatih Model pada Klaster TensorFlow

Untuk skenario yang lebih kompleks, Anda dapat mengelola klaster TensorFlow secara manual. Klaster terdiri dari beberapa *job*, dan setiap *job* dapat memiliki beberapa *task*.

**Konfigurasi Klaster:**

*   **`TF_CONFIG` Lingkungan Variabel:** Digunakan untuk mengkonfigurasi klaster dan peran setiap *task* (misalnya, *worker*, *chief*, *evaluator*, *parameter server*).

### Menjalankan Pekerjaan Pelatihan Besar di Google Cloud AI Platform

AI Platform Training menyediakan layanan terkelola untuk melatih model secara terdistribusi.

**Fitur:**

*   **Skala Otomatis:** Sumber daya penskalaan otomatis berdasarkan kebutuhan pekerjaan.
*   **Manajemen Hyperparameter:** Kemampuan untuk melakukan *hyperparameter tuning* dengan Black Box Hyperparameter Tuning.
*   **Pemantauan:** Pemantauan pekerjaan pelatihan dengan TensorBoard.

### Black Box Hyperparameter Tuning di AI Platform

AI Platform menyediakan layanan *hyperparameter tuning* yang menggunakan algoritma *black box optimization* untuk menemukan kombinasi *hyperparameter* terbaik secara otomatis. Ini mengurangi kebutuhan untuk menguji setiap kombinasi secara manual.

**Konsep:**

*   **Trial:** Satu set *hyperparameter* yang diuji dalam satu pekerjaan pelatihan.
*   **Goal:** Metrik yang ingin dioptimalkan (misalnya, akurasi, *loss*).
*   **Parameter Space:** Rentang dan jenis *hyperparameter* yang akan dieksplorasi.