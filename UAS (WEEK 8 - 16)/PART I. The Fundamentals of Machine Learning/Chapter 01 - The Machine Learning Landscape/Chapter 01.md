# Penjelasan Chapter 01 - The Machine Learning Landscape

## 1. Apa Itu Machine Learning?
Machine Learning (ML) adalah ilmu yang memprogram komputer agar dapat belajar dari data tanpa harus diprogram secara eksplisit.

### Definisi Formal:
Sebuah program komputer dikatakan belajar dari pengalaman (E) terhadap suatu tugas (T) dengan ukuran kinerja (P), jika kinerjanya pada T, yang diukur oleh P, meningkat seiring dengan pengalaman E.

### Contoh Praktis:
- **Tugas (T):** Mengklasifikasikan email baru sebagai spam atau bukan (ham).
- **Pengalaman (E):** Data latih berisi contoh email spam dan ham.
- **Kinerja (P):** Akurasi, yaitu rasio email yang diklasifikasikan dengan benar.

### Mengapa Menggunakan ML?
ML sangat berguna untuk:
- **Masalah dengan Aturan yang Kompleks:** Menggantikan daftar panjang aturan dengan algoritma yang lebih ringkas, akurat, dan mudah diperbarui.
- **Masalah Tanpa Solusi Tradisional:** Menemukan solusi untuk masalah yang tidak memiliki algoritma yang diketahui (contoh: pengenalan suara).
- **Lingkungan yang Berubah-ubah:** Sistem ML dapat beradaptasi dengan data baru secara otomatis.
- **Data Mining:** Membantu menemukan pola atau tren dalam data yang besar.

## 2. Jenis-jenis Sistem Machine Learning
### Berdasarkan Pengawasan Manusia
- **Supervised Learning:** Data latih diberi label, digunakan dalam klasifikasi dan regresi.
- **Unsupervised Learning:** Data latih tidak memiliki label, digunakan dalam clustering, anomaly detection, dimensionality reduction.
- **Semi-supervised Learning:** Kombinasi data berlabel dan tidak berlabel.
- **Reinforcement Learning:** Model belajar dengan interaksi lingkungan dan mendapatkan reward.

### Berdasarkan Kemampuan Belajar
- **Batch Learning:** Model dilatih dengan semua data sekaligus.
- **Online Learning:** Model dilatih secara bertahap dengan data yang diberikan sekuensial.

### Berdasarkan Cara Generalisasi
- **Instance-Based Learning:** Model membandingkan data baru dengan data latih (contoh: k-NN).
- **Model-Based Learning:** Model membangun abstraksi dari data latih untuk membuat prediksi.

## 3. Tantangan Utama dalam Machine Learning
- **Data Tidak Representatif:** Sampling bias dapat menyebabkan generalisasi buruk.
- **Data Berkualitas Buruk:** Error, outlier, dan noise mempersulit analisis.
- **Overfitting:** Model terlalu kompleks dan "menghafal" noise pada data latih.
- **Underfitting:** Model terlalu sederhana dan tidak dapat mengenali pola yang ada.

## 4. Pengujian dan Validasi
- **Training Set & Test Set:** Memisahkan data untuk pelatihan dan pengujian.
- **Validation Set:** Untuk seleksi model dan tuning hyperparameter.
- **Train-Dev Set:** Digunakan untuk mengatasi data mismatch.

## 5. Teorema "No Free Lunch"
Tidak ada model yang bekerja paling baik untuk semua masalah. Setiap model memiliki asumsi tertentu tentang data, sehingga penting untuk mencoba berbagai model dan mengevaluasi hasilnya.

---