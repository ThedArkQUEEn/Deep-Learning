# Penjelasan Chapter 01 - The Machine Learning Landscape

## Apa itu Machine Learning?

Machine Learning adalah ilmu (dan seni) memprogram komputer agar mereka dapat belajar dari data. Lebih formalnya:

*   **Arthur Samuel (1959):** "Machine Learning adalah bidang studi yang memberikan komputer kemampuan untuk belajar tanpa diprogram secara eksplisit."
*   **Tom Mitchell (1997):** "Program komputer dikatakan belajar dari pengalaman E sehubungan dengan tugas T dan ukuran kinerja P, jika kinerjanya pada T, sebagaimana diukur oleh P, meningkat dengan pengalaman E."

Contoh: Filter spam email. Tugas (T) adalah menandai spam, pengalaman (E) adalah data email pelatihan, dan kinerja (P) adalah akurasi klasifikasi.

## Mengapa Menggunakan Machine Learning?

*   **Solusi kompleks:** ML menyederhanakan masalah yang kompleks atau sulit diselesaikan dengan pemrograman tradisional.
*   **Adaptasi:** Sistem ML dapat beradaptasi dengan data baru dan perubahan lingkungan.
*   **Wawasan:** ML dapat membantu mengungkap pola dan wawasan tersembunyi dalam data.

## Contoh Aplikasi Machine Learning

*   **Klasifikasi gambar:** Mengklasifikasikan gambar produk di jalur produksi.
*   **Deteksi tumor:** Mendeteksi tumor dalam pemindaian otak.
*   **Pemrosesan bahasa alami (NLP):** Mengklasifikasikan artikel berita, menandai komentar ofensif, meringkas dokumen.
*   **Peramalan:** Meramalkan pendapatan perusahaan.
*   **Deteksi penipuan:** Mendeteksi penipuan kartu kredit.
*   **Sistem rekomendasi:** Merekomendasikan produk kepada pelanggan.

## Jenis Sistem Machine Learning

Sistem Machine Learning dapat diklasifikasikan berdasarkan:

### 1.  Supervisi Selama Pelatihan

*   **Supervised Learning (Pembelajaran Terawasi):** Data pelatihan berisi label (solusi yang diinginkan). Contoh: Klasifikasi, Regresi.
*   **Unsupervised Learning (Pembelajaran Tanpa Awasi):** Data pelatihan tidak berlabel. Contoh: Clustering, Reduksi Dimensi, Deteksi Anomali.
*   **Semisupervised Learning (Pembelajaran Semi-Terawasi):** Data pelatihan sebagian berlabel.
*   **Reinforcement Learning (Pembelajaran Penguatan):** Agen belajar dengan berinteraksi dengan lingkungan dan menerima hadiah atau hukuman.

### 2.  Kemampuan Belajar Bertahap

*   **Batch Learning (Pembelajaran Batch):** Sistem belajar dari semua data yang tersedia sekaligus, offline.
*   **Online Learning (Pembelajaran Online):** Sistem belajar secara bertahap dari aliran data yang masuk, online.

### 3.  Metode Generalisasi

*   **Instance-Based Learning (Pembelajaran Berbasis Instans):** Generalisasi dengan membandingkan data baru dengan contoh yang dipelajari menggunakan ukuran kesamaan.
*   **Model-Based Learning (Pembelajaran Berbasis Model):** Generalisasi dengan membangun model dari data pelatihan dan menggunakan model tersebut untuk prediksi.

## Tantangan Utama Machine Learning

*   **Data Pelatihan Tidak Cukup:** Algoritma ML memerlukan banyak data untuk bekerja dengan baik.
*   **Data Pelatihan Tidak Representatif:** Data pelatihan harus mewakili kasus baru yang ingin digeneralisasikan.
*   **Data Berkualitas Buruk:** Data yang penuh kesalahan, outlier, atau noise dapat menghambat pembelajaran.
*   **Fitur Tidak Relevan:** Fitur yang tidak relevan dapat menyulitkan sistem untuk menemukan pola.
*   **Overfitting Data Pelatihan:** Model terlalu cocok dengan data pelatihan dan tidak dapat menggeneralisasi dengan baik untuk data baru.
*   **Underfitting Data Pelatihan:** Model terlalu sederhana dan gagal menangkap pola dasar dalam data.

## Testing dan Validasi

*   Penting untuk mengevaluasi model pada data yang tidak terlihat (set pengujian) untuk mengukur kemampuan generalisasi.
*   Penyetelan hiperparameter dan pemilihan model memerlukan validasi (misalnya, validasi silang).
