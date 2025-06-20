# Penjelasan Chapter 13 -  Loading and Preprocessing Data with TensorFlow

Bab ini membahas cara memuat dan memproses data secara efisien menggunakan API `tf.data` dan berbagai teknik pra-pemrosesan yang disediakan oleh TensorFlow dan Keras.

## The Data API

TensorFlow Data API (`tf.data`) adalah cara yang efisien untuk membangun *pipeline* input yang kompleks dari data sederhana. Ini memungkinkan Anda menangani kumpulan data yang besar yang tidak muat dalam memori dan menerapkan transformasi yang efisien.

### Merangkai Transformasi

Anda dapat merangkai berbagai transformasi dataset dengan memanggil metode seperti `map()`, `batch()`, `shuffle()`, dan `prefetch()`.

### Mengacak Data

Untuk memastikan bahwa model menerima *batch* data yang beragam selama pelatihan, Anda perlu mengacak dataset.

Rumus:
- **`dataset.shuffle(buffer_size)`**: Mengacak elemen dataset. `buffer_size` menentukan ukuran buffer dari mana elemen akan diambil secara acak. Buffer yang lebih besar memberikan pengacakan yang lebih baik tetapi membutuhkan lebih banyak memori.

### Pra-pemrosesan Data

Pra-pemrosesan data melibatkan berbagai langkah untuk mengubah data mentah ke format yang lebih sesuai untuk model pembelajaran mesin.

### Menyatukan Semuanya

Membangun *pipeline* input yang lengkap melibatkan penggabungan langkah-langkah seperti memuat data, memecahnya menjadi *batch*, mengacaknya, dan melakukan pra-pemrosesan.

### Prefetching

`prefetch()` memungkinkan *pipeline* data untuk menghasilkan *batch* di latar belakang saat model sedang melatih *batch* saat ini, sehingga mengurangi latensi dan menjaga GPU tetap sibuk.

Rumus:
- **`dataset.prefetch(buffer_size)`**: Memungkinkan *pipeline* untuk mengambil elemen berikutnya saat elemen saat ini sedang diproses. `buffer_size` adalah jumlah *batch* yang akan di-prefetch.

### Menggunakan Dataset dengan tf.keras

Dataset `tf.data` dapat dengan mudah diintegrasikan dengan API `tf.keras` untuk melatih model.

Rumus:
- **`model.fit(dataset, ...)`**: Melatih model Keras menggunakan objek dataset.

## Format TFRecord

TFRecord adalah format biner sederhana untuk data pelatihan yang dapat dioptimalkan untuk kinerja TensorFlow. Ini menyimpan data sebagai urutan catatan biner.

### File TFRecord Terkompresi

TFRecord mendukung kompresi untuk menghemat ruang disk, yang bisa sangat berguna untuk kumpulan data yang sangat besar.

### Pengenalan Singkat tentang Protocol Buffers

Protocol Buffers adalah mekanisme Google yang agnostik terhadap bahasa dan platform, dapat diperluas, untuk menserialisasi data terstruktur. TFRecord dibangun di atas Protocol Buffers.

### TensorFlow Protobufs

TensorFlow memiliki definisi Protocol Buffer khusus untuk mewakili fitur-fitur dalam dataset.

- **`tf.train.Example`**: Sebuah Protocol Buffer yang merepresentasikan sebuah instance dalam dataset, yang terdiri dari satu atau lebih fitur.
- **`tf.train.Features`**: Berisi peta dari nama fitur (string) ke `tf.train.Feature`.
- **`tf.train.Feature`**: Dapat berupa `BytesList`, `FloatList`, atau `Int64List`.

### Memuat dan Memparsing Contoh

Data yang disimpan dalam format TFRecord perlu diparse kembali ke tensor.

Rumus:
- **`tf.io.TFRecordDataset(filepaths)`**: Membuat dataset dari file TFRecord.
- **`tf.io.parse_single_example(serialized_example, feature_description)`**: Memparsing satu contoh serial menjadi tensor.
- **`tf.io.parse_example(serialized_examples, feature_description)`**: Memparsing beberapa contoh serial.

### Menangani Daftar Daftar Menggunakan Protobuf SequenceExample

Untuk data sekuensial (seperti urutan kata), `tf.train.SequenceExample` dapat digunakan. Ini memungkinkan penyimpanan data konteks (fitur tunggal) dan data sekuensial (daftar fitur).

### Pra-pemrosesan Fitur Input

Sebelum memasukkan fitur ke model, mereka sering membutuhkan pra-pemrosesan lebih lanjut.

### Mengkodekan Fitur Kategorikal Menggunakan One-Hot Vectors

Fitur kategorikal (misalnya, kota, kategori produk) perlu dikonversi menjadi representasi numerik. Salah satu cara adalah dengan menggunakan *one-hot encoding*.

Rumus:
- **One-Hot Encoding**: Untuk fitur kategorikal dengan N kategori, setiap kategori diwakili oleh vektor biner N-dimensi dengan 1 pada indeks kategorinya dan 0 di tempat lain.

### Mengkodekan Fitur Kategorikal Menggunakan Embeddings

Untuk fitur kategorikal dengan banyak kategori, *embeddings* dapat menjadi alternatif yang lebih efisien daripada *one-hot encoding*. *Embeddings* adalah vektor padat berdimensi rendah yang mewakili kategori.

### Keras Preprocessing Layers

Keras menyediakan lapisan pra-pemrosesan bawaan yang dapat disertakan langsung dalam model, memungkinkan pra-pemrosesan terjadi sebagai bagian dari grafik model.

## TF Transform

TF Transform adalah pustaka untuk pra-pemrosesan data dengan grafik TensorFlow. Ini sangat berguna untuk pra-pemrosesan yang perlu dilakukan pada data lengkap (misalnya, normalisasi berdasarkan rata-rata dan standar deviasi dari seluruh kumpulan data).

## Proyek TensorFlow Datasets (TFDS)

TensorFlow Datasets (TFDS) adalah koleksi dataset siap pakai dalam format `tf.data.Dataset`, sehingga mudah untuk diimpor dan digunakan tanpa perlu penanganan data manual.