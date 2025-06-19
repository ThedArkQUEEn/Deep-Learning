# Penjelasan Chapter 10 - Introduction to Artificial Neural Networks with Keras 

---

### 1. Dari Neuron Biologis ke Neuron Buatan

*   **Neuron Biologis**
    *   Menjelaskan struktur dan fungsi dasar neuron biologis (dendrit, soma, akson, sinapsis, potensial aksi) sebagai inspirasi untuk neuron buatan.
*   **Perhitungan Logika dengan Neuron**
    *   Menunjukkan bagaimana neuron sederhana dapat melakukan operasi logika dasar seperti AND, OR, dan NOT.
*   **Perceptron**
    *   Memperkenalkan model neuron buatan tertua, Perceptron.
        *   Ini adalah salah satu model JST yang paling sederhana.
        *   Setiap neuron menerima input numerik, menghitung jumlah bobot dari input tersebut, dan menerapkan fungsi langkah untuk menghasilkan output.
        *   **Rumus Perceptron**: `Output = step_function(sum(weights * inputs) + bias)`
        *   Perceptron tunggal dapat mengklasifikasikan data yang dapat dipisahkan secara linier.
        *   Jaringan Perceptron (Multi-Layer Perceptron) dapat mengatasi masalah non-linier.
*   **Multi-Layer Perceptron (MLP) dan Backpropagation**
    *   Menjelaskan arsitektur MLP (lapisan input, lapisan tersembunyi, lapisan output) dan algoritma Backpropagation untuk melatihnya.
    *   Backpropagation bekerja dengan menghitung gradien kesalahan terhadap setiap bobot di jaringan, kemudian menggunakan gradien ini untuk memperbarui bobot (Gradient Descent).
    *   **Konsep Utama**: Propagasi maju (menghitung output), propagasi mundur (menghitung gradien kesalahan).
*   **MLP Regresi**
    *   MLP dapat digunakan untuk tugas regresi, di mana lapisan output memiliki satu neuron (untuk regresi univariat) dan fungsi aktivasi lapisan output biasanya tidak ada (aktivasi linear).
*   **MLP Klasifikasi**
    *   MLP untuk klasifikasi biasanya memiliki neuron di lapisan output yang sama dengan jumlah kelas, dan menggunakan fungsi aktivasi seperti softmax untuk output probabilitas.
    *   **Fungsi Aktivasi Softmax**: `softmax(z_i) = exp(z_i) / sum(exp(z_j))`

### 2. Mengimplementasikan MLP dengan Keras

*   **Menginstal TensorFlow 2**
    *   Panduan untuk menginstal TensorFlow 2, yang menyertakan Keras sebagai API tingkat tinggi resminya (`tf.keras`).
*   **Membangun Pengklasifikasi Gambar Menggunakan Sequential API**
    *   Langkah-langkah untuk membangun model JST untuk klasifikasi gambar menggunakan Keras `Sequential` API.
    *   Model Sequential adalah tumpukan lapisan linear.
    *   **Contoh Kode**:
        ```python
        import tensorflow as tf
        from tensorflow import keras

        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)), # Contoh input untuk gambar 28x28
            keras.layers.Dense(300, activation="relu"),
            keras.layers.Dense(100, activation="relu"),
            keras.layers.Dense(10, activation="softmax")
        ])
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer="sgd",
                      metrics=["accuracy"])
        # Contoh penggunaan:
        # model.fit(X_train, y_train, epochs=30)
        ```
    *   **Fungsi Aktivasi yang Umum**: ReLU (Rectified Linear Unit), sigmoid, tanh, softmax.
        *   **ReLU**: `max(0, x)`
*   **Membangun MLP Regresi Menggunakan Sequential API**
    *   Contoh membangun MLP untuk tugas regresi. Lapisan output biasanya tidak memiliki fungsi aktivasi (aktivasi linier).
*   **Membangun Model Kompleks Menggunakan Functional API**
    *   Menjelaskan cara membangun model dengan topologi yang lebih kompleks (misalnya, banyak input, banyak output, atau koneksi non-sequential) menggunakan Keras Functional API.
    *   Memungkinkan membangun arsitektur yang lebih fleksibel.
    *   **Konsep**: Input dan output adalah tensor, dan lapisan dipanggil pada tensor ini.
*   **Menggunakan Subclassing API untuk Membangun Model Dinamis**
    *   Untuk arsitektur yang sangat dinamis atau ketika Anda perlu mengimplementasikan logika kustom dalam metode `call()`.
    *   Memungkinkan fleksibilitas maksimum tetapi membutuhkan pemahaman yang lebih dalam tentang Keras.

### 3. Menyimpan dan Memulihkan Model

*   Cara menyimpan bobot model dan arsitektur untuk penggunaan di masa mendatang.
    *   `model.save("my_keras_model.h5")`
    *   `loaded_model = keras.models.load_model("my_keras_model.h5")`

### 4. Menggunakan Callback

*   Menjelaskan callback Keras yang dapat digunakan selama pelatihan untuk berbagai tujuan (misalnya, menyimpan model terbaik, penghentian awal, menyesuaikan laju pembelajaran).
    *   **`ModelCheckpoint`**: Menyimpan model secara berkala.
    *   **`EarlyStopping`**: Menghentikan pelatihan ketika kinerja tidak lagi meningkat pada set validasi.
    *   **`ReduceLROnPlateau`**: Mengurangi laju pembelajaran saat metrik tidak meningkat.

### 5. Menggunakan TensorBoard untuk Visualisasi

*   Panduan untuk menggunakan TensorBoard, alat visualisasi TensorFlow, untuk memantau metrik pelatihan, melihat grafik model, dan menganalisis kinerja.
*   Membutuhkan `TensorBoard` callback saat melatih model.

### 6. Menyetel Hyperparameter Jaringan Saraf

*   **Jumlah Lapisan Tersembunyi**
    *   Diskusi tentang memilih jumlah lapisan tersembunyi yang tepat. Umumnya, satu atau dua lapisan tersembunyi cukup untuk banyak masalah, tetapi masalah yang lebih kompleks mungkin membutuhkan lebih banyak.
*   **Jumlah Neuron per Lapisan Tersembunyi**
    *   Bagaimana menentukan jumlah neuron di setiap lapisan. Pendekatan umum adalah bentuk "piramida" (lebih banyak neuron di awal, lebih sedikit di akhir) atau bentuk "jam pasir" (lebih banyak neuron di lapisan tengah).
*   **Laju Pembelajaran, Ukuran Batch, dan Hyperparameter Lainnya**
    *   Pentingnya menyetel laju pembelajaran, ukuran batch, fungsi aktivasi, optimizer, dll.

---

### Catatan Penting dan Rumus Tambahan dari Rentang Halaman:

*   **Fungsi Aktivasi (`activation`)**:
    *   Selain yang disebutkan di atas, ada juga `tanh`, `sigmoid`, dll. Mereka memperkenalkan non-linearitas ke dalam model, memungkinkan jaringan untuk mempelajari pola yang kompleks.
*   **Fungsi Kerugian (`loss`)**:
    *   Mengukur seberapa buruk kinerja model. Contoh umum:
        *   **`sparse_categorical_crossentropy`**: Untuk klasifikasi multi-kelas ketika label adalah integer.
        *   **`categorical_crossentropy`**: Untuk klasifikasi multi-kelas ketika label adalah one-hot encoded.
        *   **`mse` (Mean Squared Error)**: Untuk regresi. `MSE = (1/N) * sum((y_true - y_pred)^2)`.
*   **Optimizer (`optimizer`)**:
    *   Algoritma yang digunakan untuk memperbarui bobot model berdasarkan gradien yang dihitung.
    *   **`sgd` (Stochastic Gradient Descent)**: Algoritma optimisasi dasar. `weight = weight - learning_rate * gradient`.
    *   Akan ada diskusi lebih lanjut tentang optimizer yang lebih canggih di bab berikutnya (halaman 351).