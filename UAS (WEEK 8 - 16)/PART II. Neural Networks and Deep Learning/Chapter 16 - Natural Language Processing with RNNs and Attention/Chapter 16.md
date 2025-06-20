# Penjelasan Chapter 16 - Natural Language Processing with RNNs and Attention

### Neuron dan Lapisan Berulang (Recurrent Neurons and Layers)
*   **Neuron berulang**: Berbeda dengan neuron feedforward, neuron berulang memiliki memori internal yang memungkinkan mereka mempertahankan state dari langkah waktu sebelumnya. Hal ini memungkinkan mereka untuk memproses urutan data, di mana output saat ini bergantung pada input saat ini dan input masa lalu.
*   **Rumus sederhana neuron berulang**:
    $h_t = f(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$
    $y_t = g(W_{hy} h_t + b_y)$
    Dimana:
    *   $h_t$: state tersembunyi pada langkah waktu $t$
    *   $x_t$: input pada langkah waktu $t$
    *   $y_t$: output pada langkah waktu $t$
    *   $W_{hh}, W_{xh}, W_{hy}$: bobot
    *   $b_h, b_y$: bias
    *   $f, g$: fungsi aktivasi

### Sel Memori (Memory Cells)
*   Unit yang lebih kompleks daripada neuron berulang sederhana, dirancang untuk mengatasi masalah memori jangka pendek. Contohnya adalah Long Short-Term Memory (LSTM) dan Gated Recurrent Unit (GRU).

### Urutan Input dan Output (Input and Output Sequences)
*   **Sequence-to-sequence (encoder-decoder)**: Input berupa urutan, output berupa urutan (misalnya, terjemahan mesin).
*   **Sequence-to-vector**: Input berupa urutan, output berupa vektor tunggal (misalnya, klasifikasi sentimen).
*   **Vector-to-sequence**: Input berupa vektor tunggal, output berupa urutan (misalnya, pembuatan teks dari gambar).

### Melatih RNN (Training RNNs)
*   **Backpropagation Through Time (BPTT)**: Algoritma yang digunakan untuk melatih RNN. Ini melibatkan unrolling jaringan berulang untuk sejumlah langkah waktu dan kemudian menerapkan backpropagation standar.

### Penanganan Urutan Panjang (Handling Long Sequences)
*   **Masalah Vanishing/Exploding Gradients**: Gradien cenderung menjadi sangat kecil (vanishing) atau sangat besar (exploding) seiring waktu, yang menyulitkan pelatihan RNN pada urutan panjang.
*   **Mengatasi Vanishing/Exploding Gradients**:
    *   **Glorot/He Initialization**: Inisialisasi bobot yang tepat.
    *   **Nonsaturating Activation Functions**: Menggunakan fungsi aktivasi seperti ReLU.
    *   **Batch Normalization**: Menormalisasi input setiap lapisan.
    *   **Gradient Clipping**: Membatasi ukuran gradien untuk mencegah exploding gradients.
    *   **Tackling Short-Term Memory Problem (LSTM/GRU)**: Menggunakan sel memori yang lebih canggih.

## Bab 16: Pemrosesan Bahasa Alami dengan RNN dan Atensi

### Membuat Teks Shakespearean Menggunakan Character RNN
*   **Dataset Pelatihan**: Teks dibagi menjadi urutan karakter.
*   **Membuat Dataset Sekuensial**: Memecah urutan panjang menjadi jendela-jendela yang lebih kecil untuk pelatihan.

### Membangun dan Melatih Model Char-RNN
*   Menggunakan arsitektur RNN (misalnya, LSTM) untuk memprediksi karakter berikutnya dalam urutan.

### Menggunakan Model Char-RNN
*   **Menghasilkan Teks Shakespearean Palsu**: Model dapat menghasilkan teks baru berdasarkan pola yang dipelajarinya.
*   **Stateful RNN**: Model mempertahankan state tersembunyinya di antara batch, memungkinkan pembelajaran pada urutan yang sangat panjang.

### Analisis Sentimen (Sentiment Analysis)
*   Mengklasifikasikan polaritas sentimen dari teks.
*   **Masking**: Mengabaikan langkah waktu tertentu dalam urutan (misalnya, padding nol) saat menghitung loss atau akurasi.

### Menggunakan Embeddings yang Sudah Dilatih (Reusing Pretrained Embeddings)
*   Menggunakan representasi kata yang sudah dilatih (word embeddings) seperti Word2Vec atau GloVe untuk meningkatkan kinerja NLP.
*   Word embeddings menangkap hubungan semantik antara kata-kata.

### Jaringan Encoder-Decoder untuk Neural Machine Translation (NMT)
*   **Arsitektur Encoder-Decoder**:
    *   **Encoder**: Memproses urutan input (bahasa sumber) menjadi representasi kontekstual tunggal.
    *   **Decoder**: Menghasilkan urutan output (bahasa target) dari representasi kontekstual yang diberikan oleh encoder.
*   Seringkali menggunakan RNN (LSTM atau GRU) di kedua sisi.

### RNN Bidirectional (Bidirectional RNNs)
*   Memproses urutan dalam dua arah (maju dan mundur) untuk menangkap informasi dari kedua sisi konteks, yang bermanfaat untuk tugas-tugas seperti terjemahan mesin atau pengenalan entitas bernama.

### Beam Search
*   Teknik pencarian yang digunakan dalam decoding urutan untuk menemukan urutan output yang paling mungkin, dengan mempertimbangkan beberapa kandidat pada setiap langkah.

### Mekanisme Atensi (Attention Mechanisms)
*   Memungkinkan model untuk fokus pada bagian-bagian yang relevan dari input saat memproses urutan output. Mengatasi masalah memori jangka panjang dalam RNN untuk urutan yang sangat panjang.
*   **Visual Attention**: Mekanisme atensi juga dapat diterapkan pada tugas visi komputer.
*   **Rumus sederhana atensi**:
    *   Menghitung skor atensi antara output decoder saat ini dan setiap state tersembunyi encoder.
    *   Menerapkan softmax pada skor untuk mendapatkan bobot atensi.
    *   Menghitung vektor konteks sebagai rata-rata tertimbang dari state tersembunyi encoder menggunakan bobot atensi.
    $score(s_i, h_j) = s_i^T h_j$ (dot product attention)
    $attention\_weights = \text{softmax}(scores)$
    $context\_vector = \sum attention\_weights_j \cdot h_j$

### Atensi adalah yang Anda Butuhkan: Arsitektur Transformer
*   **Transformer**: Arsitektur jaringan saraf yang sepenuhnya mengandalkan mekanisme atensi (Self-Attention) dan sepenuhnya menghilangkan RNN.
*   **Keunggulan Transformer**:
    *   **Paralelisasi**: Memungkinkan pelatihan yang lebih cepat karena tidak ada ketergantungan sekuensial seperti pada RNN.
    *   **Penanganan Dependensi Jarak Jauh**: Lebih baik dalam menangkap dependensi jarak jauh dalam urutan.
*   **Komponen Kunci Transformer**:
    *   **Multi-Head Self-Attention**: Memungkinkan model untuk memperhatikan bagian-bagian berbeda dari urutan input secara bersamaan.
    *   **Positional Encoding**: Menambahkan informasi posisi ke embedding input karena Transformer tidak memiliki memori posisi intrinsik.

### Inovasi Terbaru dalam Model Bahasa (Recent Innovations in Language Models)
*   Model bahasa besar seperti BERT, GPT (Generative Pre-trained Transformer), dan lainnya yang mengandalkan arsitektur Transformer dan dilatih pada korpus teks yang sangat besar, menunjukkan kemampuan yang luar biasa dalam berbagai tugas NLP.