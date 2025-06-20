# Penjelasan Chapter 15 - Processing Sequences Using RNNs and CNNs

## Neuron dan Lapisan Rekuren

Neuron rekuren, tidak seperti neuron feedforward, memiliki koneksi dari outputnya sendiri kembali ke inputnya. Ini memungkinkan neuron untuk "mengingat" input sebelumnya, yang penting untuk memproses urutan data.

*   **Rumus sederhana neuron rekuren:**
    $$h_t = f(W_x x_t + W_h h_{t-1} + b)$$
    Dimana:
    *   $h_t$: state tersembunyi pada waktu $t$
    *   $x_t$: input pada waktu $t$
    *   $W_x$: matriks bobot untuk input
    *   $W_h$: matriks bobot untuk state tersembunyi sebelumnya
    *   $b$: bias
    *   $f$: fungsi aktivasi

## Sel Memori

Dalam Recurrent Neural Networks (RNNs), "sel memori" adalah unit komputasi yang dapat mempertahankan informasi untuk periode waktu yang lebih lama. Contoh umum adalah Long Short-Term Memory (LSTM) dan Gated Recurrent Unit (GRU).

## Urutan Input dan Output

RNN dapat menangani berbagai jenis urutan input dan output:
*   **Sequence-to-Vector:** Input berupa urutan, output berupa vektor tunggal (misal: klasifikasi sentimen).
*   **Vector-to-Sequence:** Input berupa vektor tunggal, output berupa urutan (misal: image captioning).
*   **Sequence-to-Sequence (Encoder-Decoder):** Input berupa urutan, output berupa urutan lain (misal: terjemahan mesin).
*   **Synchronized Sequence-to-Sequence:** Input dan output adalah urutan dengan panjang yang sama, diproses secara sinkron (misal: prediksi deret waktu).

## Melatih RNN

Melatih RNN lebih kompleks daripada melatih jaringan feedforward karena sifat rekurensinya.
*   **Backpropagation Through Time (BPTT):** Ini adalah teknik yang digunakan untuk melatih RNN. Ini melibatkan "membuka" jaringan sepanjang dimensi waktu dan menerapkan backpropagation.

## Memprediksi Deret Waktu

Deret waktu adalah urutan titik data yang diindeks dalam urutan waktu. RNN sangat cocok untuk tugas-tugas prediksi deret waktu.

## Metrik Baseline

Dalam prediksi deret waktu, penting untuk memiliki metrik baseline untuk membandingkan performa model Anda. Metrik sederhana bisa berupa memprediksi nilai sebelumnya (naive forecast).

## Mengimplementasikan RNN Sederhana

Implementasi RNN sederhana biasanya melibatkan lapisan `SimpleRNN` di Keras, di mana output dari langkah waktu sebelumnya diumpankan sebagai input ke langkah waktu berikutnya bersama dengan input saat ini.

## Deep RNN

Deep RNN tersusun dari beberapa lapisan rekuren. Ini memungkinkan model untuk mempelajari fitur-fitur yang lebih kompleks dari urutan data.

## Memprediksi Beberapa Langkah Waktu ke Depan

Untuk memprediksi beberapa langkah waktu ke depan, ada beberapa strategi:
*   **Single-shot prediction:** Melatih model untuk memprediksi seluruh urutan output sekaligus.
*   **One-shot prediction per step:** Melatih model untuk memprediksi langkah waktu berikutnya, kemudian menggunakan prediksi tersebut sebagai input untuk memprediksi langkah waktu berikutnya lagi (iteratif).

## Menangani Urutan Panjang

Urutan yang sangat panjang dapat menyebabkan masalah dalam pelatihan RNN:
*   **Vanishing Gradients:** Gradien menjadi sangat kecil saat backpropagated melalui banyak langkah waktu, menyebabkan model sulit belajar ketergantungan jangka panjang.
*   **Exploding Gradients:** Gradien menjadi sangat besar, menyebabkan bobot model meledak dan pelatihan menjadi tidak stabil.

## Melawan Masalah Gradien Tidak Stabil

### Glorot dan He Initialization

Inisialisasi bobot yang tepat dapat membantu mengurangi masalah vanishing/exploding gradients di awal pelatihan.

### Fungsi Aktivasi Non-saturating

Fungsi aktivasi seperti ReLU dan variannya (Leaky ReLU, ELU, SELU) membantu mencegah vanishing gradients karena memiliki gradien non-nol di sebagian besar rentangnya.

### Normalisasi Batch

Normalisasi Batch dapat membantu menstabilkan pelatihan dengan menormalkan input ke setiap lapisan, mengurangi masalah vanishing/exploding gradients.

### Gradient Clipping

Gradient clipping adalah teknik untuk membatasi nilai gradien agar tidak terlalu besar, sehingga mencegah exploding gradients.

### Menggunakan Kembali Lapisan yang Sudah Dilatih (Transfer Learning)

Menggunakan model yang sudah dilatih pada tugas serupa dan kemudian menyesuaikannya (fine-tuning) untuk tugas baru.

### Pelatihan Tanpa Pengawasan (Unsupervised Pretraining)

Melatih bagian dari jaringan secara unsupervised (misal, dengan autoencoder) sebelum fine-tuning dengan supervised learning.

### Pelatihan pada Tugas Tambahan (Auxiliary Task)

Melatih jaringan pada tugas terkait yang lebih mudah sebelum tugas utama.

### Pengoptimal yang Lebih Cepat

Pengoptimal adaptif seperti Momentum, Nesterov Accelerated Gradient (NAG), AdaGrad, RMSProp, Adam, dan Nadam dapat mempercepat pelatihan dan membantu mengatasi gradien yang tidak stabil.

*   **Momentum Optimization:**
    $$v_t = \beta v_{t-1} + (1 - \beta) \nabla_\theta J(\theta)$$
    $$\theta = \theta - \eta v_t$$
    Dimana:
    *   $v_t$: vektor momentum
    *   $\beta$: hyperparameter momentum (biasanya 0.9)
    *   $\nabla_\theta J(\theta)$: gradien fungsi biaya
    *   $\eta$: learning rate

*   **Nesterov Accelerated Gradient (NAG):** Mirip dengan Momentum tetapi gradien diukur sedikit di depan dalam arah momentum.
    $$v_t = \beta v_{t-1} + \eta \nabla_\theta J(\theta - \beta v_{t-1})$$
    $$\theta = \theta - v_t$$

*   **AdaGrad:** Menyesuaikan learning rate untuk setiap parameter, menurun secara signifikan untuk parameter dengan gradien sering atau besar.
    $$s_t = s_{t-1} + \nabla_\theta J(\theta)^2$$
    $$\theta = \theta - \frac{\eta}{\sqrt{s_t + \epsilon}} \nabla_\theta J(\theta)$$
    Dimana $s_t$ mengakumulasi kuadrat gradien, $\epsilon$ adalah konstanta kecil untuk stabilitas numerik.

*   **RMSProp:** Mengatasi masalah AdaGrad yang terlalu cepat menurunkan learning rate.
    $$s_t = \rho s_{t-1} + (1 - \rho) \nabla_\theta J(\theta)^2$$
    $$\theta = \theta - \frac{\eta}{\sqrt{s_t + \epsilon}} \nabla_\theta J(\theta)$$
    Dimana $\rho$ adalah decay rate (biasanya 0.9).

*   **Adam (Adaptive Moment Estimation) dan Nadam (Nesterov Adam):** Menggabungkan ide momentum dan RMSProp.
    $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla_\theta J(\theta)$$
    $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) \nabla_\theta J(\theta)^2$$
    $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
    $$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
    $$\theta = \theta - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
    Nadam menambahkan komponen Nesterov ke Adam.

### Penjadwalan Learning Rate

Mengubah learning rate selama pelatihan. Beberapa strategi:
*   **Power scheduling:** $\eta(t) = \eta_0 / (1 + t/c)^p$
*   **Exponential scheduling:** $\eta(t) = \eta_0 e^{-t/c}$
*   **Piecewise constant scheduling:** Menggunakan learning rate konstan untuk beberapa epoch, lalu menurunkannya.
*   **Performance scheduling:** Mengurangi learning rate ketika performa model berhenti meningkat.
*   **1cycle scheduling:** Meningkatkan learning rate hingga batas maksimum dan kemudian menurunkannya lagi dalam satu siklus pelatihan.

## Mengatasi Masalah Memori Jangka Pendek (Short-Term Memory Problem)

RNN sederhana memiliki masalah memori jangka pendek, di mana mereka cenderung melupakan input yang jauh di masa lalu.
*   **Solusi:** Sel memori yang lebih kompleks seperti LSTM (Long Short-Term Memory) dan GRU (Gated Recurrent Unit).

### LSTM

LSTM mengatasi masalah memori jangka pendek dengan menambahkan "gerbang" (gates) yang mengontrol aliran informasi.
*   **Forget gate ($f_t$):** Mengontrol informasi mana dari cell state sebelumnya ($C_{t-1}$) yang harus dilupakan.
    $$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$$
*   **Input gate ($i_t$):** Mengontrol informasi baru mana yang harus disimpan dalam cell state.