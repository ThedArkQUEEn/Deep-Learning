# Encoder-Decoder LSTM dengan Bahdanau Attention

## Deskripsi
Model ini menggunakan arsitektur **Sequence-to-Sequence (Seq2Seq)** yang terdiri dari **Encoder, Decoder**, dan **Mekanisme Attention**. Arsitektur ini dirancang untuk tugas-tugas seperti **terjemahan mesin**, di mana urutan input (misalnya, kalimat dalam bahasa Jerman) dikonversi menjadi urutan output (misalnya, kalimat dalam bahasa Inggris).

## Komponen Model
### Encoder (TensorFlow/PyTorch)
- **Tujuan**: Mengubah urutan input menjadi representasi konteks tetap (vektor) yang menangkap makna kalimat input.
- **Input**: Urutan token dari bahasa sumber.
- **Lapisan Embedding**: Mengubah setiap token input menjadi vektor berdimensi padat.
  - TensorFlow: `tf.keras.layers.Embedding(vocab_size, embedding_units)`
  - PyTorch: `torch.nn.Embedding(vocab_size, embedding_dim)`
- **Bidirectional LSTM**: Memproses urutan input dari dua arah (maju dan mundur).
  - TensorFlow: `tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=True, return_state=True))`
  - PyTorch: `torch.nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)`
- **Output Encoder**:
  - `outputs`: Hidden states dari setiap langkah waktu.
  - `state_h, state_c`: Final hidden state dan cell state.

### Bahdanau Attention (TensorFlow/PyTorch)
- **Tujuan**: Memungkinkan decoder untuk fokus pada bagian-bagian relevan dari input encoder saat menghasilkan setiap token output.
- **Mekanisme**: Menghitung skor keselarasan antara hidden state decoder saat ini (`query`) dan hidden states encoder (`values`), lalu dihitung bobot perhatian (`attention weights`) menggunakan softmax.
- **Rumus Konseptual**:
  - Skor keselarasan:
    
    $score(si, hj) = va^T tanh(W1 * si + W2 * hj)$
    
  - Bobot perhatian:
    
    $αij = exp(score(si, hj)) / Σk exp(score(si, hk))$
    
  - Vektor konteks:
    
    $ci = Σj αij hj$
    
- **Implementasi**:
  - TensorFlow:
    ```python
    self.W1 = layers.Dense(units)
    self.W2 = layers.Dense(units)
    self.V = layers.Dense(1)
    score = self.V(tf.nn.tanh(self.W1(query) + self.W2(values)))
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = tf.reduce_sum(attention_weights * values, axis=1)
    ```
  - PyTorch:
    Menggunakan `nn.Linear` dan operasi softmax.

### Decoder (TensorFlow/PyTorch)
- **Tujuan**: Menghasilkan urutan output berdasarkan context vector dan hidden state sebelumnya.
- **Input**: Token target sebelumnya, encoder outputs (untuk perhatian), dan hidden state decoder sebelumnya.
- **Lapisan Embedding**: Sama seperti encoder, untuk token target.
- **Lapisan LSTM**: Memproses input dan menghasilkan hidden state baru.
- **Lapisan Dense (Fully Connected)**: Mengubah hidden state decoder menjadi distribusi probabilitas di atas kosakata target.
- **Output Decoder**:
  - Logits untuk token yang diprediksi pada langkah waktu saat ini.
  - Hidden state baru dan cell state baru.
  - Bobot perhatian.
- **Implementasi**:
  - TensorFlow:
    ```python
    self.lstm = layers.LSTM(lstm_units * 2, return_sequences=True, return_state=True)
    self.fc = layers.Dense(vocab_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    ```
  - PyTorch:
    Menggunakan `nn.LSTM` dan `nn.Linear`.

## Arsitektur Model (Generalisasi)
- **Input**: Source Sequence (X = {x₁, x₂, ..., xTₓ})
- **Encoder**:
  
  $hj = EncoderLSTM(xj, hj−1)$

- **Output Encoder**:

    $h = {h₁, h₂, ..., hTₓ}, s₀ = {hTₓ, cTₓ}$

- Decoder (Loop waktu):
    - Untuk setiap langkah waktu i:
        - Input decoder: yi−1 (token target sebelumnya)
        - Attention:

        $ci = Attention(si−1, {h₁, h₂, ..., hTₓ})$

        - Decoder LSTM:

        $(si, celli) = DecoderLSTM([Embedding(yi−1), ci], si−1, celli−1)$

        - Prediksi:

        $y = {y₂, ..., yTᵧ}$

## Proses Pelatihan dan Inferensi
- Pelatihan:
    - Menggunakan teacher forcing, di mana input decoder pada langkah waktu t adalah token target yang benar dari langkah sebelumnya.
- Inferensi (Penerjemahan):
    - Saat menerjemahkan kalimat baru, decoder diberikan token <start> dan memprediksi token berikutnya berdasarkan hidden state dan context vector, hingga menemukan token <end> atau mencapai panjang maksimum.
## Kesimpulan
Arsitektur Seq2Seq dengan Bahdanau Attention meningkatkan akurasi penerjemahan dengan memungkinkan model untuk lebih fokus pada bagian relevan dari input. Pendekatan ini sangat berguna dalam tugas NLP seperti terjemahan mesin, generasi teks, dan dialog AI.







