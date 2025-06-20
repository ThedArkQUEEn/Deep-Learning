# Penjelasan Chapter 12 - Custom Models and Training with TensorFlow

Bab ini membahas bagaimana membangun model kustom dan algoritma pelatihan menggunakan TensorFlow, memberikan fleksibilitas lebih besar daripada API Keras Sequential atau Functional.

## Gambaran Singkat TensorFlow

TensorFlow dapat digunakan mirip dengan NumPy, tetapi dengan keuntungan dapat berjalan di GPU untuk komputasi yang lebih cepat dan memiliki dukungan untuk autogradient (autodiff).

### Tensor dan Operasi

TensorFlow merepresentasikan data sebagai objek `tf.Tensor`.
- **Konstanta:** `tf.constant()`
- **Operasi Aritmetika:** `tf.add()`, `tf.multiply()`, dll., atau operator Python standar (`+`, `*`).
- **Operasi Matematika Lanjut:** `tf.sin()`, `tf.exp()`, `tf.reduce_sum()`, dll.
- **Indexing:** Mirip dengan NumPy array.

### Tensor dan NumPy

TensorFlow `tf.Tensor` dapat dengan mudah dikonversi ke dan dari NumPy array:
- TensorFlow ke NumPy: `tensor.numpy()`
- NumPy ke TensorFlow: `tf.constant(numpy_array)`

### Konversi Tipe

TensorFlow sangat ketat tentang tipe data. Operasi biner (misalnya, penjumlahan) memerlukan operand dengan tipe data yang sama.
- Konversi tipe: `tf.cast(tensor, dtype=tf.float32)`

### Variabel

Untuk variabel yang dapat diubah dan digunakan dalam pelatihan model, gunakan `tf.Variable()`.
- **Inisialisasi:** `v = tf.Variable([3.])`
- **Penugasan (Assignment):** `v.assign([5.])` atau `v.assign_add([1.])`

### Struktur Data Lainnya

TensorFlow juga mendukung struktur data lain seperti `tf.RaggedTensor` untuk data tidak beraturan dan `tf.string` untuk string.

## Menyesuaikan Model dan Algoritma Pelatihan

Bagian ini menunjukkan bagaimana mengimplementasikan komponen-komponen kustom seperti fungsi *loss*, *layer*, dan *model*.

### Fungsi Loss Kustom

Anda dapat membuat fungsi *loss* kustom. Fungsi ini akan menerima *true targets* dan *predictions* sebagai input.

```python
def huber_loss(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss = tf.abs(error) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss)
```
### Custom Training Loops

Untuk kendali penuh atas proses pelatihan, Anda dapat mengimplementasikan *custom training loops*. Ini melibatkan penggunaan `tf.GradientTape` untuk menghitung gradien dan *optimizer* untuk menerapkan pembaruan bobot.

```python
# Contoh sederhana custom training loop
# Misalkan ada model, loss function, dan optimizer yang sudah didefinisikan

# model = MyCustomModel()
# loss_fn = tf.keras.losses.MeanSquaredError()
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# @tf.function
# def train_step(X_batch, y_batch):
#     with tf.GradientTape() as tape:
#         y_pred = model(X_batch)
#         loss = loss_fn(y_batch, y_pred)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     return loss

# for epoch in range(num_epochs):
#     for X_batch, y_batch in dataset:
#         loss_value = train_step(X_batch, y_batch)
#     print(f"Epoch {epoch}, Loss: {loss_value.numpy()}")
```

### TensorFlow Functions dan Graphs

TensorFlow 2.x menggabungkan *eager execution* dengan kemampuan untuk membangun *graphs* kinerja tinggi menggunakan `tf.function`.

- **`tf.function`**: Dekorator ini mengompilasi fungsi Python menjadi *graph* TensorFlow yang dapat dieksekusi dengan cepat.

```python
@tf.function
def cube(x):
    return x**3

# Saat pertama kali dipanggil, fungsi akan di-"traced" (dikompilasi menjadi graph)
# Panggilan berikutnya akan mengeksekusi graph yang telah dikompilasi
result = cube(tf.constant(2))
print(result) # tf.Tensor(8, shape=(), dtype=int32)
```
### AutoGraph dan Tracing

- **AutoGraph**: Fitur di TensorFlow yang secara otomatis mengubah kode Python menjadi *graph* TensorFlow. Ini terjadi secara implisit ketika Anda menggunakan `tf.function`.
- **Tracing**: Proses di mana `tf.function` menganalisis kode Python Anda dan membangun *graph* TensorFlow. Ini terjadi saat fungsi pertama kali dipanggil dengan *signature input* tertentu.

### Aturan `tf.function`

Ada beberapa aturan yang perlu diikuti saat menggunakan `tf.function` untuk memastikan *tracing* dan eksekusi yang benar:

- **Operasi TensorFlow**: Sebagian besar operasi TensorFlow akan diubah menjadi operasi *graph*.
- **Kontrol Alur Python (Python control flow)**: `if`, `for`, `while` akan diubah menjadi operasi *graph* yang setara (`tf.cond`, `tf.while`).
- **Efek samping (side effects)**: Hati-hati dengan efek samping di dalam `tf.function`, karena mungkin hanya terjadi sekali saat *tracing*. Untuk efek samping yang ingin selalu terjadi, gunakan `tf.print()` atau `tf.summary.*`.
- **Variabel Python (Python variables)**: Variabel Python primitif akan di-*captured* berdasarkan nilainya saat *tracing*. Untuk variabel yang dapat diubah, gunakan `tf.Variable`.