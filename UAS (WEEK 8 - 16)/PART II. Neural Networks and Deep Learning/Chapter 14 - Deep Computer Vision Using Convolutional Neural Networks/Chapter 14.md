# Penjelasan Chapter 14 - Deep Computer Vision Using Convolutional Neural Networks

## Arsitektur Korteks Visual
Membahas tentang bagaimana korteks visual mamalia bekerja dan bagaimana arsitektur CNN terinspirasi dari struktur ini.

## Lapisan Konvolusional
Lapisan ini adalah inti dari CNN. Setiap neuron dalam lapisan konvolusional hanya terhubung ke sebagian kecil neuron dari lapisan sebelumnya, disebut *receptive field*.

### Filter
Filter (atau kernel) adalah matriks kecil yang digunakan untuk mendeteksi fitur tertentu dalam input, seperti tepi, garis, atau pola. Sebuah lapisan konvolusional dapat memiliki banyak filter, masing-masing mendeteksi pola yang berbeda.
* **Padding**: Menambahkan nol di sekitar input untuk mengontrol ukuran output.
    * "valid" (tanpa padding): output akan lebih kecil dari input.
    * "same" (padding sama): output akan memiliki ukuran yang sama dengan input (jika `strides=1`).
* **Strides**: Seberapa besar "langkah" filter melintasi input. `strides=1` berarti filter bergerak satu piksel pada satu waktu.
    * Output size formula (valid padding):
        $$(N - F + 1) / S$$
        dimana $N$ = input size, $F$ = filter size, $S$ = stride.
    * Output size formula (same padding):
        $$N / S$$
        (dibulatkan ke atas jika perlu)

### Menumpuk Beberapa Peta Fitur
Ketika filter diterapkan, mereka menghasilkan "peta fitur" (feature map). Lapisan konvolusional biasanya memiliki banyak filter, menghasilkan banyak peta fitur yang ditumpuk.

### Persyaratan Memori
Membahas tentang persyaratan memori yang tinggi dari lapisan konvolusional, terutama untuk lapisan awal.

## Lapisan Pooling
Lapisan pooling bertujuan untuk mengurangi ukuran data masukan secara spasial, yang mengurangi jumlah parameter dan komputasi, serta membuat model lebih toleran terhadap sedikit pergeseran dalam gambar.

### Max Pooling
Lapisan Max Pooling mengambil nilai maksimum dari setiap *patch* masukan.

### Rata-rata Pooling
Lapisan Average Pooling mengambil nilai rata-rata dari setiap *patch* masukan.

## Arsitektur CNN
Membahas beberapa arsitektur CNN terkenal yang telah mencapai hasil luar biasa dalam tugas penglihatan komputer.

### LeNet-5
Salah satu CNN paling awal dan berpengaruh, dikembangkan oleh Yann LeCun untuk pengenalan digit tulisan tangan.
* Arsitektur: CONV -> POOL -> CONV -> POOL -> FC -> FC -> Output.

### AlexNet
Arsitektur ini memenangkan kompetisi ImageNet pada tahun 2012, secara signifikan lebih dalam dan lebar dari LeNet-5.
* Fitur utama: Banyak lapisan konvolusional, Max Pooling, Dropout, ReLU sebagai fungsi aktivasi.

### GoogLeNet
Memenangkan ImageNet pada tahun 2014. Memperkenalkan modul "Inception" yang memungkinkan jaringan mempelajari beberapa skala fitur secara paralel.
* Fitur utama: Modul Inception, Global Average Pooling.

### VGGNet
Arsitektur yang sangat sederhana, hanya menggunakan lapisan konvolusional 3x3 dan Max Pooling.
* Fitur utama: Kedalaman yang sangat tinggi (hingga 19 lapisan), keseragaman arsitektur.

### ResNet (Residual Network)
Memenangkan ImageNet pada tahun 2015. Memperkenalkan "residual connections" atau "skip connections" untuk mengatasi masalah vanishing gradient pada jaringan yang sangat dalam.
* Rumus blok residual:
    $$y = F(x) + x$$
    dimana $x$ adalah input, $F(x)$ adalah transformasi lapisan, dan $y$ adalah output.

### Xception
Arsitektur yang dibangun di atas ide Inception, tetapi menggunakan "depthwise separable convolutions".
* Fitur utama: Depthwise separable convolutions, yang memisahkan operasi konvolusi spasial dari konvolusi channel.

### SENet (Squeeze-and-Excitation Network)
Memenangkan ImageNet pada tahun 2017. Fokus pada peningkatan representasi fitur dengan mengadaptasi kalibrasi fitur.
* Fitur utama: Modul Squeeze-and-Excitation yang secara adaptif mengkalibrasi respons channel.

### Menggunakan Model Pra-terlatih dari Keras
Keras menyediakan banyak model pra-terlatih (misalnya, ResNet, VGGNet, MobileNet) yang dapat langsung digunakan atau disetel ulang (fine-tuned) untuk tugas baru.

### Model Pra-terlatih untuk Transfer Learning
Menggunakan model yang telah dilatih pada dataset besar (misalnya, ImageNet) sebagai titik awal untuk tugas baru yang terkait. Ini menghemat waktu dan sumber daya komputasi.

## Klasifikasi dan Lokalisasi
Selain mengklasifikasikan gambar, beberapa model dapat melokalisasi objek dalam gambar dengan memprediksi *bounding box*.

### Deteksi Objek
Tugas untuk mengidentifikasi dan melokalisasi beberapa objek dalam sebuah gambar.

### Fully Convolutional Networks (FCNs)
Jaringan yang hanya terdiri dari lapisan konvolusional, memungkinkan input berukuran arbritrary dan output peta spasial. Digunakan dalam segmentasi semantik.

### You Only Look Once (YOLO)
Algoritma deteksi objek yang sangat cepat dan akurat. Mendeteksi objek secara langsung dari gambar lengkap dalam satu kali melewati jaringan.
* Membagi gambar menjadi grid, setiap sel grid bertanggung jawab memprediksi *bounding box* dan probabilitas kelas.

### Segmentasi Semantik
Tugas untuk mengklasifikasikan setiap piksel dalam gambar ke dalam kategori objek.
* Outputnya adalah peta segmentasi di mana setiap piksel diberi label kelas.
* Sering menggunakan FCNs atau arsitektur seperti U-Net.