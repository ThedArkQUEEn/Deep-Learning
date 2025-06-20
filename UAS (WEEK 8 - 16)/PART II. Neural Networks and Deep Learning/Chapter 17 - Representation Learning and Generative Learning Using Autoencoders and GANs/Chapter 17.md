# Penjelasan Chapter 17 - Representation Learning and Generative Learning Using Autoencoders and GANs

## Efficient Data Representations

*   **Tujuan:** Belajar representasi data yang lebih efisien dan bermakna.
*   **Manfaat:** Mengurangi dimensi data, menghilangkan noise, mengisi nilai yang hilang, dan membantu dalam tugas-tugas downstream.

## Performing PCA with an Undercomplete Linear Autoencoder

*   **Autoencoder:** Jaringan saraf yang dilatih untuk merekonstruksi inputnya.
*   **Undercomplete:** Lapisan tersembunyi (encoding layer) memiliki dimensi yang lebih rendah daripada input.
*   **Linear:** Menggunakan fungsi aktivasi linear di semua neuron.
*   **Fungsi Cost:** Mean Squared Error (MSE) antara input dan output rekonstruksi.
*   **Rumus MSE:** $MSE = \frac{1}{m} \sum_{i=1}^{m} (x^{(i)} - \hat{x}^{(i)})^2$
    *   $x^{(i)}$: input
    *   $\hat{x}^{(i)}$: rekonstruksi output
*   **Hasil:** Autoencoder linear yang undercomplete akan belajar untuk melakukan PCA (Principal Component Analysis).

## Stacked Autoencoders

*   **Definisi:** Autoencoder yang terdiri dari beberapa autoencoder sederhana yang ditumpuk.
*   **Arsitektur:**
    *   Encoder: Memiliki beberapa lapisan tersembunyi yang mengurangi dimensi data secara bertahap.
    *   Decoder: Memiliki beberapa lapisan tersembunyi yang meningkatkan dimensi data secara bertahap untuk merekonstruksi input.
*   **Pelatihan:** Dapat dilatih secara greedily layer per layer secara unsupervised, kemudian fine-tuned menggunakan supervised learning.

## Visualizing the Reconstructions

*   **Tujuan:** Memeriksa seberapa baik autoencoder merekonstruksi input.
*   **Indikasi:** Rekonstruksi yang mirip dengan input asli menunjukkan representasi yang baik.

## Visualizing the Fashion MNIST Dataset

*   **Dataset:** Dataset gambar grayscale dari item pakaian.
*   **Penggunaan Autoencoder:** Untuk mereduksi dimensi gambar dan memvisualisasikan data dalam ruang dimensi rendah.

## Unsupervised Pretraining Using Stacked Autoencoders

*   **Manfaat:** Membantu melatih jaringan saraf yang sangat dalam ketika data berlabel terbatas.
*   **Proses:**
    1.  Latih autoencoder pertama secara unsupervised pada data asli.
    2.  Gunakan output lapisan tersembunyi autoencoder pertama sebagai input untuk melatih autoencoder kedua.
    3.  Ulangi hingga semua lapisan encoder dilatih.
    4.  Gabungkan semua lapisan encoder yang telah dilatih dengan lapisan output klasifikasi/regresi, lalu fine-tune seluruh jaringan menggunakan data berlabel.

## Tying Weights

*   **Definisi:** Memaksa bobot lapisan decoder menjadi transpose dari bobot lapisan encoder.
*   **Manfaat:** Mengurangi jumlah parameter, mempercepat pelatihan, dan berpotensi mengurangi overfitting.
*   **Rumus:** $W_{decoder} = W_{encoder}^T$

## Training One Autoencoder at a Time

*   **Pendekatan:** Melatih setiap autoencoder dalam tumpukan secara terpisah dan berurutan.

## Convolutional Autoencoders

*   **Definisi:** Autoencoder yang menggunakan lapisan konvolusional dan pooling (encoder) serta lapisan dekonvolusional (decoder).
*   **Penggunaan:** Cocok untuk data gambar karena memanfaatkan struktur spasial data.

## Recurrent Autoencoders

*   **Definisi:** Autoencoder yang menggunakan unit rekuren (misalnya, LSTM atau GRU) di encoder dan decoder.
*   **Penggunaan:** Cocok untuk data sekuensial (misalnya, teks, deret waktu).

## Denoising Autoencoders

*   **Tujuan:** Merekonstruksi input asli dari input yang telah rusak (ber-noise).
*   **Proses Pelatihan:**
    1.  Berikan input yang di-noise ke autoencoder.
    2.  Autoencoder dilatih untuk merekonstruksi input asli yang bersih.
*   **Fungsi Cost:** MSE antara input asli dan output rekonstruksi.
*   **Manfaat:** Memaksa autoencoder untuk mempelajari fitur-fitur yang lebih kuat dan tidak rentan terhadap noise.

## Sparse Autoencoders

*   **Tujuan:** Memaksa autoencoder untuk mengaktifkan hanya sebagian kecil dari neuron tersembunyi.
*   **Mekanisme:** Menambahkan istilah "sparsity penalty" ke fungsi cost.
*   **Sparsity Penalty:**
    *   **Kullback-Leibler (KL) Divergence:** Mengukur seberapa banyak distribusi probabilitas neuron tersembunyi menyimpang dari distribusi yang diinginkan (misalnya, distribusi Bernoulli dengan probabilitas kecil).
    *   **Rumus KL Divergence:** $KL(P || Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}$
        *   $P$: distribusi aktual (aktivasi neuron tersembunyi)
        *   $Q$: distribusi target (misalnya, probabilitas aktivasi yang diinginkan)
*   **Manfaat:** Memaksa autoencoder untuk mempelajari representasi yang lebih disebarluaskan dan disentangled.

## Variational Autoencoders

*   **Definisi:** Model generatif probabilistik yang belajar representasi data laten (latent space) dan dapat menghasilkan data baru yang realistis.
*   **Perbedaan dengan Autoencoder Klasik:**
    *   Variational Autoencoder (VAE) tidak hanya belajar encoder/decoder, tetapi juga belajar distribusi probabilitas untuk latent space.
    *   Encoder VAE menghasilkan mean ($\mu$) dan log variance ($\log \sigma^2$) dari distribusi Gaussian (biasanya) di latent space.
    *   Decoder VAE mengambil sampel dari distribusi ini untuk menghasilkan output.
*   **Fungsi Cost (Evidence Lower Bound - ELBO):** Kombinasi dari dua istilah:
    1.  **Rekonstruksi Loss:** Mengukur seberapa baik decoder merekonstruksi input. (Misalnya, MSE atau Binary Cross-Entropy).
    2.  **KL Divergence Regularization:** Mengukur seberapa dekat distribusi yang dipelajari oleh encoder dengan distribusi prior (misalnya, Gaussian standar). Ini mendorong latent space agar terstruktur dengan baik dan dapat diinterpolasi.
    *   **Rumus KL Divergence untuk VAE:** $KL(Q(z|x) || P(z)) = -0.5 \sum_{j=1}^{D} (1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2)$
        *   $Q(z|x)$: distribusi posterior yang dipelajari oleh encoder
        *   $P(z)$: distribusi prior (Gaussian standar)
*   **Proses Generasi:** Ambil sampel dari distribusi prior di latent space, kemudian gunakan decoder untuk menghasilkan data baru.

## Generating Fashion MNIST Images

*   **Penggunaan VAE:** Untuk menghasilkan gambar Fashion MNIST baru yang belum pernah dilihat sebelumnya, dengan menginterpolasi di latent space.

## Generative Adversarial Networks (GANs)

*   **Definisi:** Terdiri dari dua jaringan saraf yang saling bersaing:
    1.  **Generator:** Dilatih untuk menghasilkan data palsu yang realistis.
    2.  **Discriminator:** Dilatih untuk membedakan antara data asli dan data palsu yang dihasilkan oleh generator.
*   **Proses Pelatihan:**
    1.  **Latih Discriminator:** Dengan data asli (label 1) dan data palsu dari generator (label 0).
    2.  **Latih Generator:** Dengan menyesuaikan bobotnya sehingga discriminator salah mengklasifikasikan output generator sebagai asli (label 1).
*   **Fungsi Cost:**
    *   **Discriminator:** Binary Cross-Entropy. Tujuannya adalah memaksimalkan probabilitas penugasan label yang benar untuk data asli dan palsu.
    *   **Generator:** Binary Cross-Entropy. Tujuannya adalah meminimalkan log probabilitas discriminator mengklasifikasikan data palsu sebagai palsu.
*   **Permainan Minimax (Original GAN):**
    *   **Fungsi Tujuan:** $\min_G \max_D V(D,G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_{z}(z)}[\log (1 - D(G(z)))]$
        *   $D(x)$: Probabilitas discriminator bahwa $x$ adalah asli.
        *   $G(z)$: Output generator dari noise $z$.
*   **Equilibrium:** Idealnya, generator menghasilkan data yang begitu realistis sehingga discriminator tidak dapat membedakannya dari data asli (probabilitas 0.5).

## The Difficulties of Training GANs

*   **Mode Collapse:** Generator menghasilkan variasi data yang terbatas.
*   **Vanishing Gradients:** Diskriminator menjadi terlalu baik sehingga generator menerima gradien yang sangat kecil, menghambat pembelajarannya.
*   **Training Instability:** Pelatihan GAN sangat sensitif terhadap hyperparameter dan arsitektur.

## Deep Convolutional GANs (DCGANs)

*   **Definisi:** Arsitektur GAN yang menggunakan lapisan konvolusional di generator dan diskriminator.
*   **Karakteristik:**
    *   Mengganti lapisan pooling dengan *strided convolutions* (discriminator) dan *fractional-strided convolutions* (generator).
    *   Menghilangkan lapisan *fully connected* (kecuali untuk input generator dan output discriminator).
    *   Menggunakan *Batch Normalization* di sebagian besar lapisan.
    *   Menggunakan fungsi aktivasi ReLU di generator (kecuali lapisan output yang menggunakan Tanh).
    *   Menggunakan fungsi aktivasi Leaky ReLU di discriminator.
*   **Manfaat:** Menghasilkan gambar yang lebih stabil dan berkualitas tinggi.

## Progressive Growing of GANs

*   **Definisi:** Teknik pelatihan di mana generator dan diskriminator dimulai dengan resolusi gambar yang sangat rendah (misalnya, 4x4 piksel) dan secara bertahap menambahkan lapisan baru untuk meningkatkan resolusi (misalnya, 8x8, 16x16, dst.).
*   **Manfaat:**
    *   Meningkatkan stabilitas pelatihan.
    *   Memungkinkan pelatihan GAN untuk menghasilkan gambar resolusi sangat tinggi.
    *   Mempercepat pelatihan.

## StyleGANs

*   **Definisi:** Arsitektur GAN yang memperkenalkan *style-based generator* untuk mengontrol berbagai aspek gambar yang dihasilkan (misalnya, pose, identitas, warna).
*   **Karakteristik Utama:**
    *   Memisahkan pemetaan laten (latent mapping) dari noise yang dimasukkan.
    *   Menggunakan *adaptive instance normalization* (AdaIN) untuk menyuntikkan "gaya" ke setiap skala dalam generator.
    *   Memasukkan noise di setiap skala untuk detail stokastik.
*   **Manfaat:** Menghasilkan gambar yang sangat realistis dan memungkinkan kontrol yang lebih baik terhadap atribut gambar yang dihasilkan.
