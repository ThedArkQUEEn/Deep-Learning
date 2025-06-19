# Penjelasan Chapter 11 - Training Deep Neural Networks

---

-   **Masalah Vanishing/Exploding Gradients**
    -   **Deskripsi:** Gradien seringkali menjadi semakin kecil (vanishing gradients) atau semakin besar (exploding gradients) saat mengalir mundur melalui lapisan-lapisan jaringan yang dalam, membuat pelatihan sulit atau tidak stabil.
    -   **Penyebab:** Terutama disebabkan oleh fungsi aktivasi sigmoid (logistic) dan hyperbolic tangent (tanh) yang saturasi, serta inisialisasi bobot yang tidak tepat.

-   **Glorot dan He Initialization**
    -   **Deskripsi:** Teknik inisialisasi bobot yang dirancang untuk menjaga varian gradien tetap stabil di semua lapisan.
    -   **Rumus (Glorot Initialization):**
        -   Untuk fungsi aktivasi sigmoid atau tanh:
            -   Inisialisasi bobot dengan distribusi normal dengan mean 0 dan varian $\sigma^2 = \frac{2}{n_{inputs} + n_{outputs}}$
            -   Atau distribusi seragam antara $-r$ dan $r$, dengan $r = \sqrt{\frac{6}{n_{inputs} + n_{outputs}}}$
        -   Untuk fungsi aktivasi ReLU atau variannya (He Initialization):
            -   Inisialisasi bobot dengan distribusi normal dengan mean 0 dan varian$ \sigma^2 = \frac{2}{n_{inputs}}$
            -   Atau distribusi seragam antara $-r$ dan $r$, dengan $r = \sqrt{\frac{6}{n_{inputs}}}$
        -   $n_{inputs}$: jumlah input dari lapisan
        -   $n_{outputs}$: jumlah output dari lapisan

-   **Fungsi Aktivasi Nonsaturasi**
    -   **Deskripsi:** Penggunaan fungsi aktivasi seperti ReLU (Rectified Linear Unit), Leaky ReLU, ELU (Exponential Linear Unit), dan SELU untuk mengatasi masalah vanishing gradients.
    -   **Rumus:**
        -   **ReLU:** $ReLU(z) = \max(0, z)$
        -   **Leaky ReLU:** $LeakyReLU(z) = \max(\alpha z, z)$ (dengan $\alpha$ adalah hyperparameter kecil, misal 0.01)
        -   **ELU:** $ELU(z) = \begin{cases} z & \text{if } z \ge 0 \\ \alpha(e^z - 1) & \text{if } z < 0 \end{cases}$ (dengan $\alpha$ adalah hyperparameter, umumnya 1)
        -   **SELU:** $SELU(z) = \lambda \begin{cases} z & \text{if } z > 0 \\ \alpha(e^z - 1) & \text{if } z \le 0 \end{cases}$ (dengan $\lambda \approx 1.0507$ dan $\alpha \approx 1.6732$)

-   **Normalisasi Batch (Batch Normalization)**
    -   **Deskripsi:** Teknik untuk menormalkan input dari setiap lapisan dalam mini-batch. Ini membantu mengatasi masalah perubahan distribusi input antar lapisan (internal covariate shift) dan memungkinkan penggunaan learning rate yang lebih tinggi.
    -   **Rumus:**
        -   Normalisasi input: $\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$
        -   Penskalaan dan pergeseran: $z_i = \gamma \hat{x}_i + \beta$
        -   $\mu_B$: mean mini-batch
        -   $\sigma_B^2$: varian mini-batch
        -   $\epsilon$: konstanta kecil untuk stabilitas numerik
        -   $\gamma, \beta$: parameter yang dipelajari (scale dan offset)

-   **Gradient Clipping**
    -   **Deskripsi:** Membatasi ukuran gradien selama backpropagation untuk mencegah exploding gradients.
    -   **Rumus:** Jika norm gradien melebihi ambang batas, gradien diskalakan ke bawah agar normenya sama dengan ambang batas. Misalnya, jika $|g|_2 > threshold$, maka $g_{clipped} = g \frac{threshold}{|g|_2}$.

-   **Menggunakan Ulang Lapisan yang Sudah Dilatih (Reusing Pretrained Layers)**
    -   **Deskripsi:** Teknik transfer learning, di mana lapisan-lapisan dari model yang sudah dilatih sebelumnya pada tugas serupa digunakan sebagai bagian dari model baru. Ini sangat berguna ketika data pelatihan baru terbatas.

-   **Optimizers yang Lebih Cepat (Faster Optimizers)**
    -   **Deskripsi:** Algoritma optimasi yang mempercepat proses konvergensi dibandingkan dengan Gradient Descent dasar.
    -   **Momentum Optimization**
        -   **Deskripsi:** Mempercepat SGD dengan menambahkan proporsi gradien dari langkah sebelumnya ke gradien saat ini, membantu melewati rintangan lokal.
        -   **Rumus:**
            -   $m \leftarrow \beta m + \eta \nabla_\theta J(\theta)$
            -   $\theta \leftarrow \theta - m$
            -   $m$: vektor momentum
            -   $\beta$: hyperparameter momentum (umumnya 0.9)
            -   $\eta$: learning rate
            -   $\nabla_\theta J(\theta)$: gradien fungsi biaya terhadap parameter $\theta$
    -   **Nesterov Accelerated Gradient (NAG)**
        -   **Deskripsi:** Varian momentum yang mengukur gradien tidak pada posisi saat ini, tetapi sedikit di depan pada arah momentum.
        -   **Rumus:**
            -   $m \leftarrow \beta m + \eta \nabla_\theta J(\theta - \beta m)$
            -   $\theta \leftarrow \theta - m$
    -   **AdaGrad**
        -   **Deskripsi:** Menyesuaikan learning rate untuk setiap parameter secara adaptif, semakin kecil untuk parameter yang sering diperbarui dan semakin besar untuk parameter yang jarang diperbarui.
        -   **Rumus:**
            -   $s_i \leftarrow s_i + (\nabla_{\theta_i} J(\theta))^2$
            -   $\theta_i \leftarrow \theta_i - \frac{\eta}{\sqrt{s_i + \epsilon}} \nabla_{\theta_i} J(\theta)$
            -   $s_i$: vektor kuadrat gradien kumulatif untuk parameter $i$
    -   **RMSProp**
        -   **Deskripsi:** Mirip dengan AdaGrad tetapi mengatasi kelemahan AdaGrad yang learning rate-nya bisa menjadi sangat kecil. RMSProp hanya mengakumulasi gradien dari iterasi terakhir.
        -   **Rumus:**
            -   $s \leftarrow \beta s + (1 - \beta) (\nabla_\theta J(\theta))^2$
            -   $\theta \leftarrow \theta - \frac{\eta}{\sqrt{s + \epsilon}} \nabla_\theta J(\theta)$
            -   $\beta$: hyperparameter (umumnya 0.9)
    -   **Adam dan Nadam Optimization**
        -   **Deskripsi:** Adaptive Moment Estimation (Adam) menggabungkan ide Momentum Optimization dan RMSProp. Nadam (Nesterov Adam) adalah versi Adam yang menggabungkan NAG.
        -   **Rumus (Adam):**
            -   $m \leftarrow \beta_1 m + (1 - \beta_1) \nabla_\theta J(\theta)$ (momentum orde pertama)
            -   $v \leftarrow \beta_2 v + (1 - \beta_2) (\nabla_\theta J(\theta))^2$ (momentum orde kedua)
            -   $\hat{m} \leftarrow \frac{m}{1 - \beta_1^t}$ (koreksi bias untuk $m$)
            -   $\hat{v} \leftarrow \frac{v}{1 - \beta_2^t}$ (koreksi bias untuk $v$)
            -   $\theta \leftarrow \theta - \frac{\eta}{\sqrt{\hat{v} + \epsilon}} \hat{m}$
            -   $\beta_1, \beta_2$: hyperparameter (umumnya 0.9 dan 0.999)
            -   $t$: timestep
            -   $\epsilon$: konstanta kecil untuk stabilitas numerik

-   **Learning Rate Scheduling**
    -   **Deskripsi:** Strategi untuk mengubah learning rate selama pelatihan untuk mencapai konvergensi yang lebih baik.
    -   **Jenis-jenis umum:**
        -   **Power scheduling:** Learning rate menurun secara bertahap seiring waktu. $\eta(t) = \eta_0 / (1 + t/c)^k$
        -   **Exponential scheduling:** Learning rate menurun secara eksponensial. $\eta(t) = \eta_0 0.1^{t/c}$
        -   **Piecewise constant scheduling:** Learning rate tetap konstan untuk beberapa epoch, lalu turun.
        -   **Performance scheduling:** Learning rate menurun ketika kinerja tidak lagi membaik.
        -   **One-cycle scheduling:** Learning rate meningkat dari nilai awal hingga maksimum, lalu turun ke nilai minimum.

-   **Menghindari Overfitting Melalui Regularisasi**
    -   **Deskripsi:** Teknik untuk mencegah model menjadi terlalu kompleks dan cocok dengan data pelatihan (overfitting), sehingga tidak dapat digeneralisasi dengan baik ke data baru.
    -   **L1 dan L2 Regularization**
        -   **Deskripsi:** Menambahkan penalti ke fungsi biaya berdasarkan ukuran bobot model.
        -   **Rumus (L1 Regularization - Lasso Regression):** $Cost Function + \alpha \sum_{i=1}^{n} |\theta_i|$
        -   **Rumus (L2 Regularization - Ridge Regression):** $Cost Function + \alpha \sum_{i=1}^{n} \theta_i^2$
        -   $\alpha$: hyperparameter regularisasi
    -   **Dropout**
        -   **Deskripsi:** Selama pelatihan, secara acak menonaktifkan (drop out) beberapa neuron pada setiap langkah pelatihan. Neuron yang "dibuang" tidak berkontribusi pada aktivasi feedforward atau backpropagation.
        -   **Efek:** Mencegah neuron menjadi terlalu bergantung pada neuron tetangga, memaksa mereka untuk lebih tangguh dan beradaptasi.
    -   **Monte Carlo (MC) Dropout**
        -   **Deskripsi:** Mengaktifkan dropout selama inferensi (bukan hanya pelatihan) untuk mendapatkan perkiraan ketidakpastian prediksi.
        -   **Kegunaan:** Berguna untuk memperkirakan ketidakpastian model, misalnya dalam sistem keamanan atau medis.
    -   **Max-Norm Regularization**
        -   **Deskripsi:** Membatasi norm vektor bobot input untuk setiap neuron.
        -   **Rumus:** Untuk setiap neuron $j$ di lapisan, $\|w_j\|_2 \le r$, di mana $r$ adalah hyperparameter max-norm. Jika norm bobot melebihi $r$, bobot disesuaikan ke bawah.