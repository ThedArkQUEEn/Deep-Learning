# Penjelasan Chapter 09 - Unsupervised Learning Techniques 
---
## 1. Clustering (Pengelompokan)

### Definisi
Teknik untuk mendeteksi kelompok-kelompok instansi yang serupa tanpa label.

### Contoh Aplikasi
- Segmentasi pengunjung blog berdasarkan perilaku
- Segmentasi pelanggan berdasarkan riwayat pembelian

---

### 1.1. K-Means

#### Konsep
Membagi $n$ observasi menjadi $k$ kelompok berdasarkan jarak terdekat ke pusat kelompok (centroid).

#### Langkah-langkah
1. Inisialisasi $k$ centroid secara acak
2. Ulangi sampai konvergensi:
   - Tetapkan setiap instansi ke centroid terdekat
   - Perbarui centroid menjadi rata-rata instansi dalam klaster

#### Fungsi Biaya (Inertia)
$$J = \sum_{i=0}^{n} \min_{j \in \{0, \dots, k-1\}} \left\|x_i - \mu_j\right\|^2$$

#### Batasan
- Harus menentukan $k$ di awal
- Sensitif terhadap inisialisasi centroid
- Tidak optimal untuk cluster non-globular atau dengan outlier

#### Penggunaan
- Segmentasi citra
- Pra-pemrosesan untuk supervised learning
- Pembelajaran semi-supervised

---

### 1.2. DBSCAN

#### Konsep
Clustering berbasis kepadatan yang membentuk klaster dari area padat dan mengabaikan outlier.

#### Terminologi
- **Epsilon (ε)**: Radius maksimum untuk tetangga
- **MinPts**: Minimum titik untuk mendefinisikan area padat
- **Core instance**: Titik dengan ≥ MinPts dalam radius ε
- **Border instance**: Dalam radius ε dari core, tapi bukan core
- **Noise instance**: Tidak termasuk dalam cluster mana pun

#### Kelebihan
- Tidak perlu menentukan jumlah cluster
- Mendeteksi bentuk cluster arbitrer
- Tangguh terhadap outlier

#### Kekurangan
- Tidak cocok untuk data dengan kepadatan bervariasi
- Sensitif terhadap ε dan MinPts

---

### 1.3. Algoritma Clustering Lainnya

- **Agglomerative Clustering**: Hierarki bottom-up
- **Birch**: Efisien untuk dataset besar (menggunakan Clustering Feature Tree)
- **Mean-Shift**: Menggeser centroid menuju kepadatan maksimal
- **Spectral Clustering**: Menggunakan matriks keserupaan dan reduksi dimensi

---

## 2. Gaussian Mixtures (Campuran Gaussian)

### Konsep
Model probabilistik yang mengasumsikan tiap instansi dihasilkan dari campuran beberapa distribusi Gaussian.

### Rumus Probabilitas
$$p(x) = \sum_{k=1}^{K} \phi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$$

- $\phi_k$: Bobot campuran
- $\mu_k$, $\Sigma_k$: Mean dan kovarians dari komponen $k$

### Algoritma: Expectation-Maximization (EM)

---

### Penggunaan GMM

- **Anomaly Detection**: Titik dengan densitas rendah = anomali
- **Novelty Detection**: Sama, tapi model dilatih hanya dengan data "bersih"

### Evaluasi: Menentukan Jumlah Klaster
- **BIC**: $BIC = -2 \log(\hat{L}) + p \log(n)$
- **AIC**: $AIC = -2 \log(\hat{L}) + 2p$

Pilih model dengan nilai BIC/AIC terendah

---

### 2.1. Bayesian Gaussian Mixture Models (BGMM)

- Memilih jumlah komponen Gaussian secara otomatis
- Lebih tahan terhadap overfitting
- Tidak perlu menentukan jumlah cluster manual

---

### 2.2. Algoritma Lain untuk Deteksi Anomali / Kebaruan

- **PCA**: Deteksi berdasarkan error rekonstruksi
- **Fast-MCD**: Berdasarkan estimasi kovarians yang robust
- **Isolation Forest**: Isolasi instansi dengan pohon acak
- **LOF (Local Outlier Factor)**: Deteksi berbasis densitas lokal
- **One-class SVM**: Belajar dari data normal dan mendeteksi penyimpangan

---