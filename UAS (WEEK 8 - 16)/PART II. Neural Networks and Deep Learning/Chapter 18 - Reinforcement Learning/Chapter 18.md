# Penjelasan Chapter 18 - Reinforcement Learning

Reinforcement Learning (RL) adalah cabang Machine Learning yang melatih agen untuk memilih tindakan yang akan memaksimalkan imbalan mereka seiring waktu dalam lingkungan tertentu. Agen mempelajari strategi, yang disebut kebijakan, untuk mendapatkan imbalan maksimal.

## Pencarian Kebijakan

Dalam pencarian kebijakan, agen belajar dengan mencoba berbagai tindakan dan mengamati hasilnya untuk menemukan kebijakan optimal.

## Pengantar OpenAI Gym

OpenAI Gym adalah toolkit untuk mengembangkan dan membandingkan algoritma reinforcement learning. Ini menyediakan lingkungan standar untuk melatih agen.

## Kebijakan Jaringan Saraf

Kebijakan dapat diimplementasikan menggunakan jaringan saraf, di mana input adalah observasi lingkungan dan output adalah tindakan yang harus diambil oleh agen.

## Mengevaluasi Tindakan: Masalah Penugasan Kredit

Masalah penugasan kredit adalah tantangan dalam RL di mana agen harus menentukan tindakan mana yang berkontribusi pada imbalan jangka panjang, terutama ketika imbalan tertunda.

## Policy Gradients

Policy Gradients adalah kelas algoritma RL yang secara langsung mengoptimalkan parameter kebijakan untuk memaksimalkan total imbalan. Ide dasarnya adalah untuk menggeser parameter kebijakan ke arah yang meningkatkan probabilitas tindakan yang baik dan menurunkan probabilitas tindakan yang buruk.

### Algoritma Policy Gradients Dasar

Dalam bentuk paling sederhana, algoritma Policy Gradients menghitung gradien fungsi tujuan (misalnya, total imbalan yang diharapkan) sehubungan dengan parameter kebijakan dan kemudian melakukan langkah gradien naik.

**Rumus umum untuk gradien kebijakan:**
$$\nabla_{\theta} J(\theta) \approx \frac{1}{M} \sum_{i=1}^{M} \sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}(a_{t}^{(i)} | s_{t}^{(i)}) R(\tau^{(i)})$$
Di mana:
- $J(\theta)$ adalah fungsi tujuan (imbalan yang diharapkan).
- $\theta$ adalah parameter kebijakan.
- $M$ adalah jumlah episode.
- $T$ adalah panjang episode.
- $\pi_{\theta}(a | s)$ adalah probabilitas memilih tindakan $a$ di keadaan $s$ di bawah kebijakan $\theta$.
- $R(\tau)$ adalah total imbalan kumulatif dari episode $\tau$.

### Policy Gradients dengan Baseline

Untuk mengurangi varians gradien, sering digunakan baseline, yaitu nilai yang dikurangkan dari imbalan.

**Rumus dengan baseline:**
$$\nabla_{\theta} J(\theta) \approx \frac{1}{M} \sum_{i=1}^{M} \sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}(a_{t}^{(i)} | s_{t}^{(i)}) (R(\tau^{(i)}) - b)$$
Di mana $b$ adalah baseline, seringkali merupakan estimasi nilai keadaan.

## Proses Keputusan Markov (MDP)

MDP adalah kerangka matematis untuk memodelkan masalah pengambilan keputusan di mana hasil parsial acak dan di bawah kendali agen. MDP didefinisikan oleh:
- Keadaan ($S$)
- Tindakan ($A$)
- Fungsi transisi probabilitas ($P(s' | s, a)$)
- Fungsi imbalan ($R(s, a, s')$)
- Faktor diskon ($\gamma$)

## Pembelajaran Perbedaan Temporal (TD Learning)

TD Learning adalah kelas algoritma RL tanpa model yang belajar dari pengalaman tanpa model lingkungan. Ini memperbarui estimasi nilai berdasarkan estimasi nilai berikutnya (bootstrap).

## Q-Learning

Q-Learning adalah algoritma RL TD tanpa model yang belajar fungsi nilai aksi-keadaan (Q-function), yang memberikan nilai yang diharapkan dari mengambil tindakan tertentu di keadaan tertentu dan kemudian mengikuti kebijakan optimal.

**Persamaan Bellman untuk Q-Learning:**
$$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$
Di mana:
- $Q(s, a)$ adalah nilai Q untuk keadaan $s$ dan tindakan $a$.
- $\alpha$ adalah laju pembelajaran.
- $r$ adalah imbalan.
- $\gamma$ adalah faktor diskon.
- $s'$ adalah keadaan berikutnya.
- $a'$ adalah tindakan berikutnya.

## Kebijakan Eksplorasi

Untuk memastikan agen menjelajahi lingkungan secara efektif, kebijakan eksplorasi digunakan, seperti $\epsilon$-greedy, yang memilih tindakan acak dengan probabilitas $\epsilon$ dan tindakan optimal dengan probabilitas $1-\epsilon$.

## Q-Learning Aproksimasi dan Deep Q-Learning (DQN)

Ketika ruang keadaan dan tindakan terlalu besar untuk tabel Q, Q-Learning aproksimasi digunakan di mana Q-function diaproksimasi oleh fungsi parameter, seperti jaringan saraf (Deep Q-Network).

### Implementasi Deep Q-Learning

DQN melatih jaringan saraf untuk memprediksi Q-value untuk setiap tindakan dalam keadaan tertentu.

## Varian Deep Q-Learning

### Target Q-Value Tetap

Menggunakan jaringan target terpisah dengan parameter yang diperbarui secara berkala dari jaringan utama untuk menstabilkan pelatihan.

### Double DQN

Mengatasi overestimasi Q-value dengan menggunakan jaringan utama untuk memilih tindakan dan jaringan target untuk mengevaluasi Q-value dari tindakan tersebut.

### Prioritized Experience Replay

Memberikan prioritas yang lebih tinggi untuk pengalaman yang menghasilkan kesalahan TD yang lebih besar, memungkinkan agen untuk belajar lebih efisien dari pengalaman yang lebih informatif.

### Dueling DQN

Menguraikan Q-function menjadi komponen nilai keadaan (state-value) dan keuntungan (advantage), memungkinkan jaringan untuk belajar estimasi nilai keadaan secara terpisah dari keuntungan tindakan.

## Pustaka TF-Agents

TF-Agents adalah pustaka TensorFlow yang menyediakan komponen modular untuk merancang, mengimplementasikan, dan mengevaluasi algoritma RL.

### Lingkungan TF-Agents

TF-Agents menyediakan abstraksi untuk lingkungan RL, memungkinkan interaksi yang konsisten dengan berbagai lingkungan.

### Spesifikasi Lingkungan

Mendefinisikan ruang observasi, ruang tindakan, dan jenis data untuk imbalan.

### Pembungkus Lingkungan dan Pra-pemrosesan Atari

Pembungkus lingkungan digunakan untuk memodifikasi lingkungan (misalnya, pra-pemrosesan observasi). Contoh pra-pemrosesan untuk game Atari seperti mengubah ukuran gambar, mengubah menjadi grayscale, dan menumpuk frame.

### Arsitektur Pelatihan

Mencakup komponen seperti jaringan saraf, agen, buffer replay, metrik pelatihan, dan driver pengumpul data.

### Membuat Deep Q-Network

Membangun arsitektur jaringan saraf yang digunakan oleh agen DQN.

### Membuat Agen DQN

Menginisialisasi agen DQN dengan Deep Q-Network yang telah dibuat.

### Membuat Buffer Replay dan Observer yang Sesuai

Buffer replay menyimpan pengalaman agen, dan observer digunakan untuk menambahkan pengalaman ke buffer.

### Membuat Metrik Pelatihan

Metrik digunakan untuk memantau kinerja agen selama pelatihan (misalnya, imbalan rata-rata per episode).

### Membuat Collect Driver

Collect Driver bertanggung jawab untuk mengumpulkan pengalaman dari lingkungan dan menambahkannya ke buffer replay.

### Membuat Dataset

Dataset dibuat dari buffer replay untuk melatih agen secara efisien.

### Membuat Loop Pelatihan

Mendefinisikan proses iteratif di mana agen berinteraksi dengan lingkungan, mengumpulkan pengalaman, dan memperbarui jaringannya.

## Gambaran Umum Beberapa Algoritma RL Populer

*   **REINFORCE (Monte Carlo Policy Gradient):** Mengupdate kebijakan berdasarkan total imbalan episode.
*   **Actor-Critic:** Menggabungkan elemen policy gradient dan Q-learning, menggunakan 'actor' untuk memilih tindakan dan 'critic' untuk mengevaluasi tindakan tersebut.
*   **A2C (Advantage Actor-Critic):** Versi synchronous dan deterministik dari A3C.
*   **A3C (Asynchronous Advantage Actor-Critic):** Menggunakan beberapa agen yang berinteraksi dengan lingkungannya secara paralel.
*   **PPO (Proximal Policy Optimization):** Salah satu algoritma Policy Gradient yang paling populer, yang mencoba untuk menjaga agar kebijakan baru tidak terlalu jauh dari kebijakan lama.
*   **DDPG (Deep Deterministic Policy Gradient):** Algoritma off-policy actor-critic untuk ruang tindakan kontinu.
*   **TD3 (Twin Delayed DDPG):** Peningkatan dari DDPG yang mengurangi overestimasi Q-value.
*   **SAC (Soft Actor-Critic):** Algoritma off-policy yang mengoptimalkan kompromi antara imbalan dan entropi (eksplorasi).