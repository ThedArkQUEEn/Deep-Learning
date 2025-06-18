# Penjelasan Chapter 07 - Ensemble Learning and Random Forests

## Pengantar
Bab ini membahas bagaimana menggabungkan beberapa model pembelajaran mesin untuk meningkatkan **kinerja prediksi**, yang dikenal sebagai **pembelajaran ensembel**. Teknik ini seringkali lebih baik dibandingkan menggunakan **model tunggal**, karena dapat mengurangi bias dan variance.

## Konsep Utama
- **Pembelajaran Ensembel (Ensemble Learning)**: Menggabungkan prediksi dari beberapa model dasar untuk menghasilkan prediksi yang lebih baik daripada model individu.
- **Pengklasifikasi Voting (Voting Classifiers)**: Menggabungkan prediksi dari beberapa pengklasifikasi (model) dan memprediksi kelas berdasarkan mayoritas suara.
- **Bagging dan Pasting**: Metode untuk melatih beberapa model pada subset data pelatihan yang berbeda secara acak.
- **Hutan Acak (Random Forests)**: Ensembel dari pohon keputusan, dilatih menggunakan bagging atau pasting dan fitur subset acak.
- **Boosting**: Metode yang menggabungkan beberapa model lemah secara berurutan, di mana setiap model berikutnya mencoba memperbaiki kesalahan model sebelumnya.
- **AdaBoost (Adaptive Boosting)**: Algoritma boosting populer di mana setiap model diberi bobot berdasarkan kinerjanya.
- **Gradient Boosting**: Algoritma boosting lain yang melatih model baru untuk memprediksi residu (kesalahan) dari model sebelumnya.
- **Stacking**: Metode untuk menggabungkan prediksi dari beberapa model dengan melatih model "meta-learner" untuk memprediksi output akhir berdasarkan output model dasar.

---

## 1. Pengklasifikasi Voting (Voting Classifiers)
### Voting Hard
Memilih kelas dengan **suara terbanyak** dari semua pengklasifikasi:

$y_{pred} = \underset{k}{\operatorname{mode}} (y_1, y_2, ..., y_n)$

Di mana ( $y_1, y_2, ..., y_n$ ) adalah prediksi dari setiap model.


### Voting Soft
Menghitung rata-rata probabilitas prediksi dan memilih kelas dengan probabilitas tertinggi:

$P(y_j) = \frac{1}{n} \sum_{i=1}^{n} P_i(y_j)$

## 2. Bagging dan Pasting
### Bagging (Bootstrap Aggregating)
Setiap model dilatih dengan sampel acak dengan penggantian dari data pelatihan.

### Pasting
Setiap model dilatih dengan sampel acak tanpa penggantian.
Prediksi akhir diambil dari mayoritas prediksi:

$y_{ensemble} = \frac{1}{n} \sum_{i=1}^{n} y_i$

### 3. Hutan Acak (Random Forests)
Random Forest adalah bagging dengan tambahan pemilihan fitur acak untuk setiap pohon keputusan.

Prediksi ensemble untuk klasifikasi:

$y_{pred} = \underset{k}{\operatorname{mode}} (y_1, y_2, ..., y_n)$

Prediksi ensemble untuk regresi:

$y_{pred} = \frac{1}{n} \sum_{i=1}^{n} y_i$

### 4. AdaBoost (Adaptive Boosting)
Setiap instance pelatihan diberi bobot ( $w_i$ ) dan diperbarui berdasarkan kesalahan klasifikasi:

$w_i = w_i \cdot e^{\alpha}$

Di mana ( $\alpha$ ) adalah bobot kesalahan.

### 5. Gradient Boosting
Melatih model secara berurutan dengan memprediksi residu dari model sebelumnya:

$r_i = y_{\text{actual}} - y_{\text{pred}}$

Model baru dilatih untuk memprediksi residu guna meningkatkan akurasi.

### 6. Stacking
Model "meta-learner" melatih output dari model dasar sebagai input:

$y_{pred} = f(y_1, y_2, ..., y_n)$

Di mana ($f$) adalah model meta-learning.

### Implementasi dalam Scikit-Learn
Teknik-teknik ini dapat diimplementasikan menggunakan Scikit-Learn di Python:


**from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier**

**Voting Classifier**

voting_clf = VotingClassifier(estimators=[
    ('lr', LogisticRegression()),
    ('rf', RandomForestClassifier()),
    ('svc', SVC(probability=True))
], voting='soft')

**Bagging Classifier**

bagging_clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=50)

**Random Forest**

rf_clf = RandomForestClassifier(n_estimators=50)

**AdaBoost**

adaboost_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=50)

**Gradient Boosting**

gb_clf = GradientBoostingClassifier(n_estimators=50)