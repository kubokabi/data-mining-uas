1. Perhitungan K-Nearest Neighbors (KNN) dengan Python

---

```markdown
# 🧠 Perhitungan K-Nearest Neighbors (KNN) dengan Python

Repositori ini berisi implementasi algoritma **K-Nearest Neighbors (KNN)** menggunakan dataset **COVID-19**, dengan data yang sudah dikonversi ke dalam format numerik.

---

## 📖 Deskripsi Proyek

Pada proyek ini, kita akan menggunakan **KNN** untuk mengklasifikasikan dataset **COVID-19** berdasarkan fitur yang telah dikonversi ke dalam bentuk numerik. Tahapan utama dalam proyek ini meliputi:

1. **Mengimpor library yang diperlukan**.
2. **Membaca dataset dan melakukan eksplorasi awal**.
3. **Memisahkan dataset menjadi fitur predictor (`X`) dan target (`y`)**.
4. **Melakukan train-test split (70% training - 30% testing)**.
5. **Membangun model KNN dan melakukan pelatihan (`fit`)**.
6. **Melakukan prediksi dan evaluasi akurasi model**.
7. **Memprediksi data baru menggunakan model yang sudah dilatih**.

---

## 🚀 Instalasi

Sebelum menjalankan kode, pastikan pustaka yang dibutuhkan telah terinstal dengan menjalankan perintah berikut:

```bash
pip install pandas numpy scikit-learn
```

---

## 📂 1. Import Library

```python
# Import library yang dibutuhkan
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
```

📌 **Penjelasan:**
- `pandas` → Untuk membaca dan mengelola dataset.
- `numpy` → Untuk operasi perhitungan numerik.
- `train_test_split` → Untuk membagi dataset menjadi data latih dan uji.
- `KNeighborsClassifier` → Untuk membangun model **KNN**.
- `metrics` → Untuk evaluasi model.

---

## 📊 2. Membaca Dataset CSV

```python
# Membaca dataset dari file CSV
df = pd.read_csv('/content/drive/MyDrive/Dataset/gizi.csv')

# Menampilkan 5 data pertama
print(df.head())
```

📌 **Penjelasan:**
- Dataset dibaca menggunakan `pd.read_csv()`.
- Jika dataset tidak ada di folder proyek, pastikan jalur lengkapnya ditentukan, misalnya: `"D:/data/gizi.csv"`.
- `df.head()` digunakan untuk melihat **5 data pertama**.

### **Output Contoh Dataset:**
```
   Tinggi  Berat  L Perut  L Panggul  Lemak  Label
0   160.0     70    78.0       99.0   33.3      3
1   162.0     56    74.0       90.0   31.7      3
2   155.0     63    76.5       95.5   37.8      3
3   156.0     54    74.0       88.0   31.0      2
4   155.0     55    79.0       88.0   27.0      3
```

---

## 📌 3. Menentukan Fitur Predictor (`X`) dan Target (`y`)

```python
# Menentukan variabel X (fitur) dan y (label target)
X = df[['Tinggi', 'Berat', 'L Perut', 'L Panggul', 'Lemak']]
y = df['Label']

# Menampilkan X
print(X.head())

# Menampilkan y
print(y.head())
```

📌 **Penjelasan:**
- `X` berisi fitur predictor: **Tinggi, Berat, L Perut, L Panggul, dan Lemak**.
- `y` berisi target klasifikasi: **Label**.

---

## 🏗️ 4. Membagi Data: Train-Test Split

```python
# Membagi dataset menjadi training (70%) dan testing (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Menampilkan jumlah data training dan testing
print(f"Jumlah data training: {X_train.shape[0]}")
print(f"Jumlah data testing: {X_test.shape[0]}")
```

📌 **Penjelasan:**
- `test_size=0.3` → 30% data digunakan sebagai data uji.
- `random_state=1` → Agar pembagian dataset tetap konsisten setiap kali dijalankan.

---

## 🤖 5. Membangun Model KNN

```python
# Membuat model KNN dengan K=3 dan Euclidean Distance
model = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='euclidean')

# Melatih model menggunakan data training
model.fit(X_train, y_train)
```

📌 **Penjelasan:**
- **`n_neighbors=3`** → Model menggunakan **3 tetangga terdekat**.
- **`weights='distance'`** → Memberikan bobot berdasarkan jarak.
- **`metric='euclidean'`** → Menggunakan **Euclidean Distance** untuk menghitung jarak antar titik.

---

## 🔮 6. Prediksi Data Testing

```python
# Melakukan prediksi pada data testing
y_pred = model.predict(X_test)

# Menampilkan hasil prediksi
print(y_pred)
```

📌 **Penjelasan:**
- Model melakukan prediksi terhadap **data uji (30% dataset yang dipisahkan sebelumnya).**

---

## 🎯 7. Pengujian Model dengan Data Baru

```python
# Memprediksi data baru
data_baru = [[159, 49, 65, 87, 24.6]]
hasil_prediksi = model.predict(data_baru)

print(f"Hasil Prediksi Data Baru: {hasil_prediksi}")
```

📌 **Hasil Prediksi:** `[2]`  
Artinya, **model memprediksi bahwa data tersebut masuk dalam kategori Label 2**.

---

## 📊 8. Evaluasi Model

```python
# Menghitung akurasi model
akurasi = metrics.accuracy_score(y_test, y_pred)

print(f"Accuracy: {akurasi:.2f}")
```

📌 **Hasil:**
```
Accuracy: 1.00
```

