<span style="color:red">1. Perhitungan K-Nearest Neighbors (KNN) dengan Python (Pertemuan 8)</span>

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

<span style="color:red">2. Decision Tree (Pertemuan 9)</span>


---

```markdown
# 🌳 Decision Tree - Klasifikasi Data dengan Python

Repositori ini berisi implementasi **Decision Tree** untuk klasifikasi menggunakan **dataset iris** dan **dataset cuaca untuk prediksi bermain golf**.  

---

## 📖 Deskripsi Proyek

Dalam proyek ini, kita akan membahas dan mengimplementasikan **Decision Tree** untuk **binary classification** dan **multiclass classification**.  
Tahapan utama dalam proyek ini meliputi:

1. **Memahami klasifikasi biner dan klasifikasi banyak kelas**.
2. **Menjelaskan konsep Decision Tree**.
3. **Melakukan implementasi Decision Tree menggunakan dataset Iris**.
4. **Melatih model dan melakukan evaluasi**.
5. **Memprediksi data baru**.
6. **Membuat visualisasi Decision Tree**.

---

## 📌 1. Klasifikasi Data

### 🟥 Binary Classification
- Klasifikasi **biner** membagi data menjadi **dua kelas**.
- Contohnya adalah **klasifikasi email spam atau bukan spam**.
- **Contoh lain:** Membedakan **buah apel vs pisang** berdasarkan fitur seperti warna dan bentuk.

### 🟦 Multiclass Classification
- Klasifikasi **banyak kelas** membagi data menjadi **lebih dari dua kategori**.
- **Contoh:** Dataset **Iris** yang mengklasifikasikan tiga spesies bunga **(Setosa, Versicolor, Virginica)** berdasarkan panjang dan lebar sepal serta petal.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
```

---

## 📊 2. Dataset Iris

Dataset **Iris** terdiri dari **4 fitur** utama:
- **SepalLengthCm** → Panjang sepal
- **SepalWidthCm** → Lebar sepal
- **PetalLengthCm** → Panjang petal
- **PetalWidthCm** → Lebar petal
- **Species** → Label klasifikasi (Setosa, Versicolor, Virginica)

### 🔹 Membaca Dataset

```python
# Membaca file iris.csv
iris = pd.read_csv("Iris.csv")

# Menampilkan 5 baris pertama
print(iris.head())
```

### **Output Contoh Dataset**
```
   Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm      Species
0   1           5.1          3.5          1.4          0.2  Iris-setosa
1   2           4.9          3.0          1.4          0.2  Iris-setosa
2   3           4.7          3.2          1.3          0.2  Iris-setosa
3   4           4.6          3.1          1.5          0.2  Iris-setosa
4   5           5.0          3.6          1.4          0.2  Iris-setosa
```

---

## 🏗️ 3. Preprocessing Data

### 🔹 Menghapus Kolom yang Tidak Diperlukan

```python
# Menghapus kolom 'Id' karena tidak relevan
iris.drop('Id', axis=1, inplace=True)
```

### 🔹 Memisahkan Fitur dan Label

```python
# Memisahkan atribut (X) dan label (y)
X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris['Species']
```

### 🔹 Membagi Data Menjadi Training dan Testing

```python
# Membagi dataset menjadi data latih dan data uji (90% latih, 10% uji)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)
```

---

## 🌳 4. Membuat Model Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier

# Membuat model Decision Tree
tree_model = DecisionTreeClassifier()

# Melatih model dengan data latih
tree_model.fit(X_train, y_train)
```

---

## 🎯 5. Prediksi dan Evaluasi Model

### 🔹 Memprediksi Data Uji

```python
# Melakukan prediksi pada data uji
y_pred = tree_model.predict(X_test)
```

### 🔹 Menghitung Akurasi Model

```python
from sklearn.metrics import accuracy_score

# Evaluasi model menggunakan metrik akurasi
acc_score = round(accuracy_score(y_test, y_pred), 3)
print(f"Accuracy: {acc_score}")
```

**Output:**
```
Accuracy: 0.97
```

📌 **Kesimpulan:**
- Model Decision Tree memiliki **akurasi 97%** pada dataset Iris.

---

## 🔮 6. Memprediksi Data Baru

```python
# Prediksi spesies bunga iris baru
new_data = [[6.2, 3.4, 5.4, 2.3]]
prediction = tree_model.predict(new_data)

print(f"Prediksi Spesies: {prediction[0]}")
```

**Output:**
```
Prediksi Spesies: Iris-virginica
```

---

## 📊 7. Visualisasi Decision Tree

```python
from sklearn.tree import export_graphviz

# Menyimpan decision tree dalam file dot
export_graphviz(
    tree_model,
    out_file="iris_tree.dot",
    feature_names=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],
    class_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
    rounded=True,
    filled=True
)
```

📌 **Catatan:**  
- File **iris_tree.dot** bisa dikonversi menjadi gambar **PNG** menggunakan situs seperti [Online Convert](https://onlineconvertfree.com/converter/images/).

---

## 🌦️ 8. Contoh Lain: Decision Tree untuk Prediksi Golf

Dataset cuaca digunakan untuk memprediksi apakah seseorang akan bermain golf berdasarkan kondisi cuaca.

### 🔹 Contoh Data

| Outlook  | Temperature | Humidity | Windy | Play Golf |
|----------|------------|----------|-------|-----------|
| Rainy    | Hot        | High     | False | No        |
| Rainy    | Hot        | High     | True  | No        |
| Overcast | Hot        | High     | False | Yes       |
| Sunny    | Mild       | High     | False | Yes       |
| Sunny    | Cold       | Normal   | True  | No        |

### 🔹 Implementasi Model Decision Tree

```python
# Membaca dataset cuaca
golf = pd.read_csv("golf.csv")

# Memisahkan fitur dan label
X_golf = golf[['Outlook', 'Temperature', 'Humidity', 'Windy']]
y_golf = golf['Play Golf']

# Mengonversi data kategori menjadi numerik
X_golf = pd.get_dummies(X_golf)

# Membagi dataset
X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_golf, y_golf, test_size=0.2, random_state=42)

# Membuat dan melatih model Decision Tree
tree_golf = DecisionTreeClassifier()
tree_golf.fit(X_train_g, y_train_g)

# Evaluasi model
y_pred_g = tree_golf.predict(X_test_g)
acc_golf = accuracy_score(y_test_g, y_pred_g)

print(f"Akurasi Model Golf: {acc_golf:.2f}")
```

---

## 🎯 Kesimpulan

- **Decision Tree** adalah algoritma powerful untuk klasifikasi data.
- **Dataset Iris** berhasil diklasifikasikan dengan **akurasi 97%**.
- **Decision Tree dapat digunakan untuk berbagai dataset**, termasuk prediksi bermain golf berdasarkan cuaca.

---

## 📌 Referensi
- [Towards Data Science - Decision Tree](https://towardsdatascience.com/decision-trees-in-machine-learning-641b9c4e8052)

---

<span style="color:red">3. Perhitungan Python Naïve Bayes Classifier</span>

---

```markdown
# 🧠 Perhitungan Python Naïve Bayes Classifier

Repositori ini berisi implementasi **Naïve Bayes Classifier** menggunakan dataset **covid19** yang telah dikonversi ke dalam format numerik.

---

## 📖 Deskripsi Proyek

Pada proyek ini, kita akan menggunakan **Naïve Bayes Classifier** untuk **klasifikasi dataset covid19** berdasarkan fitur yang telah dikonversi ke dalam bentuk numerik. Tahapan utama dalam proyek ini meliputi:

1. **Mengimpor library yang diperlukan**.
2. **Membaca dataset dan melakukan eksplorasi awal**.
3. **Memisahkan dataset menjadi fitur predictor (`X`) dan target (`y`)**.
4. **Melakukan train-test split (70% training - 30% testing)**.
5. **Membangun model Naïve Bayes dan melakukan pelatihan (`fit`)**.
6. **Melakukan prediksi dan evaluasi akurasi model**.

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
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
```

📌 **Penjelasan:**
- `pandas` → Untuk membaca dan mengelola dataset.
- `numpy` → Untuk operasi perhitungan numerik.
- `train_test_split` → Untuk membagi dataset menjadi data latih dan uji.
- `GaussianNB` → Untuk membangun model **Naïve Bayes**.
- `metrics` → Untuk evaluasi model.

---

## 📊 2. Membaca Dataset

```python
# Membaca dataset dari file CSV
df = pd.read_csv('/content/drive/MyDrive/Data Mining/studi_num.csv')

# Menampilkan 5 data pertama
print(df.head())
```

📌 **Penjelasan:**
- Dataset dibaca menggunakan `pd.read_csv()`.
- Jika dataset tidak ada di folder proyek, pastikan jalur lengkapnya ditentukan, misalnya: `"D:/data/studi_num.csv"`.
- `df.head()` digunakan untuk melihat **5 data pertama**.

### **Output Contoh Dataset**
```
   JURUSAN  GENDER  ASAL_SEKOLAH  RERATA_SKS  ASISTEN  LAMA_STUDI
0        1       1            1           1        1          1
1        1       1            1           1        1          1
2        1       1            1           1        1          1
3        1       1            1           1        1          1
4        1       1            1           1        1          1
```

---

## 📌 3. Menentukan Fitur Predictor (`X`) dan Target (`y`)

```python
# Menentukan variabel X (fitur) dan y (label target)
X = df.iloc[:, :-1].values
y = df.iloc[:, 5].values
```

📌 **Penjelasan:**
- `X` berisi **semua kolom kecuali kolom terakhir** sebagai fitur predictor.
- `y` berisi **kolom terakhir** sebagai target klasifikasi.

---

## 🏗️ 4. Membagi Data: Train-Test Split

```python
# Membagi dataset menjadi training (70%) dan testing (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
```

📌 **Penjelasan:**
- `test_size=0.3` → 30% data digunakan sebagai data uji.
- `random_state=1` → Agar pembagian dataset tetap konsisten setiap kali dijalankan.

---

## 🤖 5. Membangun Model Naïve Bayes

```python
# Membuat model Naïve Bayes
model = GaussianNB()

# Melatih model menggunakan data training
model.fit(X_train, y_train)
```

📌 **Penjelasan:**
- `GaussianNB()` → Menggunakan **Gaussian Naïve Bayes** sebagai model.
- `model.fit(X_train, y_train)` → Model dilatih menggunakan data training.

---

## 🔮 6. Prediksi Data Testing

```python
# Melakukan prediksi pada data testing
y_pred = model.predict(X_test)

# Menampilkan hasil prediksi
print(y_pred)
```

📌 **Output Contoh:**
```
[1, 2, 1, 1, 1]
```

---

## 📊 7. Evaluasi Model

### 🔹 Menghitung Akurasi Model

```python
# Menghitung akurasi model
accuracy = metrics.accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
```

📌 **Hasil:**
```
Accuracy: 0.75
```

📌 **Kesimpulan:**
- Model Naïve Bayes memiliki **akurasi 75%** pada dataset.

---

## 🎯 Kesimpulan

- **Naïve Bayes** adalah algoritma powerful untuk klasifikasi data.
- **Dataset studi mahasiswa** berhasil diklasifikasikan dengan **akurasi 75%**.
- **Naïve Bayes bekerja dengan baik pada dataset dengan fitur kategori**.

---

## 📌 Referensi
- [Scikit-learn Naïve Bayes Documentation](https://scikit-learn.org/stable/modules/naive_bayes.html)

<span style="color:red">4. Perhitungan Python K-Means</span>

---

```markdown
# 🔍 Perhitungan Python K-Means Clustering

Repositori ini berisi implementasi **K-Means Clustering** menggunakan **dataset gizi bayi** dalam format **Excel (.xlsx)**.

---

## 📖 Deskripsi Proyek

Pada proyek ini, kita akan menggunakan **K-Means Clustering** untuk mengelompokkan data bayi berdasarkan tinggi badan (**TB**) dan berat badan (**BB**).  
Tahapan utama dalam proyek ini meliputi:

1. **Mengimpor library yang diperlukan**.
2. **Membaca dataset dan melakukan eksplorasi awal**.
3. **Memilih fitur yang akan digunakan untuk clustering**.
4. **Melakukan visualisasi awal data**.
5. **Melakukan normalisasi data dengan MinMaxScaler**.
6. **Mengimplementasikan algoritma K-Means untuk clustering**.
7. **Menganalisis dan memvisualisasikan hasil clustering**.

---

## 🚀 Instalasi

Sebelum menjalankan kode, pastikan pustaka yang dibutuhkan telah terinstal dengan menjalankan perintah berikut:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn openpyxl
```

---

## 📂 1. Import Library

```python
# Import library yang dibutuhkan
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
```

📌 **Penjelasan:**
- `pandas` → Untuk membaca dan mengelola dataset.
- `numpy` → Untuk operasi perhitungan numerik.
- `seaborn` → Untuk visualisasi data.
- `matplotlib.pyplot` → Untuk membuat grafik.
- `KMeans` → Algoritma **K-Means Clustering** dari scikit-learn.
- `MinMaxScaler` → Untuk normalisasi data.

---

## 📊 2. Membaca Dataset

```python
# Membaca dataset dari file Excel
gizi = pd.read_excel("dataset_gizi.xlsx")

# Menampilkan 5 data pertama
print(gizi.head())
```

📌 **Penjelasan:**
- Dataset dibaca menggunakan `pd.read_excel()`.
- Jika dataset tidak ada di folder proyek, pastikan jalur lengkapnya ditentukan, misalnya: `"D:/data/dataset_gizi.xlsx"`.
- `gizi.head()` digunakan untuk melihat **5 data pertama**.

### **Output Contoh Dataset**
```
   No  Balita ke-    TB    BB
0   1     Balita 1  52.0   5.8
1   2     Balita 2  51.0   5.0
2   3     Balita 3  71.5   8.5
3   4     Balita 4  55.0   5.5
4   5     Balita 5  92.5   6.5
```

---

## 🏗️ 3. Eksplorasi Data

### 🔹 Menampilkan Informasi Dataset

```python
# Menampilkan informasi dataset
print(gizi.info())
```

📌 **Penjelasan:**
- `gizi.info()` digunakan untuk melihat tipe data dan jumlah nilai yang tersedia.

### 🔹 Memilih Fitur yang Digunakan untuk Clustering

```python
# Memilih fitur tinggi badan (TB) dan berat badan (BB)
gizi_x = gizi.iloc[:, 2:4]

# Menampilkan 5 baris pertama
print(gizi_x.head())
```

📌 **Penjelasan:**
- `gizi.iloc[:, 2:4]` → Memilih **kolom ke-2 hingga ke-3** yaitu **TB dan BB**.

---

## 📊 4. Visualisasi Data Awal

```python
# Scatter plot untuk melihat sebaran data
plt.scatter(gizi.TB, gizi.BB, s=10, c="c", marker="o", alpha=1)
plt.show()
```

📌 **Penjelasan:**
- `plt.scatter()` → Membuat plot **Tinggi Badan (TB)** vs **Berat Badan (BB)**.

---

## 🔄 5. Normalisasi Data

```python
# Normalisasi data dengan MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(np.array(gizi_x))

# Menampilkan hasil normalisasi
print(x_scaled)
```

📌 **Penjelasan:**
- **MinMaxScaler** memastikan nilai dalam **rentang 0 hingga 1** untuk mempermudah perhitungan clustering.

---

## 🤖 6. Implementasi K-Means Clustering

```python
# Membuat model K-Means dengan 5 cluster
kmeans = KMeans(n_clusters=5, random_state=123)

# Melakukan clustering
kmeans.fit(x_scaled)
```

📌 **Penjelasan:**
- `n_clusters=5` → Membagi dataset menjadi **5 kelompok (cluster)**.
- `random_state=123` → Supaya hasil clustering tetap sama setiap kali dijalankan.

---

## 📊 7. Hasil Clustering

```python
# Menampilkan pusat cluster
print(kmeans.cluster_centers_)

# Menambahkan label cluster ke dalam dataset
gizi["cluster"] = kmeans.labels_

# Menampilkan dataset dengan label cluster
print(gizi.head())
```

📌 **Penjelasan:**
- `kmeans.cluster_centers_` → Menampilkan koordinat pusat setiap cluster.
- `kmeans.labels_` → Menambahkan **label cluster** ke dataset.

### **Output Contoh Dataset dengan Cluster**
```
   No  Balita ke-    TB    BB  cluster
0   1     Balita 1  52.0   5.8        0
1   2     Balita 2  51.0   5.0        1
2   3     Balita 3  71.5   8.5        2
3   4     Balita 4  55.0   5.5        3
4   5     Balita 5  92.5   6.5        4
```

---

## 📊 8. Visualisasi Hasil Clustering

```python
# Visualisasi hasil clustering
plt.scatter(x_scaled[:, 0], x_scaled[:, 1], s=100, c=gizi.cluster, marker="o", alpha=1)

# Menampilkan centroid cluster dengan warna merah
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c="red", s=200, marker="s")

plt.title("Hasil Clustering K-Means")
plt.colorbar()
plt.show()
```

📌 **Penjelasan:**
- **Cluster bayi ditampilkan dengan warna berbeda**.
- **Centroid cluster ditandai dengan kotak merah**.

---

## 🎯 Kesimpulan

- **Dataset gizi bayi** berhasil dikelompokkan ke dalam **5 cluster**.
- **Centroid cluster** menunjukkan titik tengah dari setiap kelompok data.
- **Visualisasi scatter plot membantu memahami hasil clustering**.

---

## 📌 Referensi
- [Scikit-learn K-Means Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

<span style="color:red">5. Perhitungan Python Apriory</span>

---

```markdown
# 🛒 Perhitungan Python Apriori - Market Basket Analysis

Repositori ini berisi implementasi **algoritma Apriori** untuk melakukan **Market Basket Analysis** menggunakan dataset **Grocery Store Dataset**.

---

## 📖 Deskripsi Proyek

Dalam proyek ini, kita akan menggunakan **Apriori Algorithm** untuk **menemukan pola asosiasi** antara produk yang sering dibeli bersama dalam dataset transaksi supermarket.  

Tahapan utama dalam proyek ini meliputi:

1. **Mengimpor library yang diperlukan**.
2. **Membaca dataset transaksi dan melakukan eksplorasi awal**.
3. **Memproses data transaksi menjadi format yang bisa diproses oleh algoritma Apriori**.
4. **Menjalankan algoritma Apriori untuk menemukan pola asosiasi**.
5. **Menganalisis hasil aturan asosiasi (Association Rules)**.

---

## 🚀 Instalasi

Sebelum menjalankan kode, pastikan pustaka yang dibutuhkan telah terinstal dengan menjalankan perintah berikut:

```bash
pip install pandas numpy mlxtend
```

---

## 📂 1. Import Library

```python
# Import library yang dibutuhkan
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
```

📌 **Penjelasan:**
- `pandas` → Untuk membaca dan mengelola dataset.
- `numpy` → Untuk operasi perhitungan numerik.
- `apriori` → Algoritma Apriori untuk menemukan frequent itemsets.
- `association_rules` → Untuk membangun aturan asosiasi dari frequent itemsets.

---

## 📊 2. Membaca Dataset

```python
# Membaca dataset dari file CSV
df = pd.read_csv("GroceryStoreDataset.csv", names=['products'], sep=',')

# Menampilkan 5 data pertama
print(df.head())
```

📌 **Penjelasan:**
- Dataset dibaca menggunakan `pd.read_csv()`, dengan separator `,`.
- `names=['products']` digunakan untuk memberikan nama kolom **"products"**.
- `df.head()` digunakan untuk melihat **5 data pertama**.

### **Output Contoh Dataset**
```
            products
0    MILK,BREAD,BISCUIT
1  BREAD,MILK,BISCUIT,CORNFLAKES
2        BREAD,TEA,BOURNVITA
3    JAM,MAGGI,BREAD,MILK
4        MAGGI,TEA,BISCUIT
```

---

## 🏗️ 3. Mengonversi Data Transaksi

```python
# Memproses data transaksi menjadi format list
data = list(df["products"].apply(lambda x:x.split(",")))

# Menampilkan hasil
print(data)
```

📌 **Penjelasan:**
- Menggunakan `.apply(lambda x:x.split(","))` untuk **memisahkan item dalam satu transaksi**.

### **Output Contoh List Transaksi**
```
[['MILK', 'BREAD', 'BISCUIT'],
 ['BREAD', 'MILK', 'BISCUIT', 'CORNFLAKES'],
 ['BREAD', 'TEA', 'BOURNVITA'],
 ['JAM', 'MAGGI', 'BREAD', 'MILK'],
 ['MAGGI', 'TEA', 'BISCUIT']]
```

---

## 🔄 4. Encoding Data dengan Transaction Encoder

```python
from mlxtend.preprocessing import TransactionEncoder

# Menggunakan TransactionEncoder untuk mengubah data ke bentuk binary matrix
encoder = TransactionEncoder()
encoded_data = encoder.fit(data).transform(data)

# Mengonversi menjadi DataFrame Pandas
df_encoded = pd.DataFrame(encoded_data, columns=encoder.columns_)
df_encoded.replace(False, 0, inplace=True)

# Menampilkan hasil encoding
print(df_encoded.head())
```

📌 **Penjelasan:**
- **TransactionEncoder** digunakan untuk mengubah data transaksi menjadi **format matrix biner**.
- `False` diganti dengan `0` untuk kejelasan dalam analisis.

---

## 📊 5. Menjalankan Algoritma Apriori

```python
# Menjalankan algoritma apriori
df_freq = apriori(df_encoded, min_support=0.2, use_colnames=True, verbose=1)

# Menampilkan frequent itemsets
print(df_freq)
```

📌 **Penjelasan:**
- `min_support=0.2` → Menentukan bahwa hanya item yang muncul di **≥20% transaksi** yang akan diproses.

### **Output Contoh Frequent Itemsets**
```
   support       itemsets
0     0.35       (BISCUIT)
1     0.20      (BOURNVITA)
2     0.20          (BREAD)
3     0.30         (COFFEE)
4     0.30         (CORNFLAKES)
5     0.25         (MAGGI)
6     0.25          (MILK)
7     0.35          (SUGAR)
8     0.30           (TEA)
```

---

## 📊 6. Membangun Aturan Asosiasi

```python
# Menerapkan association rules dengan confidence ≥ 0.6
df_ar = association_rules(df_freq, metric="confidence", min_threshold=0.6)

# Menampilkan aturan asosiasi
print(df_ar)
```

📌 **Penjelasan:**
- `metric="confidence"` → Menggunakan **confidence** sebagai metrik untuk menilai aturan.
- `min_threshold=0.6` → Hanya aturan dengan confidence **≥ 60%** yang ditampilkan.

### **Output Contoh Association Rules**
```
  antecedents  consequents  antecedent support  consequent support  support  confidence  lift  leverage  conviction
0     (MILK)      (BREAD)               0.25                0.35    0.20        0.80   1.23   0.075       1.15
1    (SUGAR)      (BREAD)               0.30                0.35    0.25        0.83   1.25   0.08        1.20
2  (CORNFLAKES)  (COFFEE)               0.30                0.30    0.20        0.66   1.10   0.03        1.00
3   (SUGAR)      (COFFEE)               0.35                0.30    0.20        0.66   1.10   0.03        1.00
4    (MAGGI)      (TEA)                 0.25                0.30    0.20        0.80   1.28   0.08        1.22
```

---

## 🔍 7. Analisis Hasil

Dari hasil aturan asosiasi, dapat disimpulkan bahwa:

- **Pembeli yang membeli SUGAR memiliki kemungkinan tinggi untuk membeli BREAD**.
- **Pelanggan yang membeli CORNFLAKES sering membeli COFFEE bersama-sama**.
- **Jika seseorang membeli MAGGI, mereka juga cenderung membeli TEA**.

---

## 🎯 Kesimpulan

- **Apriori Algorithm berhasil menemukan pola pembelian dalam transaksi supermarket**.
- **BREAD dan SUGAR sering dibeli bersama, menunjukkan hubungan yang kuat**.
- **Aturan asosiasi dapat digunakan untuk meningkatkan strategi pemasaran** (misalnya, bundling produk).

---

## 📌 Referensi
- [Scikit-learn Apriori Documentation](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/)
 
