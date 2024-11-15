# Anime Recommendation System

## Proyek Overview

Sistem rekomendasi memainkan peran kunci dalam meningkatkan kepuasan pengguna. Sistem rekomendasi tidak hanya meningkatkan kepuasan pengguna melalui keakuratan dan keragaman rekomendasi, tetapi juga dipengaruhi oleh mekanisme psikologis yang melibatkan perasaan pengguna, dan tujuan mereka. Ketika tujuan pengguna dalam menggunakan suatu platform seperti e-commerce atau aplikasis streaming sejalan dengan hasil rekomendasi, terutama dalam hal keakuratan dan keragaman, kepuasan pengguna meningkat. Sebaliknya, pengguna yang eksploratif dapat merasa berkurangnya kepuasan saat menghadapi rekomendasi yang terlalu akurat karena aktivasi resistansi psikologis.

Selain itu sistem rekomendasi membantu pengambilan keputusan berbasis pengalaman di e-commerce, terutama melalui penerapan collaborative filtering menggunakan neural network. Studi ini menunjukkan bahwa sistem rekomendasi yang baik tidak hanya meningkatkan akurasi prediksi, tetapi juga memperkaya pengalaman pengguna dalam membuat keputusan online, seperti dalam memilih film.

Maka, proyek ini penting untuk diselesaikan karena dengan mengembangkan sistem rekomendasi yang lebih canggih dan efektif, dapat memberikan pengalaman yang lebih personal dan memuaskan bagi pengguna.

Referensi:

- X. He, Q. Liu, and S. Jung, "The Impact of Recommendation System on User Satisfaction: A Moderated Mediation Approach," J. Theor. Appl. Electron. Commer. Res., vol. 19, no. 1, pp. 448-466, Feb. 2024. <https://doi.org/10.3390/jtaer19010024>
- A. J. Lin, C. L. Hsu, and E. Y. Li, "Improving the effectiveness of experiential decisions by recommendation systems," Expert Syst. Appl., vol. 41, no. 10, pp. 4904-4914, Aug. 2014. <https://doi.org/10.1016/j.eswa.2014.01.035>

## Busines Understanding

Proyek ini bertujuan untuk membangun model Recommender System yang dapat untuk platform streaming anime.

### Problem Statements

1. Bagaimana cara menggunakan fitur konten seperti genre untuk meningkatkan akurasi rekomendasi?
2. Bagaimana cara memberikan rekomendasi anime yang relevan berdasarkan preferensi pengguna lain?

### Goals

1. Menggunakan fitur konten anime seperti genre, untuk membuat sistem rekomendasi dengan pendekatan Content-Based Filtering.
2. Membangun sistem rekomendasi berbasis Collaborative Filtering untuk memberikan rekomendasi yang relevan.

### Solution Statements

1. Menggunakan pendekatan Content-Based Filtering dengan menggunakan informasi genre untuk memberikan rekomendasi anime yang memiliki karakteristik genre yang mirip. Metode yang digunakan pada pendekatan ini adalah menggunakan TF-IDF untuk mentransformasi genre anime menjadi representasi numerik, kemudian menggunakan cosine similarity untuk mengukur kemiripan anime.
2. Menggunakan pendekatan Collaborative berdasarkan data interaksi penggunauntuk memberikan rekomendasi berdasarkan preferensi pengguna lain yang memiliki kesamaan. Metode yang digunakan adalah menggunakan model deep learning, yaitu Neural Collaborative Filtering.

## Data Understanding

Data yang digunakan dalam proyek ini adalah [Anime Recommendation Database 2020](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020) yang berisi informasi 17.563 daftar anime, dan preferensi dari 325.772 user yang berbeda. Dataset yang digunakan pada proyek ini adalah anime.csv yang akan disebut sebagai data **anime**, dan rating_complete.csv yang berisi data pengguna akan disebut sebagai data **user**.

### Variabel-variabel pada data Anime Recommendation Database 2020

**anime.csv**:

- MAL_ID: ID anime dari MyAnimeList
- Name: Nama lengkap anime
- Score: Skor rata-rata yang diberikan kepada anime dari seluruh pengguna di database MyAnimeList
- Genres: Daftar genre anime, dipisahkan dengan koma (contoh: Action, Adventure, Comedy, Drama, Sci-Fi, Space).
- English name: Nama lengkap anime dalam bahasa Inggris
- Japanese name: Nama lengkap anime dalam bahasa Jepang
- Type: Jenis anime, seperti TV, movie, OVA, dan lain sebagainya.
- Episodes: Jumlah episode dari anime tersebut.
- Aired: Tanggal siaran anime.
- Premiered: Musim tayang perdana.
- Producers: Daftar produser anime.
- Licensors: Daftar pemberi lisensi anime.
- Studios: Daftar studio yang memproduksi anime.
- Source: Sumber asli cerita anime, misalnya Manga, Light novel, dan lain sebagainya.
- Duration: Durasi tiap episode anime, atau movie anime.
- Rating: Klasifikasi usia anime.
- Ranked: Peringkat berdasarkan skor rata-rata
- Popularity: Peringkat berdasarkan jumlah pengguna yang menambahkan anime ke daftar mereka.
- Members: Jumlah anggota komunitas yang ada dalam grup anime tersebut.
- Favorites: Jumlah pengguna yang menandai anime sebagai "favorit".
- Watching: Jumlah pengguna yang sedang menonton anime.
- Completed: Jumlah pengguna yang telah menyelesaikan menonton anime.
- On-Hold: Jumlah pengguna yang menunda menonton anime.
- Dropped: Jumlah pengguna yang berhenti menonton anime.
- Plan to Watch: Jumlah pengguna yang berencana menonton anime.
- Score-1 - Score-10: Jumlah pengguna yang memberikan skor 1 sampai 10.

**rating_complete.csv**:

- user_id: ID pengguna.
- anime_id: ID anime dari dataset anime.csv yang telah dinilai oleh pengguna.
- rating: Rating atau skor yang diberikan oleh pengguna untuk anime tersebut.

### Explanatory Data Analaysis

#### Top 10 Anime Terpopuler

![Top 10 Anime Terpopuler](https://github.com/Mnjar/anime-recommendations-system/blob/main/images/Top-10%20Popularity.png?raw=true)

Dari hasil di atas, anime Death Note menempati posisi pertama sebagai anime terpopuler, diikitu Shingeki no Kyojin di posisi kedua, dan Fullmetal Alchemist: Brotherhood di posisi ketiga.

#### Top 10 Anime Terfavorit

![Top 10 Anime Terfavorit](https://github.com/Mnjar/anime-recommendations-system/blob/main/images/Top-10%20Anime%20Terfavorit.png?raw=true)

Sedangkan, untuk anime terfavorit di posisi pertama adalah Fullmetal Alchemist: Brotherhood, diikuti Steins;Gate, lalu Hunter x Hunter (2020).

#### Distribusi Genre

![Distribusi Genre](https://github.com/Mnjar/anime-recommendations-system/blob/main/images/Distribusi%20Genre.png?raw=true)

Genre paling banyak pada data ini adalah Comedy yang mencapai 6000 anime, diikuti genre Action sebanyak sekitar 3900, dan Fantasy sebanyak sekitar 3500

#### Distribusi Rating Pengguna

![Distribusi Rating Pengguna](https://github.com/Mnjar/anime-recommendations-system/blob/main/images/Distribusi%20Rating%20Pengguna.png?raw=true)

Distribusi rating yang diberikan berada pada rantang 5 hingga 10. Hanya sedikit pengguna yang memberikan rating dibawah 5.

## Data Preparation

Pada bagian ini, tahapan data preparation dibagi menjadi 2 bagian, yang pertama adalah Data Cleaning sebelum masuk ke pemodelan Content-Based Filtering, kedua adalah data preparation untuk pemodelan Neural Collaborative Filtering.

### 1. Data Cleaning

#### 1.1 Mengubah Tipe Data Yang Tidak Sesuai

Kolom Ranked, Episodes, dan kolom Score-1 sampai Score-10 saat ini bertipe data object. Kita akan mengubah tipe datanya menjadi integer. Selama proses ini, nilai yang tidak valid, seperti Unknown, akan diganti dengan 0.

Untuk kolom Score, kita akan menghitung rata-rata dari nilai di kolom Score-1 sampai Score-10. Nilai Unknown di kolom-kolom tersebut akan diubah menjadi 0, dan kolom Score akan diubah menjadi float karena ini merupakan kolom yang berisi rata-rata. Rata-rata skor untuk kolom Score akan dihitung dengan menggunakan rumus berikut:

$$\text{Score} = \frac{1 \cdot \text{Score-1} + 2 \cdot \text{Score-2} + \ldots + 10 \cdot \text{Score-10}}{\text{Score-1} + \text{Score-2} + \ldots + \text{Score-10}}$$

Ini dilakukan agar data siap digunakan pada tahap pemodelan nanti.

#### 1.2 Cek Missing Values

Melakukan pengecekan missing value. Agar pada tahap pemodelan tidak terjadi error, dan model dapat belajar dengan baik. Karena jika terdapat nilai NaN, ini berarti model kehilangan informasi dan dapat mempengaruhi kinerja model.

Pengecekan dilakukan menggunakan fungsi `isnull()` dan `sum()` dari libary pandas seperti berikut

```bash
print("Missing pada data anime:")
print(anime_data.isnull().sum())
```

```text
Missing pada data anime:
MAL_ID           0
Name             0
Score            0
Genres           0
English name     0
Japanese name    0
Type             0
Episodes         0
Aired            0
Premiered        0
Producers        0
Licensors        0
Studios          0
Source           0
Duration         0
Rating           0
Ranked           0
Popularity       0
Members          0
Favorites        0
Watching         0
Completed        0
On-Hold          0
Dropped          0
Plan to Watch    0
Score-10         0
Score-9          0
Score-8          0
Score-7          0
Score-6          0
Score-5          0
Score-4          0
Score-3          0
Score-2          0
Score-1          0
dtype: int64
```

```bash
print("Missing values pada data user:")
print(users_data.isnull().sum())
```

```text
Missing values pada data user:
user_id     0
anime_id    0
rating      0
dtype: int64
```

Setelah dilakukan pengecekan, tidak terdapat missing value pada data anime dan user.

#### 1.3 Cek Duplicate Values

Pengecekan duplicate values dilakukan untuk memastikan tidak ada terdapat nilai duplikat pada data. Karena hal tersebut hanya memperbesar dimensi data tanpa memberikan informasi yang berarti.

Pengecekan dilakukan menggunakan fungsi `duplicated()` dan `sum()` dari library pandas seperti berikut:

```bash
print("Duplicate values pada data anime:")
print(anime_data.duplicated().sum())
```

```text
Duplicate values pada data anime:
0
```

```bash
print("Duplicate values pada data user:")
print(users_data.duplicated().sum())
```

```text
Duplicate values pada data user:
0
```

Dari kedua output di atas, tidak terdapat data duplikat pada data **anime** dan **user**.

### 2. Data Preparation Untuk Pemodelan Neural Collaborative Filtering

#### 2.1 Pengambilan Sampel Data

Mengingat data user berjumlah 57.633.278 baris, dan karena keterbatasan sumber daya, data user yang digunakan pada model Neural Collaborative Filtering (NCF) hanya sebanyak 100.000 saja. Ini cukup untuk melatih model NCF untuk menghasilkan rekomendasi yang baik.

Proses pengambilan sample-nya sendiri dapat dilihat pada snippet code berikut:

```bash
users_data_subset = users_data.sample(n=100000, random_state=9)
```

#### 2.2 Encoding user_id dan anime_id

Encoding dilakukan karena model NCF membutuhkan rentang ID yang konsisten dan terurut untuk menghasilkan embedding yang efektif. Karena, bisa saja ID memiliki angka yang sangat besar dan tidak terurut. Jika langsung menggunakan ID asli yang mungkin memiliki nilai yang sangat besar, ini akan menciptakan embedding matrix yang sangat besar dan sebagian besar dari matrix tersebut tidak akan digunakan, sehingga menghabiskan banyak memori.

Proses dari encoding adalah sebagai berikut:

- Mengambil daftar unik dari user_id dan anime_id, untuk memastikan bahwa setiap user dan anime hanya muncul sekali dalam list.
- Membuat dictionary untuk user dan anime. Setiap user dan anime diberi indeks unik (integer mulai dari 0). Ini memastikan bahwa semua ID diubah menjadi nilai yang kecil, yang lebih mudah diproses oleh model.
- Mapping kembali ke dataframe menggunakan hasil dictionary, dengan membuat kolom baru yaitu **user_encoded** dan **anime_encoded** di dalam dataset user.

#### 2.3 Normalisasi Rating Ke Rentang 0-1

Selanjutnya adalah melakukan Normalisasi pada kolom rating. Normalisasi rating diperlukan untuk menyesuaikan skala nilai yang digunakan dalam model. Normalisasi membantu model neural network untuk lebih cepat dalam mencapai konvergensi selama pelatihan karena distribusi nilai yang lebih seragam. Jika rating tidak dinormalisasi, gradien yang dihasilkan bisa terlalu besar atau terlalu kecil, yang memperlambat proses pelatihan. Selain itu, banyak fungsi aktivasi seperti sigmoid  bekerja lebih baik dengan input yang dinormalisasi dalam rentang tertentu, biasanya 0-1 untuk sigmoid.

Proses normalisasi dilakukan sebagai berikut:

- Dapatkan nilai minimum (min_rating) dan maksimum (max_rating) dari data rating, kita menentukan batas rentang nilai rating.
- Aplikasikan normalisasi menggunakan rumus berikut:

$$\text{NormalizedRating} = \frac{\text{rating} - \text{minRating}}{\text{maxRating} - \text{minRating}}$$

#### 2.4 Split Data menjadi Data Training (70%), Data Validation (20%) dan Data Test (10%)

Setelah melakukan encoding dan normalisasi, langkah selanjutnya yaitu melakukan splitting dataset. Splitting dataset diperlukan untuk memastikan bahwa model dievaluasi secara objektif dan tidak overfitting, yaitu terlalu fokus pada pola data training. Dengan memisahkan dataset menjadi data train, val, dan test, performa model dapat diukur pada data yang belum pernah dilihat, yang penting untuk memperkirakan kinerja model sebenarnya.

## Modeling

Untuk menyelesaikan permasalahan, model yang digunakan pada kasus ini adalah Content-Based Filtering dengan Cosine Similarity, Dan Collaborative Filtering menggunakan Neural Network.

### 1. Content-Based Filtering Dengan Cosine Similarity

Content-Based Filtering dengan Cosine Similarity bekerja dengan merekomendasikan item berdasarkan fitur-fitur yang dimiliki item itu sendiri. Dalam konteks ini, model melihat kesamaan antara anime berdasarkan genre mereka. Algoritma ini menggunakan **TF-IDF Vectorization** untuk mengubah teks genre pada setiap anime menjadi vektor numerik yang mewakili bobot setiap genre dalam konteks anime tersebut. Setelah vektor TF-IDF dihasilkan, selanjutnya adalah menghitung kemiripan antar anime berdasarkan genre menggunakan cosine similarity. Ini mengukur sudut antara dua vektor genre, dengan hasil kemiripan antara 0 (tidak mirip) dan 1 (sangat mirip).

Berikut **Top-N recommendations** berdasarkan anime_id=20 (naruto) yang dihasilkan Content-Based Filtering dengan Cosine Similarity:

![Top-N recommendations](https://github.com/Mnjar/anime-recommendations-system/blob/main/images/Top-10%20CBF.png?raw=true)

#### 1.2 Kelebihan

- Mudah dipahami karena model mendasarkan rekomendasinya pada fitur-fitur yang eksplisit (seperti genre).
- Mampu memberikan rekomendasi berdasarkan preferensi unik pengguna.
- Cold-Start Problem: Content-based filtering mengatasi masalah "cold start" pada item baru karena rekomendasi didasarkan pada fitur item, bukan perilaku pengguna.

#### 1.3 Kekurangan

- Model hanya merekomendasikan anime yang sangat mirip dengan anime yang sudah diketahui, sehingga dapat kehilangan keragaman rekomendasi.
- Fitur yang terbatas. Artinya, kualitas rekomendasi sangat tergantung pada seberapa baik fitur (genre) mewakili keseluruhan kualitas anime. Jika hanya menggunakan genre, aspek lain seperti gaya visual atau kualitas produksi tidak diperhitungkan.
- Rekomendasi yang terbatas, karena tidak bisa memberikan rekomendasi anime dengan genre yang berbeda dari yang sudah diketahui.

### 2. Collaborative Filtering dengan Neural Network

Pada proyek ini, Collaborative Filtering dengan Neural Network menggunakan embedding untuk merepresentasikan pengguna dan item (anime) dalam vektor berdimensi rendah. Setiap user dan item (anime) direpresentasikan sebagai vektor embedding. Embedding ini adalah representasi berdimensi rendah yang dihasilkan oleh lapisan embedding. Setelah mendapatkan embedding user dan item, keduanya digabungkan menjadi satu vektor. Vektor gabungan ini kemudian dilewatkan melalui beberapa lapisan fully connected (MLP) untuk memodelkan interaksi kompleks antara user dan item. Model juga menghitung dot product antara vektor embedding user dan item sebagai representasi interaksi linier antara mereka.

Berikut contoh **Top-N recommendations** yang dihasilkan Collaborative Filtering dengan Neural Network:

![Top-N recommendations](https://github.com/Mnjar/anime-recommendations-system/blob/main/images/Top-10%20CF.png?raw=true)

#### 2.1 Kelebihan

- Neural Collaborative Filtering (NCF) lebih fleksibel dibandingkan pendekatan berbasis matriks faktorisasi standar karena mampu memodelkan interaksi non-linier antar user dan item melalui MLP.
- Kemampuan Generalisasi yang baik. Dengan menggunakan embedding dan jaringan saraf, model ini lebih mampu menangkap pola tersembunyi yang kompleks, yang mungkin terlewatkan oleh metode faktorisasi matriks tradisional.
- Model ini dapat menangani masalah sparsity pada data rating yang sering terjadi pada Collaborative Filtering, karena tidak sepenuhnya bergantung pada hubungan langsung antar item.

#### 2.2 Kekurangan

- Neural network memerlukan tuning hyperparameter yang lebih rumit dan sumber daya komputasi yang lebih besar dibandingkan metode tradisional.
- Jika tidak dilengkapi dengan metode regularisasi yang tepat seperti dropout, model ini rentan terhadap overfitting terutama jika jumlah data training terbatas.
- Karena lebih kompleks, model ini memerlukan waktu pelatihan yang lebih lama dibandingkan metode simpler seperti faktorisasi matriks.

## Evaluasi

### 1. Metrik evaluasi Content-Based Filtering Dengan Cosine Similarity

Model Content-Based Filtering dievaluasi menggunakan metrik precision yang digunakan untuk mengukur kualitas rekomendasi.

**Formula**:

$$\text{Precission} = \frac{\text{Jumlah rekomendasi yang relevan}}{\text{Jumlah total rekomendasi}}$$

Precision mengukur berapa banyak rekomendasi yang diberikan oleh model yang relevan (misalnya, anime dengan skor di atas 6.0 dianggap relevan). Formula precision menghitung persentase dari rekomendasi yang memang relevan dengan preferensi pengguna, atau dalam konteks ini, rekomendasi yang memiliki skor rating di atas ambang batas.

### 2. Metrik evaluasi Collaborative Filtering Dengan Neural Network

Model Collaborative Filtering Dengan Neural Network dievaluasi menggunakan RMSE (Root Mean Squared Error). RMSE mengukur seberapa jauh prediksi model dari nilai aktualnya.

**Formula**:

![RMSE](https://github.com/Mnjar/anime-recommendations-system/blob/main/images/RMSE.png?raw=true)

RMSE mengukur perbedaan rata-rata antara prediksi dan observasi aktual. Semakin kecil nilai RMSE, semakin baik model dalam memprediksi nilai rating. Tetapi, RMSE sensitif terhadap outlier, karena kesalahan besar dihitung secara kuadrat, yang artinya error besar memiliki pengaruh yang lebih signifikan dibandingkan error kecil.
