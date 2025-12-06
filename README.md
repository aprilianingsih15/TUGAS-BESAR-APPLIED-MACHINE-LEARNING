
LK.8 Perancangan Project Data Science 
Nama 		: Aprilianingsih
Tanggal 	: 06 Desember 2025
Kelas 		: 5A
Judul Project     : Klasifikasi kondisi Sekolah Dasar Berdasarkan Data Pendidikan Tahun 2024 Menggunakan Metode K- Nearest Neighbor dan Naïve Bayes

A.	Instruksi 
Peserta diminta untuk merancang sebuah proyek Data Science yang berfokus pada permasalahan di bidang pendidikan. Rancangan proyek ini harus disusun secara sistematis berdasarkan metodologi CRISP-DM (Cross Industry Standard Process for Data Mining) yang mencakup enam tahapan utama, yaitu:
1.	Business Understanding (Pemahaman Bisnis)
a.	Tujuan dan Fokus
Tujuan dari proyek ini adalah membangun sistem klasifikasi berbasis Machine Learning untuk mengelompokkan kondisi Sekolah Dasar beradasarkan data pendidikan tahun 2024. Fokus utama proyek adalah membantu analisis kondisi sekolah secara objektif melalui data, sehingga dapat digunakan sebagai pendukung pengambilan keputusan di bidang pendidikan.
b.	Langkahn dan Metode
Langkah – langkah yang dilakukan pada tahap ini meliputi:
•	Menentukan konteks pendidikan, yaitu sekolah Dasar (SD).
•	Mengidentifikasi permasalahan terkait kondisi sekolah berdasarkan data.
•	Menentukan tujuan proyek, yaitu klasifikasi kondisi sekolah.
•	Menentukan algoritma machine Learning yang digunakan, yaitu KNN dan Naïve Bayes.
c.	Jenis dan sumber data
Jenis data berupa data pendidikan Sekolah Dasar yang terdiri dari data numerik dan kategorikal. Sumber data berasal darai dataset “Gambaran umum Keadaan Sekolah Dasar Tiap Provinsi/Kabupaten Tahun 2024”.
d.	Hasil/Keluaran
Hasil yang diharapkan dari tahap ini adalah:
•	Rumusan masalah yang jelas.
•	Tujuan proyek yang terarah.
•	Penentuan metode klasifikasi yang akan digunakan.
2.	Data Understanding (Pemahaman Data)
a.	Tujuan dan Fokus Kegiatan
Tahap ini digunakan untuk memahami karakter dataset yang digunakan, baik dari segi struktur data, jenis atribut, maupun kualitas data.
b.	Langkah dan Metode
Langkah – langkah yang digunakan yaitu:
•	Menampilkan data awal menggunakan perintah df.head().
•	Menampilkan struktur data menggunakan perintah df.info().
•	Menampilkan statistik deskriptif menggunakan perintah df.describe().
•	Mengidentifikasi missing value pada dataset.

c.	Jenis dan Sumber Data
Data ini terdiri dari data numerik dan data kategorikal.
d.	Hasil dan Keluaran
Hasil dari tahap ini yaitu, informasi struktur dataset, identifikasi atribut numerik dan kategorikal, serta informasi awal terkaiyt kualitas dan kelengkapan data.
3.	Data Preparation (Persiapan Data)
a.	Tujuan dan fokus kegiatan
Tahap ini bertujuan untuk memersihkan da menyiapkan data agar layak digunakan dalam proses pelatihan model Machine Learning.
b.	Langkah dan Metode
Langkah – langkah yang dilakukan adalah:
•	Menangani missing value dengan mengganti nilai kosong menjadi ‘0.
•	Melakukan encoding fata kategorikal menggunakan metode label Encoding.
•	Menyimpan data hasil processing ke dalam file baru.
c.	Jenis dan Sumber Data
Data yang diginakan adalah data hasil pemahaman sebelumnya yang berasal dari dataset Sekolah Dasar tahun 2024.
d.	Hasil dan Keluaran
Hasil dari tahap ini yaitu, dataset yang telah bersih dan siap digunakan serta file hasil processing: hasil_preprocessing_sd_2024.csv
4.	Modeling (Pemodelan)
a.	Tujuan dan Fokus Kegiatan
Tahap pemodelan bertujuan untuk membangun model klasifikasi yang mampu mengelompokkan kondisi Sekolah berdasarka data pendidikan.
b.	Langkah dan Metode
Langkah – langkah yang dilakukan yaitu adalah:
•	Pembagian dataset menjadi data latih (80%) dan data uji (20%).
•	Penerapan algoritma K-Nearest Neighbor (KNN).
•	Penerapan algoritma Naïve Bayes.
•	Serta pelatihan menggunakan data latih.
c.	Jenis dan Sumber Data
Data yang digunakan adalah dataset hasil preprocessing.
d.	Hasil dan Keluaran
Hasil yang diperoleh dari tahap ini adalah yaitu, modek KNN yang telah dilatih dan model Naïve Bayes yang telah dilatih.
5.	Evaluation (Evaluasi)
a.	Tujuan dan Fokus Kegiatan
Tahap evaluasi bertujuan untuk mengukur performa model klasifikasi yang telah dibuat.
b.	Langkah dan Metode
Metode evaluasi yang digunakan yaitu, perhitungan nilai akurasi, analisis Confusion Matrix, dan analisis Classification Report.
c.	Jenis dan Sumber Data
Data yang digunakan adalah data uji (testing) hasil pembagian dataset.
d.	Hasil atau Keluaran
Hasil dari tahap ini yaitu adalah nilai akurasi KNN, nilai akurasi Naïve Bayes, Confusion Matrix, dan Classification Report.
6.	Deployment (Penerapan)
a.	Tujuan dan Fokus Kegiatan
Tahap deployment bertujuan untuk menerapkan model klasifikasi agar dapat digunakan oleh pengguna secara langsung.
b.	Langkah dan Metode
Langkah – langkah implementasinya yaitu:
•	Menghubungkan model Machine learning dengan aplikasi web.
•	Menggunakan Gradio sebagai antarmuka pengguna.
•	Menyediakan form input data sekolah.
•	Dan menampilkan hasil prediksi secara otomatis.
c.	Jenis dan Sumber Data
Data yang diunakan berasal dari data yang diinput oleh pengguna melalui aplikasi Gradio.
d.	Hasil dan Keluaran
Hasil dari tahap ini yaitu untuk aplikasi prediksi kondisi sekolah berbasis web serta sistem klasifikasi sekolah yang mudah digunakan oleh pengguna di bidang pendidikan.
B.	Format Perancangan 
Tahapan CRISP-DM	Instruksi untuk Peserta	Rancangan Implementasi
1. Business Understanding (Pemahaman Bisnis)	1.	Konteks pendidikan yag digunakan dalam proyek ini adalah Sekolah Dasar (SD).
2.	Permasalahan yang diangkat adalah belum adanya sistem klasifikasi berbasis data untuk mengetahui kondisi sekolah secara objektif berdasarkan data pendididkan. Selama ini penilaian kondisi sekolah masih dilakukan secara manual dan kurang optimal.
3.	Tujuan dari proyek ini adalah membangun model Machine Learning yang mampu mengklasifikasikan kondisi sekolah berdasarkan data Sekolah Dasar Tahun 2024, sehingga dapat membantu pihak terkait dalam pengambilan keputusan, khususnya dalam pemerataan dan peningkatan kualitas pendidikan.	
2. Data Understanding (Pemahaman Data)	1.	Sumber data yang digunakan berasal dari dataset terbuka pendidikan Sekolah Dasar Tahun 2024, yang berisi gambaran umum kondisi sekolah di tiap wilayah.
2.	Jenis data yang digunakan yaitu data numerik (seperti jumlah sekolah, jumlah siswa, jumlah guru, dan fasilitas pendidikan) dan data kategorikal (seperti wilayah(provinsi/kabupaten)).
3.	Fitur (atribut) yang digunakan adalah seluruh variabel pendukung kondisi sekolah, sedangkan target (label) adalah kolom terakhir yang digunakan sebagai kelas dalam proses klasifikasi kondisi sekolah.	
3. Data Preparation (Persiapan Data)	1.	Menangani nilai kosong (Missing Value) dengan mengganti nilai kosong menjadi 0.
2.	Melakukan encoding data kategorikal menggunakan metode label encoding, agar data dapat diproses oleh algoritma Machine Learning, memastikan seluruh data telah berbentuk numerik, dan menyimpan hasil preprocessing ke dalam file hasil_preprocessing_sd_2024.csv tahapan ini bertujuan agar data siap digunakan dalam proses pelatihan model.	
4. Modeling (Pemodelan)	1.	Algoritma yang digunakan dalam penelitian ini yaitu adalah K-Nearest Neighbor (KNN) dan Naïve Bayes.
2.	Alasan pemilihan kedua algoritma ini karena algortma  KNN mampu melakukan klasifikasi berdasarkan jarak terdekat antar data sedangkan Naïve Bayes digunakan karena memiliki konsep probabilistik yang sederhana dan efektif untuk klasifikasi data. Dataset dibagi menjadi  2 yaitu 80% data training dan 20% data testing.	
5. Evaluation (Evaluasi)	Metode evaluasi yang digunakan yaitu, Accuracy Score, Confusion Matrix, dan Classification Report. Evaluasi ini dilakukan untuk mengukur tingkat keberhasilan model dalam melakukan klasifikasi kondisi sekolah. Model KNN dan Naïve Bayes dibandingkan berdasarkan nilai akurasinya untuk mengetahui model mana yang memiliki performa terbaik.
	
6. Deployment (Penerapan / Implementasi)	Model yang telah dilatih diimplementasikan dalam bentuk notebook interaktif menggunakan Google Colab.	
