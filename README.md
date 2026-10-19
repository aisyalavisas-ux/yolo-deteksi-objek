# Praktik Terpadu Minggu 4–7 — Deteksi Objek dengan YOLO (RET503)

| | |
|---|---|
| NIM | 4222401086 |
| Nama | Ais Ya Lavisa Sahra |
| Tanggal pelatihan | 19-10-2026 |
| Lingkungan | Google Colab, GPU T4 (host `27631c81cf8f`, user `root`) |
| Sidik jari label dataset | `0901297D1BF3` |

> Seluruh angka pada dokumen ini berasal dari keluaran asli `06_latih_bandingkan.py`, `yolo val`, dan kurva `results.png` kedua eksperimen.

---

## 1. Data proyek (Tahap 1–2)

### 1.1 Sumber dan kelas

- Sumber: COCO val2017 (https://cocodataset.org/), diambil dengan `00_ambil_subset_coco.py` (seed 0).
- Kelas: `0 = person`, `1 = car`, `2 = bottle`.
- Total 220 gambar: 200 gambar berobjek + 20 gambar latar (label kosong, agar model belajar membedakan objek dan latar).
- Format label YOLO: `class_id cx cy w h`, dinormalisasi 0–1. Anotasi `iscrowd` tidak dikonversi.

### 1.2 Pembagian data (70/20/10, seed 42)

| Split | Gambar | Latar | person | car | bottle | Total kotak |
|---|---:|---:|---:|---:|---:|---:|
| train | 154 | 13 | 515 | 120 | 37 | 672 |
| val | 44 | 6 | 203 | 38 | 2 | 243 |
| test | 22 | 1 | 90 | 15 | 4 | 109 |
| **Total** | **220** | **20** | **808** | **173** | **43** | **1.024** |

Grafik: `Tahap1-2_Data_Projek/distribusi_kelas.png`

### 1.3 Pemeriksaan kualitas anotasi (`01_cek_anotasi.py`)

- Tidak ada file label hilang, baris label rusak, `class_id` salah, nilai di luar 0–1, kotak keluar batas, maupun gambar duplikat.
- Skrip memberi 103 peringatan; semuanya berjenis "kotak sangat kecil" (sisi < 8 px), yaitu 103 dari 1.024 kotak (10,1%): person 77, car 24, bottle 2. Ini objek kecil/jauh bawaan COCO, bukan kesalahan anotasi, sehingga dipertahankan.
- Pemeriksaan visual: `Tahap1-2_Data_Projek/contoh_cek_visual.jpg`. Kotak berada tepat pada objek.
- Laporan lengkap: `Tahap1-2_Data_Projek/laporan_cek_anotasi.txt`.

### 1.4 Konversi format

Uji YOLO → COCO (`02_yolo_ke_coco.py`) → YOLO (`03_coco_ke_yolo.py`) menghasilkan 220 file label yang identik dengan aslinya (selisih maksimum 0,000001). Hasil COCO JSON: `Tahap1-2_Data_Projek/annotations_coco.json` (220 gambar, 1.024 anotasi, 3 kategori).

### 1.5 Keterbatasan data

- Kelas tidak seimbang: person 79% dari seluruh kotak, bottle hanya 4%.
- Kelas `bottle` hanya punya **2 kotak di val** dan **4 kotak di test**. AP kelas ini sangat tidak stabil, dan karena mAP adalah rata-rata antarkelas, satu kelas dengan 2 contoh tetap berbobot 1/3.
- Val hanya 44 gambar dan test 22 gambar, sehingga selisih kecil antar model bisa jadi hanya noise.

---

## 2. Eksperimen (Tahap 3)

Pengaturan sama untuk semua eksperimen: model **yolo26n**, epochs 30, imgsz 416, batch 16, seed 0, `deterministic=True`.

| Kode | Deskripsi |
|---|---|
| E1 | Pretrained COCO (transfer learning), baseline |
| E2 | Arsitektur sama, bobot acak (tanpa transfer), pembanding |
| E3 | E1 + **satu** perubahan: `imgsz=640` (dari 416) |

### 2.1 Hasil pada data val (`hasil_praktik_4222401086.csv`)

| Eksperimen | Perubahan | mAP50 | mAP50-95 | Precision | Recall | Inferensi (ms) | Waktu latih (menit) |
|---|---|---:|---:|---:|---:|---:|---:|
| E1 | - | 0,630 | 0,426 | 0,637 | 0,600 | 9,0 | 1,7 |
| E2 | - | 0,004 | 0,001 | 0,346 | 0,030 | 4,1 | 1,4 |
| E3 | imgsz=640 | 0,642 | 0,499 | 0,751 | 0,609 | 8,6 | 1,8 |

Catatan: nilai E1 berasal dari `best.pt` yang kemungkinan besar dari epoch 1 (lihat Bagian 4.3).

Selisih E3 terhadap E1: mAP50 **+0,012**, mAP50-95 **+0,073** (naik 17% secara relatif), Precision **+0,114**, Recall **+0,009**.

### 2.2 Kurva latih

- E1: `hasil_praktik/E1_pretrained/results.png` (dan `results.csv`)
- E3: `hasil_praktik/E3_tuning/results.png` (dan `results.csv`)

---

## 3. Pemilihan model dan hasil test (Tahap 4–6)

- Kriteria pemilihan: mAP50-95 tertinggi pada data **val**.
- Model terpilih: **E3_tuning** (mAP50-95 val 0,499 dibanding 0,426 pada E1 dan 0,001 pada E2).
- Hasil pada data **test** (dipakai sekali saja): **mAP50 0,390**, **mAP50-95 0,265**.

| Data | mAP50 | mAP50-95 |
|---|---:|---:|
| val (dasar pemilihan) | 0,642 | 0,499 |
| test (22 gambar) | 0,390 | 0,265 |

Laporan otomatis: `hasil_praktik_4222401086.txt` dan `hasil_praktik_4222401086.csv`.

---

## 4. Analisis

**4.1 Pretrained vs scratch (E1 vs E2).**
E2 hampir tidak mendeteksi apa pun: mAP50 0,004, mAP50-95 0,001, dan recall hanya 0,030. Precision 0,346 tidak bermakna karena dihitung dari sangat sedikit prediksi (recall 3%). Penyebabnya, bobot acak harus mempelajari fitur visual dari nol, sedangkan 30 epoch pada 154 gambar latih jauh dari cukup. E1 memulai dari bobot hasil pelatihan COCO sehingga langsung mencapai mAP50 0,630 dengan waktu latih yang hampir sama (1,7 vs 1,4 menit). Perlu dicatat bahwa person, car, dan bottle sudah termasuk kelas COCO, sehingga keuntungan transfer learning di sini lebih besar daripada untuk kelas yang benar-benar baru.

**4.2 Pengaruh tuning (E1 vs E3).**
Menaikkan imgsz dari 416 ke 640 menaikkan mAP50-95 val dari 0,426 menjadi 0,499 (+0,073), mAP50 dari 0,630 menjadi 0,642 (+0,012), precision dari 0,637 menjadi 0,751 (+0,114), dan recall dari 0,600 menjadi 0,609 (+0,009). Metrik per kelas (Bagian 4.4) menunjukkan dari mana kenaikan mAP50-95 itu berasal: sekitar 58% dari car (+0,126), 34% dari bottle (+0,074), dan hanya 8% dari person (+0,017). Person adalah kelas terbesar (203 dari 243 kotak val), tetapi mAP adalah rata-rata antarkelas dengan bobot sama, sehingga kelas kecil ikut menentukan.

Kenaikan terbesar pada car sejalan dengan dugaan awal bahwa resolusi lebih tinggi membantu objek kecil: car adalah kelas dengan proporsi objek kecil tertinggi (52% kotak berluas < 32×32 piksel), dan pada imgsz 416 objek mengecil menjadi sekitar 0,65 kali karena sisi terpanjang gambar umumnya 640 px. Namun beberapa hal membatasi kesimpulan ini:
1. Person juga banyak berisi objek kecil (31%), tetapi nyaris tidak membaik, dan mAP50-nya malah turun 0,021 (0,563 → 0,542). Jadi "resolusi tinggi membantu objek kecil" tidak berlaku merata.
2. Kenaikan precision keseluruhan (+0,114) hampir seluruhnya berasal dari bottle (0,589 → 1,000 pada hanya 2 kotak), sedangkan precision car justru turun (0,696 → 0,556). Kenaikan precision tidak bisa dibaca sebagai peningkatan umum.
3. Bobot terbaik E1 kemungkinan besar berasal dari epoch 1 dan model memburuk setelahnya (Bagian 4.3), sementara E3 pulih. Jadi E1 vs E3 tidak hanya membandingkan resolusi, tetapi juga perilaku pelatihan yang berbeda, dan satu kali pelatihan (seed 0) tidak cukup untuk memisahkannya.
4. Val hanya 44 gambar dengan 38 kotak car dan 2 kotak bottle.

Meski begitu, E3 tetap lebih baik walaupun bottle dikeluarkan: rata-rata mAP50-95 person dan car adalah sekitar 0,23 pada E1 dan 0,30 pada E3. Biaya waktu latih kecil: 1,8 menit dibanding 1,7 menit. Waktu inferensi tidak dijadikan dasar perbandingan karena pengukurannya tidak konsisten: laporan mencatat 9,0 ms (E1), 4,1 ms (E2), dan 8,6 ms (E3), padahal E2 berarsitektur sama dengan E1, dan evaluasi ulang menghasilkan 8,1 ms (E1) dan 6,9 ms (E3). Angka-angka ini kemungkinan dipengaruhi pemanasan GPU dan jumlah gambar val yang kecil.

**4.3 Kurva latih (results.png).**

*E1* (`hasil_praktik/E1_pretrained/results.png`):
- Loss latih naik singkat pada epoch 1–5 lalu turun terus (box_loss sekitar 1,74 → 1,97 → 1,55; cls_loss sekitar 1,74 → 1,09).
- Loss val naik pada fase awal (box_loss sampai sekitar 2,12; cls_loss sampai sekitar 2,31 pada epoch 16) dan pada akhir pelatihan masih lebih tinggi daripada epoch 1 (cls_loss sekitar 2,09 vs 1,75; box_loss sekitar 1,86 vs 1,82).
- Metrik val **terbaik justru berada di epoch 1** (mAP50 sekitar 0,63; mAP50-95 sekitar 0,424), lalu bergerigi dan memburuk: mAP50-95 turun ke sekitar 0,20 pada epoch 8, naik ke sekitar 0,36 pada epoch 10–15, lalu turun lagi ke sekitar 0,19–0,20 pada epoch 25–30. mAP50 berakhir di sekitar 0,44 dan recall turun dari sekitar 0,59 ke sekitar 0,33.

Nilai E1 pada Bagian 2.1 (mAP50 0,630; mAP50-95 0,426) hampir identik dengan titik epoch 1 itu, sehingga `best.pt` E1 kemungkinan besar berasal dari epoch 1. Artinya 29 epoch berikutnya tidak memperbaiki model, bahkan memburukkannya: loss latih terus turun sementara metrik val memburuk. Ini pola yang khas untuk overfitting atau terganggunya fitur pretrained pada data latih sekecil 154 gambar. Dugaan penyebabnya, yang belum diuji, adalah laju belajar atau augmentasi bawaan yang terlalu agresif untuk fine-tuning pada data kecil.

*E3* (`hasil_praktik/E3_tuning/results.png`):
- Loss latih (box, cls, l1) naik singkat pada epoch 1–5, lalu turun terus sampai epoch 30 (box_loss sekitar 1,7 → 1,23; cls_loss sekitar 1,47 → 0,94) dan belum mendatar.
- Loss val naik pada fase awal (box_loss sampai sekitar 1,9; cls_loss sampai sekitar 2,4 pada epoch 14), lalu turun dan stabil di sekitar 1,72 (box) dan 1,75 (cls), meski masih lebih tinggi daripada epoch 1.
- Metrik val tidak stabil di paruh pertama (mAP50 turun dari sekitar 0,65 ke sekitar 0,40; mAP50-95 dari sekitar 0,47 ke sekitar 0,28; recall sampai sekitar 0,33), tetapi **pulih mulai sekitar epoch 20** dan mencapai mAP50 sekitar 0,647 serta mAP50-95 sekitar 0,50 pada epoch 28–30.

Berbeda dengan E1, E3 pulih dan menutup pelatihan pada nilai terbaiknya, sehingga tidak tampak overfitting pada akhir pelatihan. Pemulihan itu bertepatan dengan 10 epoch terakhir, saat Ultralytics secara bawaan menonaktifkan augmentasi mosaic (dugaan, tidak diuji). Mengapa E3 pulih sedangkan E1 tidak belum diketahui: bisa jadi imgsz 640 membuat pelatihan lebih stabil, atau hanya variasi antar pelatihan karena memakai satu seed.

*Apakah 30 epoch cukup?* Untuk E1, epoch tambahan justru merugikan; untuk E3, metrik val mendatar pada 3 epoch terakhir tetapi loss latih masih turun. Kedua kurva bergerigi karena val hanya 44 gambar, sehingga jumlah epoch yang ideal perlu diuji dengan percobaan lain.

**4.4 Kelas dan kesalahan.**
Metrik per kelas pada data val (`yolo val ... split=val`; E1 dengan imgsz 416, E3 dengan imgsz 640):

| Kelas | Kotak val | mAP50-95 E1 | mAP50-95 E3 | mAP50 E1 | mAP50 E3 | Recall E1 | Recall E3 | Precision E1 | Precision E3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| person | 203 | 0,301 | 0,318 | 0,563 | 0,542 | 0,536 | 0,493 | 0,626 | 0,697 |
| car | 38 | 0,157 | 0,283 | 0,332 | 0,390 | 0,263 | 0,368 | 0,696 | 0,556 |
| bottle | 2 | 0,821 | 0,895 | 0,995 | 0,995 | 1,000 | 0,965 | 0,589 | 1,000 |
| **all** | 243 | 0,426 | 0,499 | 0,630 | 0,642 | 0,600 | 0,609 | 0,637 | 0,751 |

Person dan car adalah kelas yang paling lemah pada kedua model. Car paling parah pada E1: recall hanya 0,263, artinya sekitar tiga perempat mobil tidak terdeteksi, dan mAP50-95 hanya 0,157. Pada E3 car membaik (recall 0,368; mAP50-95 0,283), sedangkan person nyaris tidak berubah (mAP50-95 0,301 → 0,318). Dugaan penyebabnya adalah banyaknya objek kecil atau jauh (di seluruh dataset, sekitar 31% kotak person dan 52% kotak car berluas kurang dari 32×32 piksel) serta contoh car yang sedikit (120 kotak di train).

Nilai bottle yang sangat tinggi (mAP50-95 0,821 dan 0,895) **tidak boleh dibaca sebagai "bottle mudah"**: val hanya berisi 2 kotak bottle pada 2 gambar, sehingga satu deteksi benar atau salah sudah mengubah nilainya drastis. Kelas ini menaikkan rata-rata: tanpa bottle, rata-rata person dan car pada E3 hanya 0,30 (mAP50-95) dan 0,47 (mAP50), bukan 0,499 dan 0,642.

**4.5 Val vs test.**
Model terpilih (E3) turun dari mAP50 0,642 (val) ke 0,390 (test), dan dari mAP50-95 0,499 ke 0,265, yaitu penurunan sekitar 47% pada mAP50-95. Beberapa penyebab yang masuk akal: (1) val dipakai untuk memilih bobot terbaik antar-epoch dan memilih antar-eksperimen, sedangkan kurva val bergerigi, sehingga mengambil nilai tertinggi cenderung optimistis; (2) rata-rata val ditopang kelas bottle (4.4), sedangkan rata-rata person dan car saja (0,47 dan 0,30) jauh lebih dekat ke hasil test; (3) test hanya 22 gambar dengan 109 kotak, termasuk 4 kotak bottle yang berbobot 1/3 pada rata-rata mAP, sehingga variansnya besar. AP per kelas pada test tidak dihitung karena data test hanya dipakai sekali, jadi sumbangan tiap penyebab tidak bisa dipisahkan dan kemiripan dengan angka person/car di atas hanyalah indikasi. Angka test (0,390 / 0,265) lebih jujur dipakai sebagai perkiraan kinerja pada data baru, tetapi ketidakpastiannya besar.

**4.6 Kesimpulan dan saran.**
Transfer learning sangat menentukan pada data kecil (E1 jauh di atas E2). Menaikkan imgsz ke 640 menaikkan mAP50-95 val dari 0,426 ke 0,499 dengan biaya waktu yang kecil, terutama lewat perbaikan pada car dan bottle, sedangkan person hampir tidak berubah. Kurva E1 menunjukkan fine-tuning dengan pengaturan bawaan tidak stabil: model terbaik E1 kemungkinan hanya satu epoch dan memburuk setelahnya. Model E3 dipilih dan mencapai mAP50 0,390 serta mAP50-95 0,265 pada test. Kinerja yang lebih realistis terbaca dari test dan dari kelas person/car (mAP50-95 sekitar 0,3), bukan dari rata-rata val yang ditopang bottle.

Saran perbaikan yang bisa dicoba berikutnya: (a) laju belajar lebih kecil atau membekukan sebagian layer awal (`lr0`, `freeze`) karena E1 memburuk setelah epoch 1; (b) menambah contoh car dan bottle serta memperbesar val/test agar evaluasi lebih stabil; (c) mengulang eksperimen dengan beberapa seed untuk memastikan selisih E1–E3 bukan kebetulan; (d) mencoba ukuran model yang lebih besar.

---

## 5. Cara mereproduksi (Google Colab, GPU T4)

```bash
pip install -U ultralytics
unzip -q YOLO.zip
unzip -q dataset_coco_3kelas_yolo.zip
python YOLO/YOLO/06_latih_bandingkan.py --nim 4222401086 --nama "Ais Ya Lavisa Sahra" --data dataset_v1/data.yaml --device 0
```

Dataset siap pakai berada di `dataset_v1/` (hasil `05_split_dataset_yolo.py`). Keluaran: `hasil_praktik_4222401086.txt`, `hasil_praktik_4222401086.csv`, dan folder `hasil_praktik/` berisi grafik tiap eksperimen.
