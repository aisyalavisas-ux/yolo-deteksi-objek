DATASET DETEKSI OBJEK — FORMAT YOLO

Sumber: COCO val2017
Situs sumber: https://cocodataset.org/
Kelas: 0=person, 1=car, 2=bottle
Jumlah gambar: 220
Gambar berobjek: 200
Gambar latar: 20

train: 154 gambar, 13 latar, kotak={'person': 515, 'car': 120, 'bottle': 37}
val: 44 gambar, 6 latar, kotak={'car': 38, 'person': 203, 'bottle': 2}
test: 22 gambar, 1 latar, kotak={'person': 90, 'bottle': 4, 'car': 15}

Jumlah kotak keseluruhan: {'person': 808, 'car': 173, 'bottle': 43}

Format label: class_id cx cy width height
Koordinat dinormalisasi ke rentang 0–1.
Label kosong menandakan gambar latar.
Anotasi crowd tidak dikonversi, mengikuti skrip tugas.
Jumlah objek antarkelas tidak dijamin seimbang.
Pemilihan gambar menggunakan seed 0; split menggunakan seed 42.
Periksa kotak secara visual sebelum pelatihan.
Lisensi gambar mengikuti masing-masing gambar sumber COCO.
