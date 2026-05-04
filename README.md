# 🚀 YOLOv8 + Re3: Hibrit Nesne Takip ve Hata Düzeltme Sistemi
Bu proje, YOLOv8 nesne algılama modeli ile Re3 (Real-Time Recurrent Regression) takipçisini entegre eden, hata payı minimize edilmiş bir takip sistemidir. Sistem, takipçinin nesneyi kaybetme ihtimaline karşı sürekli doğrulama (validation) yaparak kendini günceller.

## ✨ Öne Çıkan Özellikler
Hibrit Mimari: YOLOv8'in kesin tespiti ile Re3'ün hızlı takibi birleştirilmiştir.
Otomatik Hata Düzeltme: IoU ve Öklid mesafesi metrikleri kullanılarak takip kaybı algılanır ve sistem otomatik olarak yeniden başlatılır.
Dinamik Kontrast İyileştirme: CLAHE algoritması ile zorlu ışık koşullarında takip kararlılığı artırılmıştır.
Görsel Metrikler: IoU skoru ve merkez uzaklığı canlı olarak ekran üzerine yansıtılır.

## 🚀 Kurulum ve Kullanım

1. Gerekli kütüphaneleri yükleyin:

```bash
pip install ultralytics torch opencv-python numpy
```

