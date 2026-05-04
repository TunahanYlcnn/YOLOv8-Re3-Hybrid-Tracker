### Kodun Çalışma Mantığı:
1.  **Sistem-Başlat-Ayarları:** İlk karede YOLOv8 nesneleri tespit eder.
2.  **Takip-Döngüsü:** Her yeni karede Re3, nesnenin yeni konumunu tahmin eder.
3.  **Hata-Düzeltme:** Eğer tahmin ve tespit arasında büyük bir fark varsa, sistem kendini YOLOv8 verisine göre günceller.

# 🚀 YOLOv8 & Re3 Hybrid Tracker

Bu proje, yüksek performanslı **YOLOv8** dedektörü ile dayanıklı **Re3 (Recurrent Residual Regression)** takip algoritmasını birleştiren hibrit bir nesne takip sistemidir. Özellikle nesnelerin birbirini kapattığı (occlusion) veya dedektörün nesneyi kaçırdığı anlarda, Re3'ün geçmiş karelerdeki veriyi hatırlama yeteneği sayesinde kesintisiz bir takip deneyimi sunar.

---

## 🛠 Teknik Mimari ve Özellikler
*   **YOLOv8 Dedektörü:** Görüntüdeki nesnelerin konumlarını yüksek doğrulukla belirler.
*   **Re3 Tracker:** RNN tabanlı yapısı sayesinde nesnenin sadece o anki görüntüsüne değil, önceki karelerdeki hareketine de odaklanır.
*   **Hibrit Karar Mekanizması:** Dedektörden gelen veri ile tracker verisini birleştirerek, dedektörün nesneyi kaybettiği durumlarda tracker üzerinden devam eder.
*   **Kalman Filtresi Entegrasyonu:** Hareket tahminini stabilize etmek ve ani sıçramaları engellemek için kullanılır.

---

## 📋 Gereksinimler

Projenin çalışması için bilgisayarında Python yüklü olmalıdır. Gerekli kütüphaneler şunlardır:

| Kütüphane | Kullanım Amacı |
| :--- | :--- |
| `ultralytics` | YOLOv8 modelini çalıştırmak için. |
| `opencv-python` | Görüntü işleme ve görselleştirme için. |
| `torch` & `torchvision` | Derin öğrenme modellerinin GPU/CPU üzerinde koşması için. |
| `numpy` | Matris işlemleri ve koordinat hesaplamaları için. |

---

## ⚙️ Kurulum (Sıfırdan Başlangıç)

1.  **Depoyu Klonla:**
```bash
git clone [https://github.com/TunahanYlcnn/YOLOv8-Re3-Hybrid-Tracker.git](https://github.com/TunahanYlcnn/YOLOv8-Re3-Hybrid-Tracker.git)
cd YOLOv8-Re3-Hybrid-Tracker
```
2.  **Kütüphaneleri Yükle:**
```bash
pip install -r requirements.txt
```
