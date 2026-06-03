# VBM686 Programming Assignment 3: Image Classification and Object Detection

**Hazırlayan:** Salih Karayılan  
**Kurum:** Hacettepe Üniversitesi, Bilişim Enstitüsü  

## Proje Özeti
Bu proje, Bilgisayarlı Görü (Computer Vision) alanındaki temel sınıflandırma (classification) ve nesne tespiti (object detection) problemlerini derin öğrenme mimarileri kullanarak çözmeyi amaçlamaktadır. Proje, veri ön işleme, sıfırdan model kurulumu, transfer öğrenme (Transfer Learning) ve çoklu görev öğrenmesi (Multi-task Learning) olmak üzere üç ana bölümden oluşmaktadır. Modeller, Google Colab üzerindeki T4 GPU altyapısı kullanılarak TensorFlow ve Keras kütüphaneleriyle geliştirilmiştir.

---

## Part 1: CNN Sınıflandırıcı Mimarisi (From Scratch)
MIT Indoor Scenes veri seti kullanılarak, 15 farklı iç mekan kategorisi için sıfırdan bir Evrişimli Sinir Ağı (CNN) tasarlanmıştır. Veri seti **Train (3000)**, **Validation (750)** ve **Test (750)** olarak yönergelere uygun şekilde bölünmüştür.

### Hiperparametre Optimizasyonu (Grid Search)
Aşırı öğrenmeyi (overfitting) engellemek ve en iyi performansı bulmak amacıyla 24 farklı konfigürasyon (3 Learning Rate x 2 Batch Size x 4 Dropout Rate) 100'er epoch boyunca test edilmiştir.

* **Denenen Parametreler:**
  * Batch Sizes: [32, 64]
  * Learning Rates: [0.01, 0.001, 0.0001]
  * Dropout Rates: [0.2, 0.3, 0.4, 0.5]
* **En İyi Model Sonuçları:**
  * En iyi hiperparametreler: `Batch Size: 32`, `Learning Rate: 0.001`, `Dropout: 0.5`
  * **Test Seti Doğruluğu (Test Accuracy):** %47.07

*(Not: Sınırlı veri boyutuyla sıfırdan eğitilen bir ağ için bu oran literatürdeki beklentilerle örtüşmektedir. Eğitim sürecine ait Loss/Accuracy grafikleri ve Confusion Matrix `src/` klasöründeki kodlar çalıştırılarak incelenebilir.)*

---

## Part 2: VGG-16 ile Transfer Öğrenme (Transfer Learning)
Sıfırdan eğitilen modelin limitlerini aşmak için ImageNet üzerinde önceden eğitilmiş VGG-16 mimarisi kullanılmıştır.

1. **Feature Extraction (Özellik Çıkarımı):** VGG-16'nın konvolüsyon katmanları dondurulmuş (frozen) ve sadece projeye özel eklenen Sınıflandırma (Fully Connected) katmanları eğitilmiştir. Test başarısı: **%47.07**.
2. **Fine-Tuning (İnce Ayar):** VGG-16'nın son konvolüsyon bloğunun kilidi açılarak, `1e-5` gibi çok düşük bir öğrenme oranıyla ağ veri setine özel olarak ince ayardan geçirilmiştir. 
* **Final Test Seti Doğruluğu:** **%60.40**
* **Sonuç:** Transfer öğrenme uygulanarak başarı oranında gözle görülür bir artış sağlanmıştır.

---

## Part 3: ResNet50 ile Nesne Tespiti (Object Detection)
Bu bölümde Raccoon (Rakun) veri seti kullanılarak, resimdeki rakunun varlığını ve konumunu tespit eden Çok Başlıklı (Multi-Task) bir mimari kurulmuştur. Temel ağ (Backbone) olarak önceden eğitilmiş `ResNet50` kullanılmıştır.

### Mimari Detaylar
Ağ, ResNet50 gövdesinden sonra iki farklı göreve ayrılmıştır:
* **Classification Head (Sınıflandırma):** Rakun var/yok tespiti için `Softmax` aktivasyonu ve `Categorical Crossentropy Loss` kullanılmıştır.
* **Regression Head (Konumlandırma):** Bounding Box (sınırlayıcı kutu) koordinatlarının tahmini için `Sigmoid` aktivasyonu ve `Mean Squared Error (MSE) Loss` (L2) kullanılmıştır.

### Performans ve Sonuçlar
İki kayıp fonksiyonu birleştirilerek (L_total = L_softmax + L_2) model eşzamanlı olarak eğitilmiştir. Bounding box tahminlerinin başarısını ölçmek için Intersection over Union (IoU) metriği kullanılmıştır.
* **Mean IoU (MIoU) Skoru:** **0.5683** * Model, standart kabul barajı olan 0.5'i rahatlıkla geçerek kutu tahmininde (Localization) yüksek başarı göstermiştir. Tespit edilen örnek Bounding Box çıktılarına ait görseller proje demosu içerisinde mevcuttur.

---

## Kurulum ve Çalıştırma
1. `src/` dizini altındaki Python/Colab betikleri sırasıyla çalıştırılmalıdır.
2. `kaggle.json` bağımlılığını kaldırmak adına MIT veri seti ve Raccoon veri seti kod blokları içerisinde otomatik olarak indirilip ön işleme (preprocessing) tabi tutulacak şekilde ayarlanmıştır.
3. Çıktılar ve ağırlıklar (keras uzantılı model dosyaları) çalışma dizinine kaydedilecektir.
