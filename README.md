# 🤟 ASL (American Sign Language) Alphabet Recognition with ResNet50

Bu proje, **Amerikan İşaret Dili (ASL) Alfabesi** el hareketlerini görüntüler üzerinden sınıflandırmak amacıyla **Transfer Learning (ResNet50)** mimarisi kullanılarak geliştirilmiş bir derin öğrenme uygulamasıdır.

---

## 📊 Veri Seti ve Ön İşleme

* **Veri Seti:** [ASL Alphabet Dataset (Kaggle)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
* **Sınıf Sayısı:** 29 Sınıf (`A-Z` harfleri + `space`, `del`, `nothing`)
* **Toplam Görüntü:** 87.000 adet (Her sınıf için 3.000 görüntü)
* **Veri Bölünmesi:** %70 Eğitim (60.900 görüntü) / %30 Doğrulama (26.100 görüntü)
* **Giriş Boyutu:** `224 x 224 x 3` (RGB)

### Veri Artırma (Data Augmentation)
Modelin genelleme yeteneğini artırmak ve aşırı öğrenmeyi (overfitting) önlemek için eğitim setine `ImageDataGenerator` ile şu dönüşümler uygulanmıştır:
* Piksel normalizasyonu (`rescale=1./255`)
* Rastgele döndürme (`rotation_range=15`)
* Yatay ve dikey kaydırma (`width_shift_range=0.1`, `height_shift_range=0.1`)
* Yakınlaştırma (`zoom_range=0.1`)

---

## 🧠 Model Mimarisi

ImageNet ağırlıkları ile önceden eğitilmiş **ResNet50** omurgası temel alınmış ve tüm katmanlar ince ayar (fine-tuning) için eğitilebilir (`trainable=True`) hale getirilmiştir:

1. **Base Model:** ResNet50 (`include_top=False`, `weights='imagenet'`)
2. **Pooling:** `GlobalAveragePooling2D()`
3. **Fully Connected:** `Dense(256, activation='relu')`
4. **Regularization:** `Dropout(0.5)`
5. **Output Layer:** `Dense(29, activation='softmax')`

* **Toplam Parametre Sayısı:** ~24.1 Milyon
* **Optimizer:** Adam (`learning_rate=0.0001`)
* **Kayıp Fonksiyonu:** Categorical Crossentropy

---

## 📈 Eğitim Süreci ve Sonuçlar

Eğitim süreci `EarlyStopping` (`monitor='val_loss'`, `patience=10`, `restore_best_weights=True`) geri çağrısı ile 20 epoch boyunca yürütülmüştür.

| Metrik | En İyi Değer (10. Epoch) | Son Değer (20. Epoch) |
| :--- | :---: | :---: |
| **Eğitim Doğruluğu (Train Accuracy)** | `%99.79` | `%99.94` |
| **Doğrulama Doğruluğu (Val Accuracy)** | **`%97.39`** | `%97.12` |
| **Eğitim Kaybı (Train Loss)** | `0.0081` | `0.0023` |
| **Doğrulama Kaybı (Val Loss)** | **`0.1202`** | `0.1485` |

> `restore_best_weights=True` parametresi sayesinde modelin en düşük doğrulama kaybına (`0.1202`) ve **%97.39** doğrulama başarımına ulaştığı 10. epoch ağırlıkları korunmuştur.

---

## 🛠️ Kullanılan Teknolojiler

* **Python 3**
* **TensorFlow / Keras**
* **NumPy & Pandas**
* **Matplotlib & Seaborn**
* **Google Colab (GPU Runtime)**

---

## 🚀 Kurulum ve Kullanım

1. Depoyu klonlayın:
   ```bash
   git clone [https://github.com/kullanici-adin/repo-adin.git](https://github.com/kullanici-adin/repo-adin.git)
   cd repo-adin
