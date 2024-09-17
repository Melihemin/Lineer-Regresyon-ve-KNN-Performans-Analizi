# Makine Öğrenimi: Lineer Regresyon ve KNN Performans Analizi

Bu proje, **Lineer Regresyon** ve **K-Nearest Neighbors (KNN)** algoritmalarını büyük bir veri seti üzerinde karşılaştırır. Amacımız her iki modelin performansını değerlendirmek ve grafiklerle nasıl farklı olduklarını görmek.

## İçindekiler

- [Giriş](#giriş)
- [Veri Seti](#veri-seti)
- [Modeller](#modeller)
- [Model Eğitimi ve Performans](#model-eğitimi-ve-performans)
- [Grafikler](#grafikler)
- [Sonuçlar](#sonuçlar)

## Giriş

Bu çalışmada iki farklı makine öğrenimi modeli kullanarak bir veri setindeki tahminlerin doğruluğunu test ettik. Grafiklerle desteklenen bu analiz, her iki modelin performansını karşılaştırmamıza yardımcı oldu.

## Veri Seti

Veri seti, büyük bir veri kümesini içerir ve yaklaşık 10 MB boyutundadır. Bu veri seti, çeşitli özelliklere sahip araçların fiyatlarını tahmin etmeye yönelik kullanılmıştır.

## Modeller

### Lineer Regresyon

- **Açıklama**: Bağımlı bir değişkeni bağımsız değişkenlerle lineer bir ilişki kullanarak tahmin eder.
- **Kullanım**: Sürekli değişkenlerin tahmininde kullanılır.

### K-Nearest Neighbors (KNN)

- **Açıklama**: Sınıflandırma veya regresyon problemlerinde en yakın komşuların bilgilerini kullanarak tahmin yapar.
- **Kullanım**: Hem sınıflandırma hem de regresyon için uygundur.

## Model Eğitimi ve Performans

Her iki modeli de veri setinde eğittik ve performanslarını ölçtük. Doğruluk ve hata oranlarını hesapladık.

### Lineer Regresyon

- **Eğitim**: Modeli eğitim verisiyle eğittik.
- **Performans**: Test verisi üzerinde tahminler yaparak hata oranını hesapladık.

### KNN

- **Eğitim**: Modeli eğitim verisiyle eğittik.
- **Performans**: Test verisi üzerinde tahminler yaparak doğruluk skorunu hesapladık.

## Grafikler

Grafiklerle model performansını ve tahminlerin dağılımını karşılaştırdık.

### 1. Performans Grafikler

Aşağıdaki grafiklerde modellerin doğruluk skorları ve hata oranları gösterilmektedir:

```python
import matplotlib.pyplot as plt
import numpy as np

# Örnek veri
models = ['Lineer Regresyon', 'KNN']
accuracy_scores = [0.79, 0.85]  # Örnek doğruluk skorları
mse_scores = [0.22, 0.15]       # Örnek MSE değerleri

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Doğruluk Skorları Grafiği
ax[0].bar(models, accuracy_scores, color=['blue', 'green'])
ax[0].set_title('Model Doğruluk Skorları')
ax[0].set_xlabel('Model')
ax[0].set_ylabel('Doğruluk Skoru')

# Hata Oranı Grafiği
ax[1].bar(models, mse_scores, color=['blue', 'green'])
ax[1].set_title('Model Hata Oranı (MSE)')
ax[1].set_xlabel('Model')
ax[1].set_ylabel('Mean Squared Error')

plt.tight_layout()
plt.show()
```


## 2. Tahminlerin Dağılımı
Aşağıdaki grafiklerde, her iki modelin tahminlerinin nasıl dağıldığı gösterilmektedir:
```python
import seaborn as sns
import pandas as pd

# Model tahminlerini oluşturuyoruz
data = pd.DataFrame({
    'Gerçek': y_test,
    'Lineer Regresyon Tahmini': linear_reg.predict(X_test),
    'KNN Tahmini': knn_model.predict(X_test)
})

# Tahminler grafiklerle karşılaştırıyoruz
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

sns.histplot(data['Lineer Regresyon Tahmini'], ax=ax[0], kde=True, color='blue')
ax[0].set_title('Lineer Regresyon Tahminleri')
ax[0].set_xlabel('Tahmin Değeri')
ax[0].set_ylabel('Frekans')

sns.histplot(data['KNN Tahmini'], ax=ax[1], kde=True, color='green')
ax[1].set_title('KNN Tahminleri')
ax[1].set_xlabel('Tahmin Değeri')
ax[1].set_ylabel('Frekans')

plt.tight_layout()
plt.show()
```

## Sonuçlar
Lineer Regresyon: Genellikle daha yüksek hata oranları olabilir ve bazı durumlarda performansı sınırlı olabilir.
KNN: Doğru ayarlarla genellikle daha yüksek doğruluk ve daha düşük hata oranları sağlayabilir.
Grafikler, her iki modelin performansını ve tahmin dağılımlarını görsel olarak karşılaştırmak için kullanıldı. Bu bilgiler, hangi modelin belirli veri setlerinde daha uygun olduğunu belirlemede yardımcı olabilir.


![image](https://github.com/user-attachments/assets/597e4a34-335f-4e0b-aec0-4e5fff4c78ec)
![image](https://github.com/user-attachments/assets/74ae27f3-c152-4bd0-9f70-fa83196f09be)


kaggle: https://www.kaggle.com/code/melihemin/knnlr
