# Credit Risk Scoring with Machine Learning

An end-to-end credit risk prediction project built on the Kaggle **Credit Risk Dataset**, featuring imbalanced data handling, comparison of four machine learning models (Logistic Regression, KNN, Random Forest, and XGBoost), SHAP-based model explainability, and a Streamlit application for bank loan application form.

## Project Structure

```text
credit-risk-scoring-ml/
├── data/raw/credit_risk_dataset.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_shap_analysis.ipynb
├── app/app.py
├── models/
└── README.md
```

## Installation & Usage

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn shap matplotlib streamlit

# Run the notebooks in order: 01 -> 02 -> 03
cd app
streamlit run app.py
```

---

## Dataset

**Source:** Kaggle — Credit Risk Dataset (~32,000 records)

**Target Variable:** `loan_status`

* **1:** Default
* **0:** Loan repaid successfully


**Original Features:**

* Age
* Annual income
* Home ownership
* Employment length
* Loan purpose
* Loan grade
* Loan amount
* Interest rate
* Loan-to-income ratio
* Previous default history
* Credit history length

---

### 1. Data Cleaning

The **Interquartile Range (IQR)** method was initially evaluated for outlier detection. However, due to the highly right-skewed distributions of variables such as annual income and loan amount, IQR incorrectly flagged many legitimate borrowers as outliers.

Instead, **domain knowledge-based validation rules** were applied:

* Removed applicants with ages outside **18–90** (clear data errors)
* Removed records where **employment length exceeded age**
* Capped annual income at the **99th percentile** instead of removing high-income observations
* Missing values were imputed using the **median**

  * `loan_int_rate`: 9.6% missing
  * `person_emp_length`: 2.75% missing

---

### 2. Feature Engineering

Five additional features were created to capture more meaningful borrower characteristics:

* `credit_history_ratio` = Credit history length / Age
* `emp_length_ratio` = Employment length / Age
* `loan_to_emp_length` = Loan amount / (Employment length + 1)
* `income_to_loan` = Annual income / (Loan amount + 1)
* `age_group` = Age categorized into four groups (18–25, 26–35, 36–50, 50+)
<img width="890" height="690" alt="image" src="https://github.com/user-attachments/assets/ed098bfb-b5ce-4475-a4dd-1714b77a8af8" />

---

### 3. Imbalanced Data Handling

The dataset was split using **stratified train-test sampling**, ensuring that the test set preserved the original class distribution.

To prevent data leakage:

* **SMOTE was applied only to the training set.**
* The test set remained untouched to ensure realistic performance evaluation.

Different strategies were used depending on the model type:

* **Logistic Regression** and **KNN**

  * Trained on the SMOTE-balanced dataset.
* **Random Forest** and **XGBoost**

  * Trained on the original data using

    * `class_weight="balanced"` (Random Forest)
    * `scale_pos_weight` (XGBoost)

Tree-based models typically benefit less from synthetic oversampling and may overfit when SMOTE is applied.
<img width="954" height="860" alt="image" src="https://github.com/user-attachments/assets/71dcb9cd-4d18-43b3-8dea-eacc216ded96" />

---

### 4. Model Comparison

| Model                    | Test ROC-AUC |
| ------------------------ | -----------: |
| KNN                      |        0.870 |
| Logistic Regression      |        0.876 |
| Random Forest            |        0.930 |
| **XGBoost (Best Model)** |    **0.946** |

<img width="613" height="547" alt="image" src="https://github.com/user-attachments/assets/bc60e977-1726-49f8-af29-3d9b10391f89" />


---

### 5. Cost-Based Threshold Optimization

Instead of using the default **0.50 classification threshold**, the optimal threshold was selected using a **cost-sensitive approach**.

The following cost assumption was used:

* **False Negative (FN)** cost = **5 × False Positive (FP)** cost

This reflects a realistic banking scenario where approving a loan for a borrower who later defaults is significantly more expensive than rejecting a creditworthy applicant.

A grid search over thresholds between **0.05 and 0.95** was performed, and the threshold minimizing the total expected cost was selected.

---

### 6. Model Explainability (SHAP)

Model interpretability was performed using **SHAP's `PermutationExplainer`**.

#### Leakage Analysis

Because **`loan_grade`** appeared to be the strongest predictor, it was specifically examined for potential target leakage.

The analysis showed that `loan_grade` accounted for approximately **14.6% of the total SHAP importance**, indicating that the model does **not** rely excessively on a single feature.

Instead, predictions are driven by multiple independent risk factors, including:

* Income
* Debt burden
* Credit history
* Employment information
* Loan characteristics

No evidence of significant target leakage (e.g., one feature contributing over 50% of total importance) was observed.

---

### 7. Streamlit Application

<img width="1470" height="831" alt="image" src="https://github.com/user-attachments/assets/e5a4ad72-0876-4b53-a015-6f1a7bd60d9b" />

<img width="1470" height="831" alt="image" src="https://github.com/user-attachments/assets/fadc8477-3d76-46ad-ba8d-7657b57e5321" />

---

## Technologies Used

* **Python**
* **pandas**
* **NumPy**
* **scikit-learn** (Logistic Regression, KNN, Random Forest, preprocessing)
* **XGBoost**
* **imbalanced-learn** (SMOTE)
* **SHAP** (`PermutationExplainer`)
* **Matplotlib**
* **Streamlit**


# Kredi Risk Skoru Modeli

Kaggle "Credit Risk Dataset" üzerine kurulmuş, dengesiz veri yönetimi, 4 model
karşılaştırması (Logistic Regression, KNN, Random Forest, XGBoost), SHAP ile
açıklanabilirlik analizi ve gerçek bir banka başvuru formuna benzer Streamlit
arayüzü içeren uçtan uca kredi risk tahmin projesi.

## Proje Yapısı

```
credit-risk-scoring-ml/
├── data/raw/credit_risk_dataset.csv
├── notebooks/
│   ├── 01_eda.ipynb              
│   ├── 02_modeling.ipynb         
│   └── 03_shap_analysis.ipynb    
├── app/app.py                    
├── models/                       
└── README.md
```

## Kurulum ve Çalıştırma

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn shap matplotlib streamlit

# Notebook'ları sırayla çalıştır: 01 -> 02 -> 03
# Sonra:
cd app
streamlit run app.py
```

---

## Veri Seti

**Kaynak:** Kaggle — Credit Risk Dataset (~32.000 satır)
**Hedef değişken:** `loan_status` (1 = temerrüt/default, 0 = kredi düzgün ödendi)
**Sınıf dağılımı:** %78.2 (ödendi) / %21.8 (default) — dengesiz, ama literatürdeki
klasik "%93/%7" örneklerinden daha az agresif bir dengesizlik.

**Ham feature'lar:** yaş, yıllık gelir, ev sahiplik durumu, çalışma süresi, kredi
amacı, kredi notu (grade), kredi tutarı, faiz oranı, gelire oranla kredi tutarı,
geçmişte temerrüt kaydı, kredi geçmişi uzunluğu.

---

## Metodoloji

### 1) Veri Temizliği
IQR (Interquartile Range) yöntemi bu veri setinde denendi ama **çarpık dağılım**
(sağa çarpık gelir/kredi tutarı gibi değişkenler) nedeniyle gerçek, geçerli
müşterileri "aykırı" diye işaretlediği görüldü — bu yüzden yerine **domain
bilgisine dayalı mantıksal kurallar** kullanıldı:
- Yaş 18-90 dışı → silindi (kesin veri hatası)
- Çalışma süresi > yaş → mantıksal tutarsızlık, silindi
- Gelir → üst %1 quantile'da kırpıldı (capping) — aşırı uç ama muhtemelen gerçek
  değerler, silmek yerine sınırlandı
- Eksik değerler (`loan_int_rate` %9.6, `person_emp_length` %2.75) → medyan ile
  dolduruldu

### 2) Feature Engineering
Ham kolonlardan 5 yeni değişken türetildi:
- `credit_history_ratio` = kredi geçmişi uzunluğu / yaş
- `emp_length_ratio` = çalışma süresi / yaş
- `loan_to_emp_length` = kredi tutarı / (çalışma süresi + 1)
- `income_to_loan` = gelir / (kredi tutarı + 1)
- `age_group` = yaş kategorik gruplara ayrıldı (18-25, 26-35, 36-50, 50+)

<img width="890" height="690" alt="image" src="https://github.com/user-attachments/assets/092981d1-6def-4f3d-9bdf-9964c4cf67d2" />


### 3) Dengesiz Veri Yönetimi
- Train/test **stratify** ile bölündü, test seti gerçek dünya dağılımını korudu
- **SMOTE sadece train setine** uygulandı (test setine asla dokunulmadı —
  aksi halde metrikler yanıltıcı/iyimser çıkardı)
- Mesafe/katsayı tabanlı modeller (Logistic Regression, KNN) → SMOTE'lu veri
  ile eğitildi
- Ağaç tabanlı modeller (Random Forest, XGBoost) → `class_weight="balanced"` /
  `scale_pos_weight` ile orijinal veri kullanıldı (ağaç modelleri SMOTE'tan
  daha az fayda görür, hatta overfit riski artırabilir)
<img width="954" height="860" alt="image" src="https://github.com/user-attachments/assets/6442d390-72ea-4739-b194-d5f26bb8bcf6" />


### 4) Model Karşılaştırması

| Model | Test AUC |
|---|---|
| KNN | 0.870 |
| Logistic Regression | 0.876 |
| Random Forest | 0.930 |
| **XGBoost (en iyi)** | **0.946** |

<img width="613" height="547" alt="image" src="https://github.com/user-attachments/assets/72799780-3a54-4d66-84b5-f82c14496110" />


### 5) Threshold Seçimi (Cost-Based)
0.5 gibi keyfi bir varsayılan reddedildi. Bunun yerine, maliyet matrisi
varsayıldı: **FN maliyeti = FP maliyetinin 5 katı** (temerrüde düşecek birine
kredi vermenin maliyeti, iyi bir müşteriye kredi vermemenin fırsat
maliyetinden çok daha yüksektir). 0.05-0.95 aralığında grid search yapılarak
toplam maliyeti minimize eden threshold bulundu.

### 6)SHAP
`shap.PermutationExplainer` kullanıldı 

**Leakage kontrolü:** `loan_grade` (kredi notu), hedefe en yakın/riskli
görünen feature olduğu için özel olarak incelendi. Toplam SHAP öneminin
**%14.6**'sını oluşturduğu görüldü — bu, modelin tek bir dominant sinyale
(bankanın zaten verdiği hükme) değil, gelir, borç oranı, kredi geçmişi gibi
birden fazla bağımsız risk faktörüne dayandığını gösteriyor. Ciddi bir
leakage (örn. %50+) tespit edilmedi.

### 7) Arayüz 
<img width="1470" height="831" alt="image" src="https://github.com/user-attachments/assets/7016e687-90a7-429a-8360-6502301ffe83" />
<img width="1470" height="831" alt="image" src="https://github.com/user-attachments/assets/311bfac8-0368-44b8-83ef-36b182aaa183" />


## Kullanılan Teknolojiler
`scikit-learn` (Logistic Regression, KNN, Random Forest, preprocessing) ·
`xgboost` · `imbalanced-learn` (SMOTE) · `shap` (PermutationExplainer) ·
`pandas` / `numpy` · `matplotlib` · `streamlit`


