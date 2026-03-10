# 🏠 House Price Prediction | Deep Learning Regression

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

This project implements a **Artificial Neural Network (ANN)** to predict housing prices. By leveraging deep learning, the model identifies complex patterns between features like square footage, location (GPS coordinates), and property condition to estimate market value.

---

## 📊 Exploratory Data Analysis (EDA)
Before modeling, we performed extensive EDA to understand the distribution of prices and the impact of various features:

* **Outlier Detection:** Identified anomalies, such as a property with 33 bedrooms.
* **Feature Correlation:** Analyzed how `waterfront` views and `sqft_living` significantly drive price.
* **Time-Series Analysis:** Extracted `year` and `month` from sales dates to account for market seasonality.



---

## ⚙️ Data Preprocessing
To prepare the data for the Neural Network:
1.  **Feature Engineering:** Dropped irrelevant columns (like `id`) and transformed the `date` column into numerical features.
2.  **Train/Test Split:** Partitioned the data (**70% Train / 30% Test**) to ensure robust evaluation.
3.  **Scaling:** Applied `MinMaxScaler` to normalize all input features between 0 and 1, preventing features with large scales from dominating the model weights.

---

## 🧠 Model Architecture
The model is built using the **Keras Sequential API** with the following architecture:

| Layer | Type | Neurons | Activation |
| :--- | :--- | :--- | :--- |
| **Input** | Dense | 19 | ReLU |
| **Hidden 1** | Dense | 19 | ReLU |
| **Hidden 2** | Dense | 19 | ReLU |
| **Hidden 3** | Dense | 19 | ReLU |
| **Output** | Dense | 1 | Linear |

* **Optimizer:** Adam
* **Loss Function:** MSE (Mean Squared Error)



[Image of artificial neural network architecture diagram]


---

## 📈 Performance Evaluation
The model was trained for **400 epochs**. Evaluation metrics show high precision for properties within the standard market range.

### Metrics:
* **MAE (Mean Absolute Error):** Represents the average dollar amount the model is off.
* **RMSE (Root Mean Squared Error):** Penalizes larger errors, useful for identifying pricing misses on luxury estates.
* **Explained Variance Score:** Measures how much of the data variation the model successfully captured.



---

## 🛠️ Installation & Usage
1. Clone this repository.
2. Ensure you have `TensorFlow`, `Pandas`, and `Scikit-Learn` installed.
3. Run the Jupyter Notebook to train the model and see the results.


