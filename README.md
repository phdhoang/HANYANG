# **BERT Meta-Learning and Bayesian Hyperparameters Optimization for Regression**

This project demonstrates the application of BERT-based sentence embeddings, PCA, and one-hot encoding for feature engineering, followed by a robust regression analysis using Bayesian Optimization to tune various machine learning models. A stacking ensemble with a meta-learner is also implemented to further enhance predictive performance.

The project leverages Python with popular libraries like pandas, numpy, scikit-learn, tensorflow, optuna, sentence-transformers, and scipy.sparse.

## **Table of Contents**

* [Introduction](#bookmark=id.qp7iu71n942d)  
* [Features](#bookmark=id.6ky1lhf1mtyo)  
* [Installation](#bookmark=id.31xdh0sgs7zn)  
* [Usage](#bookmark=id.xofe0phhbpxd)  
* [Data](#bookmark=id.aovq5o7b200m)  
* [Methodology](#bookmark=id.uz6offafwz5u)  
  * [Feature Engineering](#bookmark=id.fm5v9iy2t0nv)  
  * [Bayesian Optimization (Optuna)](#bookmark=id.8aoqclms4ou9)  
  * [Stacking Ensemble](#bookmark=id.jib5jfpym6aj)  
* [Models Used](#bookmark=id.9cgt7fplp5t2)  
* [Results](#bookmark=id.gjnaehmjwrkr)  
* [Visualizations](#bookmark=id.b94pn23r3zkv)  
* [Contributing](#bookmark=id.9728ah5j5fgq)  
* [License](#bookmark=id.tbc1x6n59bzz)  
* [Acknowledgements](#bookmark=id.pybl0hil7e5l)

## **Introduction**

This repository provides a comprehensive framework for building and optimizing regression models, particularly focusing on handling diverse feature types including numerical, categorical, and textual data. The core idea is to transform the input data into a rich feature space and then employ advanced hyperparameter optimization techniques to find the best-performing models, ultimately combining them using a stacking ensemble.
The input information of the model is the parameters of construction projects, along with the history of changes (described in natural language). From there, the models perform analysis and predict the actual costs of these projects. Detailed descriptions of the inputs and outputs are presented in Table 1.

Table 1. Input/Output Formats for Project Cost Prediction Model
## **Features**
Category	Feature Name	Data Type (Before Preprocessing)	Data Type (After Preprocessing for Model Input)	Description	Example Value (Original)	Example Value (Processed)
Numerical Features (Input)	Main-contractors experience	Integer (Years)	Float (Scaled/Normalized)	Years of experience of the main contractor.	15	0.75 (after scaling)
	Number of contractors	Integer	Float (Scaled/Normalized)	Total number of contractors involved.	3	0.3 (after scaling)
	Number of Contract changes	Integer	Float (Scaled/Normalized)	Count of formal changes made to the contract.	2	0.2 (after scaling)
	Number of Design changes	Integer	Float (Scaled/Normalized)	Count of revisions made to the project design.	5	0.5 (after scaling)
	Number of Price linkage	Integer	Float (Scaled/Normalized)	Frequency of price adjustments linked to external factors.	1	0.1 (after scaling)
	Start date	Date	Numerical	Date the project officially commenced.	1/15/2013	748950 (days since epoch)
	Announcement Date	Date	Numerical	Date the project was publicly announced.	11/1/2012	748880 (days since epoch)
	Completion Date	Date	Numerical	Date the project was completed.	3/20/2014	749370 (days since epoch)
	Design amount USD	Float (USD)	Float (Scaled/Normalized)	Targeted design costs ($)	 $1,990,000 	0.15 (after scaling)
	Estimated amount USD	Float (USD)	Float (Scaled/Normalized)	Estimated cost according to tendering packs ($)	 $2,010,000 	0.8 (after scaling)
	Initial amount USD	Float (USD)	Float (Scaled/Normalized)	Initial amount of selected bid package ($)	 $1,950,000 	0.78 (after scaling)
Categorical Features (Input)	Location	String/Categorical	One-Hot Encoded Vector	Geographic location of the project.	Seoul	[0, 1, 0, ...] (for Seoul)
	Bidding methods	String/Categorical	One-Hot Encoded Vector	Method used for contractor bidding.	Lowest price	[1, 0, 0, ...] (for Open)
	Category	String/Categorical	One-Hot Encoded Vector	General classification of the project.	Residential for sale	[0, 0, 1, ...] (for Residential)
	Work type	String/Categorical	One-Hot Encoded Vector	Main work type of the general contractors.	Construction	[0, 1, 0, ...] (for New Build)
	General Contractor	String/Categorical	One-Hot Encoded Vector	Identifier for the general contractor.	A Inc. 	[1, 0, 0, ...] (for GC_A)
Text Features (Input)	Additional Description	Text	BERT Embeddings	Unstructured text providing descriptions, notes on design changes, delays or other issues	On july 2012, because of the design change, C18 beams are resized...	torch.Tensor([...]) (BERT embedding vector)
Output	Project actual cost	Float (USD)	Float (Raw)	The true, final cost of the project (target variable).	2,150,000.00	2,150,000.0 (raw) or 0.85 (if scaled)

![image](https://github.com/user-attachments/assets/e9cc1711-4db3-40d6-932a-2ae4492001c0)

* **Data Loading:**
* 
* Reads data from an Excel file (.xlsx).  
* **Feature Engineering:**  
  * **PCA (Principal Component Analysis):** For dimensionality reduction on numerical features (PC1 to PC6).  
  * **One-Hot Encoding:** For categorical features (C1 to C5).  
  * **BERT Sentence Embeddings:** Uses sentence-transformers (specifically 'all-mpnet-base-v2') to convert text features (T1) into dense vector representations.  
  * **Sparse Matrix Handling:** Efficiently combines numerical, one-hot encoded, and text embedding features into a single sparse matrix.  
* **Model Training & Evaluation:**  
  * Splits data into training and testing sets.  
  * Utilizes StandardScaler for feature scaling.  
  * Employs K-Fold Cross-Validation for robust model evaluation during hyperparameter tuning.  
* **Bayesian Hyperparameter Optimization (Optuna):**  
  * Tunes hyperparameters for individual base models:  
    * Lasso Regression  
    * Ridge Regression  
    * Gradient Boosting Regressor  
    * Random Forest Regressor  
  * Optimizes for Mean Squared Error (MSE).  
* **Stacking Ensemble:**  
  * Uses predictions from the optimized base models as input for a meta-learner (Lasso Regression).  
  * The meta-learner's hyperparameters are also tuned using Bayesian Optimization.  
* **Performance Metrics:** Reports MSE, R², RMSE, and MAPE for all models.  
* **Visualizations:** Generates scatter plots to compare actual vs. predicted values for each model.

## **Installation**

To set up the project, clone the repository and install the required Python packages.

git clone \<repository\_url\>  
cd \<repository\_name\>  
pip install \-r requirements.txt

**requirements.txt content:**

numpy==1.25.2  
scikeras==0.13.0  
gensim==4.3.3  
scikit-learn-intelex==2025.5.0  
scikit-learn==1.7.0 \# Or the version compatible with scikeras, typically \>=1.4.2  
pyswarms==1.3.0  
mealpy==3.0.2  
optuna==4.3.0  
matplotlib  
seaborn  
scipy  
joblib  
xgboost  
pandas  
tensorflow  
openpyxl  
sentence-transformers  
transformers  
torch  
h5py  
absl-py  
rich  
namex  
optree  
ml-dtypes  
packaging  
typing-extensions  
markdown-it-py  
pygments  
mdurl  
smart-open  
wrapt  
daal==2025.5.0  
tbb==2022.1.0  
tcmlib==1.3.0  
alembic  
colorlog  
sqlalchemy  
pytz  
tzdata  
charset-normalizer  
idna  
urllib3  
certifi  
networkx  
jinja2  
MarkupSafe  
regex  
tokenizers  
safetensors  
nvidia-cuda-nvrtc-cu12==12.4.127  
nvidia-cuda-runtime-cu12==12.4.127  
nvidia-cuda-cupti-cu12==12.4.127  
nvidia-cudnn-cu12==9.1.0.70  
nvidia-cublas-cu12==12.4.5.8  
nvidia-cufft-cu12==11.2.1.3  
nvidia-curand-cu12==10.3.5.147  
nvidia-cusolver-cu12==11.6.1.9  
nvidia-cusparse-cu12==12.3.1.170  
nvidia-nvjitlink-cu12==12.4.127  
triton  
sympy  
mpmath

*(Note: Some package versions are fixed for compatibility. It's recommended to use a virtual environment.)*

## **Usage**

1. **Place** your **data:** Ensure your Excel data file (20250430@DesignChange.Sumaried.AfterPCA.anonymized.xlsx or similar, as expected by the notebook) is in the root directory of the project, or update the data loading path in the notebook.  
2. **Run** the Jupyter **Notebook:** Open and run the share\_bert\_meta\_learning\_and\_bayesian\_hyperparameters\_optimization\_ipynb.ipynb notebook cell by cell. The notebook contains all the code for data preprocessing, model training, optimization, and evaluation.

## **Data**

The project expects an Excel file with the following (or similar) columns:

* **Project ID**: Unique identifier for projects.  
* **PC1 \- PC6**: Numerical features (likely principal components).  
* **C1 \- C5**: Categorical features.  
* **T1**: Textual feature (e.g., descriptions or notes related to design changes).  
* **Y1**: The target numerical variable for regression.  
* **L**: Another numerical feature (its role is not explicitly defined in the provided snippets but is present in the dataframe).

The notebook handles data loading and checks for missing values.

## **Methodology**

### **Feature Engineering**

* **Numerical Features (PC1-PC6, L):** Used directly after scaling.  
* **Categorical** Features (**C1-C5):** Transformed using OneHotEncoder to create binary features for each category.  
* **Textual Features (T1):** Processed using a pre-trained SentenceTransformer model ('all-mpnet-base-v2') to generate dense vector embeddings. These embeddings capture semantic meaning from the text.  
* **Feature Stacking:** All processed features (numerical, one-hot, and sentence embeddings) are horizontally stacked into a single sparse matrix (X\_num\_oh\_word\_tfidf).

### **Bayesian Optimization (Optuna)**

Optuna is used to find optimal hyperparameters for the base models. For each model, an objective function is defined that trains the model using 5-Fold Cross-Validation on the training data and returns the average Mean Squared Error (MSE). Optuna then explores the hyperparameter space to minimize this MSE.

Hyperparameters tuned include:

* **Lasso/Ridge:** alpha (regularization strength), fit\_intercept.  
* **Gradient Boosting Regressor:** n\_estimators, learning\_rate, max\_depth, min\_samples\_split, min\_samples\_leaf, subsample.  
* **Random Forest Regressor:** n\_estimators, max\_depth, min\_samples\_split, min\_samples\_leaf, bootstrap.

### **Stacking Ensemble**

The stacking ensemble combines the predictions of the base models.

1. **Out-of-Fold (OOF) Predictions:** Each base model generates predictions on the validation folds during its cross-validation training. These OOF predictions form the training data for the meta-learner (X\_train\_meta\_bo).  
2. **Test** Set **Predictions:** Each base model also predicts on the unseen test set, and these predictions are averaged to form the test data for the meta-learner (X\_test\_meta\_bo).  
3. **Meta-Learner:** A Lasso Regression model is used as the meta-learner, trained on the OOF predictions of the base models to learn how to best combine their outputs. Its hyperparameters are also optimized with Optuna.  
4. **Final Prediction:** The meta-learner makes the final predictions on X\_test\_meta\_bo.

## **Models Used**

**Base Models:**

* Lasso Regression (sklearn.linear\_model.Lasso)  
* Ridge Regression (sklearn.linear\_model.Ridge)  
* Gradient Boosting Regressor (sklearn.ensemble.GradientBoostingRegressor)  
* Random Forest Regressor (sklearn.ensemble.RandomForestRegressor)

**Meta-Learner:**

* Lasso Regression (sklearn.linear\_model.Lasso)

## **Results**

The performance of each optimized model (Lasso, Ridge, GBR, RF, and the Stacking Ensemble) is evaluated on the test set using:

* **Mean Squared Error (MSE)**  
* **R-squared (R²)**  
* **Root Mean Squared Error (RMSE)**  
* **Mean** Absolute Percentage Error **(MAPE)**

The notebook outputs the best hyperparameters found by Bayesian Optimization for each model, along with their respective evaluation metrics. The stacking ensemble typically achieves the best overall performance.

Example output snippet for Stacking Ensemble:

\--- Best Stacking Ensemble (BayesOpt Meta-Learner: Lasso) \---  
Test MSE (Stacking \- BayesOpt): 2.3850293266880402e+13  
Test R² (Stacking \- BayesOpt): 0.940888218122135  
Test RMSE (Stacking \- BayesOpt): 4883676.204139706  
Test MAPE (Stacking \- BayesOpt): 0.11559909067533342

Best parameters for Meta-Learner (Lasso \- BayesOpt): {'alpha': 97.10916550995319, 'fit\_intercept': False}  
Bayesian Optimization Time: 6555.61 seconds

## **Visualizations**

Scatter plots are generated to visualize the "Actual vs. Predicted" values for each model, allowing for a quick visual assessment of their performance and areas of prediction deviation. A perfect model would have all points lying on the diagonal line.

## **Contributing**

Feel free to fork this repository, submit pull requests, or open issues if you find bugs or have suggestions for improvements.

## **License**

This project is open-sourced under the MIT License. See the LICENSE file for more details.

## **Acknowledgements**

* **Optuna:** For providing an efficient hyperparameter optimization framework.  
* **Sentence-Transformers:** For easy access to pre-trained sentence embeddings.  
* **Scikit-learn:** For various machine learning models and utilities.  
* **TensorFlow/Keras:** For deep learning functionalities.
