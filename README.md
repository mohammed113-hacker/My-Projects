# My-Projects

Project Title : A Robust Machine Learning Method For Finding Insurance Fraud Claims.

INTRODUCTION:

Insurance fraud poses a significant challenge to the insurance industry, leading to substantial financial losses and eroding trust among insurers and policyholders. 
In recent years, the rise in fraudulent insurance claims had necessitated the development of effective fraud detection methods. According to the Coalition Against Insurance Fraud, 
insurance fraud costs insurers an estimated $80 billion annually in the United States alone. In the early 2000s, the insurance industry experienced a surge in fraudulent claims 
related to staged car accidents, resulting in substantial financial losses. These incidents highlighted the need for robust fraud detection systems that can identify and prevent
such fraudulent activities. In this project, i addressed the class imbalance issue in the dataset and discussed how it affects the performance of the fraud claim detection model 
and i also discussed how we can resolve it by utilizing the Synthetic Minority Over-sampling Technique (SMOTE) algorithm. It is an oversampling technique that aims to balance the 
class distribution by creating synthetic samples of the minority class. It works by generating new synthetic samples that capture the underlying patterns and characteristics of the 
minority class, SMOTE enables the model to learn from a more representative dataset, thereby improving the performance of fraud claim detection. By applying SMOTE to the dataset,
we are able to overcome the class imbalance issue and improve the performance of the fraud claim detection model. The resampled dataset provides a more accurate representation of 
the underlying distribution, leading to enhanced detection of fraudulent claims. I evaluated the performance of the model by measuring various metrics such as accuracy, precision, 
recall, and F1-score. My findings demonstrated the effectiveness of SMOTE in addressing class imbalance with improved random forest and LightGBM model for fraud claim detection process.

Web-Page for entering the data to check for Fraud Claims:


[streamlit-output-2024-01-04-22-01-30.webm](https://github.com/mohammed113-hacker/My-Projects/assets/79789933/f2f31c4b-4daa-49ae-a929-5392f4c333e0)


DATA VISUALIZATION : (In visualize.py)

Statistical techniques are used in interpreting and selecting quality features relevant to this work to have a global view of the dataset and extract essential features. For example, I analyzed the frequency of features and the correlation between them.

![image](https://github.com/mohammed113-hacker/My-Projects/assets/79789933/9dbb9371-f187-4bce-b2be-f60e249782b9)

Correlation heat map

DATA CLEANING : (In transform1.py)

Data cleansing recognizes inaccurate and unfinished parts of the data, filling in missing data and removing data that doesn’t make sense in the study, a process 
called imputation. The process improves the quality of the dataset and saves training time. This section first drops weak features that we do not intend  to use in 
building the model. Features that are removed from the list of columns include: incident_severity,incident_state,incident_type,incident_date,authorities_contacted, 
incident_city, policy_number, policy_cls, auto_model, insured_hobbies and insured_zip. I then remove all duplicated claims and incomplete claims to remove less 
significant claims. I fill in missing values by propagating the last valid observation forward to the next valid. I also replace missing values with ‘?’ character 
with ‘nan’ this is because the python numerical python library ignores the nan values while performing mathematical computations.

![image](https://github.com/mohammed113-hacker/My-Projects/assets/79789933/09c4e444-2220-41db-9666-2d3b3c858c25)

Columns conataining NAN values

DATA TRANSFORMATION: (In transform1.py)

I transformed data into formats that machine learning algorithms can understand and model  to discover helpful information that will help select the studied 
features. For example, some machine learning algorithms may not understand text values. So i applied Label-encoding and one-hot encoding techniques to handle 
categorical features.

![image](https://github.com/mohammed113-hacker/My-Projects/assets/79789933/1abef1cf-2661-4686-b8e3-ae645637ffa6)

![image](https://github.com/mohammed113-hacker/My-Projects/assets/79789933/d6a9f615-3504-4a27-91d1-81236999dbba)

SMOTE ALGORITHM : (In random1.py,lightgb.py,gaussianb.py,svm_machine.py)

I utilized the SMOTE (Synthetic Minority Over-sampling Technique) algorithm to address the issue of imbalanced data in insurance fraud claim detection. The SMOTE 
algorithm was employed to resample the dataset, effectively generating synthetic instances of the minority classes (fraudulent claims) to balance it with the 
majority classes (non-fraudulent claims). This resampling technique helped mitigate the class imbalance problem and provided a more representative training dataset.
After creating the balanced dataset it was used to train and test the Model.

![image](https://github.com/mohammed113-hacker/My-Projects/assets/79789933/c6d2cea6-f121-4d93-8366-22fb9c4af8cb)

Dataset Shape Before And After Applying SMOTE Algorithm

![image](https://github.com/mohammed113-hacker/My-Projects/assets/79789933/84873d2f-7bd5-43d1-ae91-f9f72970df1c)

Resampled Data In The Dataset

MODEL TRAINING AND TESTING : (In random1.py,lightgb.py,gaussianb.py,svm_machine.py)

After applying the SMOTE algorithm to address the class imbalance, this could results to obtained resampled dataset with an equal representation of both fraudulent and non-fraudulent claims.Each model is trained and tested individually and performance of the model is evaluated.


MODEL EVALUATION:(In random1.py,lightgb.py,gaussianb.py,svm_machine.py)

Some Metrics were employed for the model evaluation. The F1-score combines accuracy and recall into a single statistic, offering a comprehensive assessment of the 
model's efficiency. Additionally, the confusion matrix was analyzed to provide more insights into the model's performance.

![image](https://github.com/mohammed113-hacker/My-Projects/assets/79789933/8343c5b9-88f8-4160-8f1f-3e1a09850af4)

Performance Of Random-Forest Model Before Applying SMOTE

![image](https://github.com/mohammed113-hacker/My-Projects/assets/79789933/e7d12407-57d3-4461-87c8-572931b9b15a)

Performance Of Random-Forest Model After Applying SMOTE

ROC CURVE:

Receiver Operating Characteristic (ROC) curve (also known as the AUC-ROC value). The ROC curve is a graphical representation of the performance of a binary 
classification model as the discrimination threshold is varied. The AUC-ROC value quantifies the overall performance of the model by measuring the area under this 
curve. The AUC-ROC value ranges between 0 and 1, where a value of 0 indicates poor performance (the model makes incorrect predictions for all instances), and a 
value of 1 indicates perfect performance (the model makes correct predictions for all instances).An AUC-ROC value of 0.9007482394366196 shows that random forest 
model has good discriminative ability and is capable of distinguishing between the positive and negative classes with high accuracy. The closer the AUC-ROC value 
is to 1, the better the model's performance.

![image](https://github.com/mohammed113-hacker/My-Projects/assets/79789933/d53d1bc3-584b-4087-b9c7-6ff56b72ff67)

ROC Curve For The Random-Forest Model

CONCLUSION:

This project highlights the importance of addressing class imbalance and the effectiveness of SMOTE in improving model performance in such scenarios. AtLast, a Web-Page
was made to feed the Insurance data to the well trained model to check for Fraud and Legitimate claims.


FILES:

visualizing_dataset.py : It is for visualizing the dataset using Charts,Bar-Graph, and box plots

transform1.py : It is For Cleaning the dataset by eliminating duplicate values and Transforming the Non-Numeric columns into Numeric Data( Using Label encoders)

output.py : It is for provding a Input Manually in the Web-Page to check whether the Trained Model is working Properly

MODELS:

Random-forest : random_forest.py

Lightgbm : lightgb.py

SVM : svm_machine.py

Gaussian Naive bayes : gaussiannb.py

Source.gv.pdf : This file Contains the Graphical visualization structure of Random-forest Model.
