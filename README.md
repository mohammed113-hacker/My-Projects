# My-Projects
INTRODUCTION:

The insurance fraud claim detection is a critical task in the insurance industry various methodologies were proposed in terms of detecting fraud claims but the 
main thing is handling the proportionality between the fraud and non-fraud cases in the dataset before it was giving to the model in that case class imbalance may
occur. Class imbalance where the number of fraudulent claims is significantly lower than legitimate claims poses a challenge for accurate fraud detection.
In this work, I employed Synthetic Minority Over-sampling Technique (SMOTE) algorithm to address the imbalance problem and enhance the performance of the fraud 
claim detection model. SMOTE is a popular technique for oversampling the minority class by creating synthetic samples that are similar to the existing minority 
class instances. By generating synthetic examples, SMOTE helps in balancing the class distribution and allows the model to learn from a more representative 
dataset. This technique is particularly effective when the available data is limited and insufficient to capture the complexities of the limited class. This 
process results in a larger and more balanced dataset, enabling the model to learn from a diverse range of fraudulent claim patterns. By applying SMOTE to the 
dataset, we are able to overcome the class imbalance issue and improve the performance of the fraud claim detection model. The resampled dataset provides a more 
accurate representation of the underlying distribution, leading to enhanced detection of fraudulent claims. I evaluated the performance of the model by measuring 
various metrics such as accuracy, precision, recall, and F1-score. Our findings demonstrate the effectiveness of SMOTE in addressing class imbalance with improved 
random forest and LightGBM model for fraud claim detection process.

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

CONCLUSION:

This project highlights the importance of addressing class imbalance and the effectiveness of SMOTE in improving model performance in such scenarios.

MODELS:

Random-forest : random1.py

Lightgbm : lightgb.py

SVM : svm_machine.py

Gaussian Naive bayes : gaussiannb.py

Source.gv.pdf : This file Contains the Graphical visualization structure of Random-forest Model.
