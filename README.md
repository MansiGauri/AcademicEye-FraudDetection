# AcademicEye-FraudDetection
This endeavor is rooted in the research paper titled - ['Hybrid Method in Identifying Fraud Detection in Credit Cards'](https://www.springerprofessional.de/en/hybrid-method-in-identifying-the-fraud-detection-in-the-credit-c/18238758). The authors include P. Tiwari, S. Mehta, N. Sakhuja, I. Gupta, and A. K. Singh from the Department of Computer Applications at the National Institute of Technology, Kurukshetra, India.
The study investigated contemporary fraud detection techniques, revealing their inadequacy in detecting fraud promptly. Previous methods demonstrated accuracy primarily on specific datasets or features. 
Notably, SVM outperformed Logistic Regression in handling class imbalance, while Random Forest excelled among the two. The Bagging Ensemble Classifier proved effective for highly imbalanced datasets. Decision Trees and SVM performed well on raw, unsampled data, whereas ANN and Bayesian Belief Network achieved high accuracy and detection rates at the cost of intensive training. Similarly, SVM and KNN excelled with small datasets but were less suitable for larger ones. The proposed technology aims for consistent precision and accuracy across diverse scenarios and datasets by amalgamating existing techniques.

The paper introduced a model centered on average accuracy, combining K-nearest neighbor, neural networks, and decision trees for credit card fraud detection. The model assigns a new label through majority voting for each incoming transaction. As depicted in the accompanying image, the model is designed to be effective across various dataset sizes and types. Displaying substantial accuracy improvements, this fraud detection model overcomes individual model limitations, highlighting its significant practicality.

According to the paper, the model is designed to perform effectively across various dataset sizes and types. Demonstrating substantial improvements in accuracy, this fraud detection model surpasses the limitations of individual models, signifying its significant utility.


# Dataset
The dataset has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles) on big data mining and fraud detection. The dataset was made available in Kaggle - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

# Method 

This repository presents an advanced fraud detection model developed through a comprehensive study of contemporary fraud detection techniques. The investigation revealed the limitations of existing methods in promptly detecting fraud, particularly when faced with class imbalance and diverse datasets.

### Research Insights:

The study compared various fraud detection techniques, noting their performance on specific datasets and features:

- Support Vector Machines (SVM) outperformed Logistic Regression in handling class imbalance.
- Random Forest excelled, showcasing superior performance among the two.
- Decision Trees and SVM performed well on raw, unsampled data.
- Artificial Neural Networks (ANN) and Bayesian Belief Network achieved high accuracy and detection rates, albeit with intensive training requirements.
- SVM and K-Nearest Neighbors (KNN) excelled with small datasets but showed limitations with larger ones.

### Proposed Technology:

The proposed fraud detection technology aims for consistent precision and accuracy across diverse scenarios and datasets. It amalgamates existing techniques to overcome individual limitations.

### Model Introduction:

The research paper introduces a novel model centered on average accuracy for credit card fraud detection. The model combines K-Nearest Neighbor, neural networks, and decision trees, assigning a new label through majority voting for each incoming transaction. The accompanying image illustrates the model's versatility across various dataset sizes and types.

# Key Insights:

- Achieves substantial accuracy improvements.
- Overcomes individual model limitations.
- Designed for practicality across diverse scenarios.

This repository provides the implementation of this advanced fraud detection model, offering a robust solution for contemporary fraud detection challenges. Feel free to explore, contribute, and adapt the model to your specific needs.
