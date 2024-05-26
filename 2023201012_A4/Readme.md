
### Dataset:
* Divided the dataset into train and validation split (90:10).
### Model Training and Prediction:
#### Part-1 (Transfer learning):
* Finetuned ResNet34 model for age prediction, modifying its last layer and utilized L1 loss. It iterates over epochs(30), updating weights based on gradients computed from training data, and saves the model with the lowest validation loss.
* Then used this best model to predict the ages for the test dataset, and the results are saved into a CSV file.
#### Part-2 (SVR):
* It first adjusts the ResNet34 model by removing its last fully connected layer, preparing it for feature extraction. 
* It then loads the saved model's state dictionary and extracts features from the training dataset using this modified model. 
* These features are fed into an SVR (Support Vector Regressor) model, trained to predict age based on the extracted features.
* Finally, the SVR model is used to predict ages for the test dataset, and the results are saved into a CSV file.

#### Ensemble Methods:
* Combined predictions from above two baseline models (one using direct ResNet predictions and the other using SVR on ResNet features) by averaging their age predictions.
* Then saved the ensemble result into a CSV file named 'final_result.csv', effectively leveraging ensemble methods to improve prediction accuracy
