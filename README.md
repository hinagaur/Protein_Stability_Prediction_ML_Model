						
# Protein Stability Prediction ML Model
To predict thermostability of proteins, which is a critical trait that enhances various application specific functions in the realm of computational biology, I did a thorough literature search and found two promising ML models for predicting that are : XGBoost [1] and CNN [2]. I tried various other models like SVM and random forest, however their performance was pretty low compared to the two. Overall, in my case XGBoost outperformed CNN so I went ahead with it.
						
#### Data Processing and Feature Engineering:
						
For data processing, I performed one hot encoding of the amino acid sequences to represent it as a binary vector so that it can be easily interpreted by my machine learning algorithm.
						
I also did some literature search and found that certain amino acids, such as glycine, lysine, tyrosine, and isoleucine are found in a higher concentration in organisms which are thermophilic. Hence, I integrated a function to calculate GKYI content in the sequences and added this feature in the data. I further scaled the GKYI content.
						
#### Model Architecture and choice:
						
XGBOOST- which stands for Extreme Gradient Boosting, is a scalable, distributed gradient boosted decision tree (GBDT) machine learning library. I found that it performed really well (second best performing)[3] through literature search. It is a powerful and flexible model suitable for regression tasks, particularly when dealing with structured data like sequences of amino acids. For this model, I used hyperparameters such as max_depth, which controls the maximum depth of each tree in the ensemble, learning_rate, determines the step size at each iteration, and n_estimators, which sets the number of trees in the ensemble, and optimized using Randomized Search Cross-Validation.
						
CNN: It comprises two convolutional layers with 128 and 256 filters respectively, followed by max-pooling layers to capture local and higher-level features in the input sequences. A dense layer with 128 neurons and ReLU activation is utilized for learning complex representations, augmented by dropout regularization to mitigate overfitting. The output layer predicts a single continuous value without activation, suitable for regression tasks. The Adam optimizer is employed with a learning rate of 0.001, and mean squared error (MSE) is chosen as the loss function. Early stopping with a patience of 3 monitors validation loss to prevent overfitting.
									
#### ML Model Training and hyperparameter selection	
						
I did 80:20 conventional train test split, which means that 80% of the data is used for training the model, while the remaining 20% is reserved for testing its performance. For XGBoost, hyperparameters such as max_depth, learning_rate, and n_estimators are optimized using Randomized Search Cross-Validation. The max depth in the parameter grid includes values such as 3, 6, 9, and 12, the grid search explores a range of tree depths, from shallow to deep, to find the optimal balance between model complexity and generalization performance. The learning grid contains values like 0.1, 0.01, 0.001, and 0.0001, the grid search explores a wide range of learning rates to find the optimal balance between training speed and model performance. The n_estimators grid contains values such as 20, 30, 40, and 60, the grid search explores a range of ensemble sizes to find the optimal number of trees that balances model complexity and performance. Finally, the most optimum hyperparameters found were n_estimators': 60, 'max_depth': 12, 'learning_rate': 0.01. Hyperparameter tuning increased the R - squared from 0.16 to 0.29, which is slightly better than without any hyperparameter tuning.
						
I also used parallel processing with threading in Randomized Search Cross Validation to expedite the hyperparameter search process. This leverages the computational power efficiently, especially for large datasets and complex hyperparameter grids.
						
For CNN, along with the hyperparameters mentioned in the Model Architecture and choice section, I also used EarlyStopping callback plays a crucial role in training neural network models effectively by preventing overfitting, improving generalization, and optimizing resource utilization.
						
#### Results
						
In addition to the metric on kaggle, which is spearman correlation, I utilized the following metrics-
						
Mean Squared error (RME) also known as Quadratic loss, as the penalty is not proportional to the error but to the square of the error. A good model will have an MSE value closer to zero.
						
Root mean squared error(RMSE) measures the average magnitude of the errors and is concerned with the deviations from the actual value. RMSE value with zero indicates that the model has a perfect fit. The lower the RMSE, the better the model and its predictions.						
R-squared (R2)- measures the proportion of the variance in the dependent variable that is predictable from the independent variables. It ranges from 0 to 1, where 1 indicates a perfect fit.
Mean Absolute Error (MAE): calculates the average of the absolute differences between the predicted and actual values. It provides a measure of the average magnitude of errors. A lower MAE indicates superior model accuracy.
												
#### Discussion:
						
This report discusses the models used for predicting the thermostability of proteins, a critical trait that enhances various application-specific functions in the realm of computational biology. Leveraging hyperparameter optimization techniques, XGBoost emerged as the preferred model, showcasing superior performance over CNN across multiple evaluation metrics. Despite the absence of a definitive breakthrough, these findings underscore the critical role of model selection, hyperparameter tuning, and domain knowledge integration in advancing predictive capabilities in computational biology. Continued efforts in literature exploration and feature refinement hold promise for further enhancing the performance and applicability of protein thermostability prediction models.
						
#### References:
						
1- Chen, C. W., Lin, M. H., Liao, C. C., Chang, H. P., & Chu, Y. W. (2020). iStable 2.0: Predicting protein thermal stability changes by integrating various characteristic modules. Computational and structural biotechnology journal, 18, 622–630. https://doi.org/10.1016/j.csbj.2020.02.021
						
2- Fang, X., Huang, J., Zhang, R., Wang, F., Zhang, Q., Li, G., ... Xu, L. (2019). Convolution neural network-based prediction of protein thermostability. Journal of Chemical Information and Modeling, 59(11), 4833–4843. doi:10.1021/acs.jcim.9b00220
						
3- Yang, Y., Zhao, J., Zeng, L., & Vihinen, M. (2022). Protstab2 for prediction of protein thermal stabilities. International Journal of Molecular Sciences, 23(18), 10798. doi:10.3390/ijms231810798 
					
				
			
		

