from hyperopt import hp, fmin, tpe
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, average_precision_score, roc_auc_score, precision, recall, accuracy
import time
import numpy as np

def hyperopt_rf(X_train, y_train, metric='average_precision'):
    # Define the search space for RandomForestClassifier hyperparameters
    space_rf = {'max_depth': hp.randint('max_depth', 3, 10),
                'max_features': hp.uniform('max_features', 0.1, 1),
                'n_estimators': hp.randint('n_estimators', 80, 150),
                'min_samples_split': hp.uniform('min_samples_split', 0.1, 1),
                'min_samples_leaf': hp.uniform('min_samples_leaf', 0.1, 0.5),
                'bootstrap': hp.choice('bootstrap', [True, False])}

    # Define a function to be minimized (cross-validation score)
    def rf_cl_bo(params):
        params = {'max_depth': params['max_depth'],
                  'max_features': params['max_features'],
                  'n_estimators': params['n_estimators'],
                  'min_samples_split': params['min_samples_split'],
                  'min_samples_leaf': params['min_samples_leaf'],
                  'bootstrap': params['bootstrap']}
        rf_bo = RandomForestClassifier(random_state=42, **params)
        

        scoring_function = make_scorer(metric, greater_is_better=True)

        # Use stratified k-fold cross-validation to preserve the class distribution
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Perform 5-fold cross-validation and return the negative score
        score = cross_val_score(rf_bo, X_train, y_train, cv=skf, scoring=scoring_function).mean()

        return 1 - score

    # Run Bayesian Optimization for RandomForestClassifier
    start = time.time()
    rf_best_param = fmin(fn=rf_cl_bo,
                         space=space_rf,
                         max_evals=24,
                         rstate=np.random.seed(42),
                         algo=tpe.suggest)
    print('It takes %s minutes' % ((time.time() - start)/60))
    print(f"Best Hyperparameters for RandomForestClassifier (metric={metric}):", rf_best_param)

    return rf_best_param

# Example usage:
# X_train and y_train should be your training data and labels
# best_params = hyperopt_rf(X_train, y_train, metric='average_precision')



from hyperopt import fmin, tpe, hp
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, average_precision_score
import numpy as np

def hyperopt_knn(X_train, y_train, metric):
    # Define the objective function to minimize (negative AUPRC)
    def objective(params):
        # Convert the number of neighbors to an integer (as it should be an integer)
        params['n_neighbors'] = int(params['n_neighbors'])
        
        # Create a KNN classifier with the specified hyperparameters
        knn = KNeighborsClassifier(**params)
        
        # Use stratified k-fold cross-validation to preserve the class distribution
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Perform 5-fold cross-validation and return the negative AUPRC
        auprc = cross_val_score(knn, X_train, y_train, cv=skf, scoring=make_scorer(metric)).mean()
        
        return -auprc

    # Define the search space for hyperparameters
    space = {
        'n_neighbors': hp.quniform('n_neighbors', 1, 20, 1),  # Number of neighbors (integer)
        'weights': hp.choice('weights', ['uniform', 'distance']),  # Weighting scheme
        'p': hp.choice('p', [1, 2]),  # Power parameter for Minkowski distance
    }

    # Perform Bayesian hyperparameter optimization
    start = time.time()
    best = fmin(fn=objective,  # Objective function to minimize
                space=space,     # Search space
                algo=tpe.suggest,  # Optimization algorithm (Tree-structured Parzen Estimator)
                max_evals=20)     # Number of optimization trials
    
    print('It takes %s minutes' % ((time.time() - start)/60))
    # Print the best hyperparameters found
    print(f"Best Hyperparameters for KNeighborsClassifier (metric={metric}):", best)

    return best

# Example usage:
# X_train and y_train should be your training data and labels
# best_params_knn = hyperopt_knn(X_train, y_train, metric=precision)



# import numpy as np
# from hyperopt import fmin, tpe, hp

# # Assuming you have defined the Classifier, loss_batch, and train functions elsewhere in your code
# # Make sure to replace them with your actual implementation

# # Define your hyperparameter search space
# space = {
#     'n_hidden': hp.quniform('n_hidden', 5, 50, 1),
#     'drop_prob': hp.uniform('drop_prob', 0.1, 0.9),
#     'lr': hp.loguniform('lr', np.log(1e-5), np.log(1e-1))
# }

# # Define your objective function
# def objective(params):
#     n_hidden = int(params['n_hidden'])
#     drop_prob = params['drop_prob']
#     lr = params['lr']

#     # Replace with your Classifier instantiation and training code
#     model = Classifier(n_input=n_input, n_hidden=n_hidden, n_output=n_output, drop_prob=drop_prob)

#     pos_weight = torch.tensor([5])
#     opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
#     loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#     n_epoch = 10  # Adjust the number of epochs based on your preference

#     for epoch in range(n_epoch):
#         model.train()
#         for xb, yb in train_dl:
#             loss_batch(model, loss_func, xb, yb, opt)

#         model.eval()
#         with torch.no_grad():
#             losses, nums = zip(
#                 *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
#             )
#         val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

#     return val_loss

# # Function to perform hyperparameter optimization and return the best hyperparameters
# def optimize_hyperparameters(X_train, y_train, metric):
#     global n_input, n_output, train_dl, valid_dl  # Assuming these are defined somewhere in your code
    
#     # Set your data and other necessary parameters
#     n_input = X_train.shape[1]
#     n_output = 1  # Adjust based on your problem (binary classification assumed)
    
#     # Split your data into training and validation sets (train_dl, valid_dl)
#     # Make sure to replace these lines with your actual data splitting logic
    
#     # Run the hyperparameter optimization
#     best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50)

#     # Get the best hyperparameters
#     best_n_hidden = int(best['n_hidden'])
#     best_drop_prob = best['drop_prob']
#     best_lr = best['lr']

#     # Print the best hyperparameters
#     print(f"Best hyperparameters: n_hidden={best_n_hidden}, drop_prob={best_drop_prob}, lr={best_lr}")

#     return best_n_hidden, best_drop_prob, best_lr

# # # Example usage
# # best_n_hidden, best_drop_prob, best_lr = optimize_hyperparameters(X_train, y_train, 'precision')

# # # Train the model with the best hyperparameters
# # best_model = Classifier(n_input=n_input, n_hidden=best_n_hidden, n_output=n_output, drop_prob=best_drop_prob)
# # best_opt = torch.optim.SGD(best_model.parameters(), lr=best_lr, momentum=0.9)
# # best_loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# # n_epoch = 200  # Adjust the number of epochs based on your preference
# # train(n_epoch, best_model, best_loss_func, best_opt, train_dl, valid_dl)

