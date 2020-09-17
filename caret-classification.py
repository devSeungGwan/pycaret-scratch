from pycaret.datasets import get_data
from pycaret.classification import *

diabates = get_data('diabetes')
clf1 = setup(data=diabates, target='Class variable')
best = compare_models()
top3 = compare_models(n_select=3)
best = compare_models(sort='AUC')
best_specific = compare_models(whitelist=['dt','rf','xgboost'])
best_specific = compare_models(blacklist=['catboost', 'svm'])