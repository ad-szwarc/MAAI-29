##############
# Problem 4
##############


# GBRT (non-linear) for CTR estimation

import numpy as np
from scipy import sparse
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


features_t = sparse.load_npz("features_t.npz")
features_v = sparse.load_npz("features_v.npz")

X_train = features_t
y_train = train_set.click.values

X_val = features_v
y_val = validation_set.click.values


# fit model to training data
model = XGBClassifier(n_estimators = 100,
                      max_depth = 14,
                      min_child_weight = 1,
                      subsample = 0.8166888840525918)

model.fit(X_train, y_train)

# make predictions for the validation set
y_proba = model.predict_proba(X_val)

# check AUC
auc = roc_auc_score(y_val, y_proba[:,1])

# save predictions
np.savetxt("val_predictions.csv", y_proba[:,1], delimiter=",")




# Linear bidding strategy with non-linear CTR estimator

budget = 6250

train_num_imp = train_set.shape[0]
clicks_train = len(train_set[train_set['click'] == 1])
avg_ctr = clicks_train/train_num_imp
pCTR = y_proba[:, 1]


def linear_bidding_strategy(data_set, base_bid):
    bids = base_bid * (pCTR/avg_ctr)
    possible_impressions = data_set[data_set['payprice'] <= bids]
    possible_impressions['cost_cumsum'] = possible_impressions['payprice'].cumsum() / 1000
    won_impressions = possible_impressions[possible_impressions['cost_cumsum'] < budget]

    clicks = len(won_impressions[won_impressions['click'] == 1])
    cost = won_impressions['cost_cumsum'].iloc[-1]
    num_imp = won_impressions.shape[0]

    return bids, won_impressions, clicks, cost, num_imp


base_bid_range = range(0, 300)
best_clicks = 0

for bid in base_bid_range:
    bids, won_impressions, clicks, cost, num_imp = linear_bidding_strategy(validation_set, bid)
    if clicks > best_clicks:
        best_clicks = clicks
        best_base_bid = bid


#evaluate on the validation set
bids, won_impressions, clicks, cost, num_imp = linear_bidding_strategy(validation_set, best_base_bid)

ctr = clicks/num_imp
cpm = (cost/num_imp)*1000
cpc = cost/clicks




######################################
# Using Hyperopt For Grid Searching
######################################

from sklearn.metrics import roc_auc_score
import xgboost as xgb
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials


train = sparse.load_npz("features_t.npz")
valid = sparse.load_npz("features_v.npz")

y_train = train_set.click.values
y_valid = validation_set.click.values


def objective(space):

    clf = xgb.XGBClassifier(n_estimators = 100,
                            max_depth = int(space['max_depth']),
                            min_child_weight = space['min_child_weight'],
                            subsample = space['subsample'])

    eval_set  = [(train, y_train), (valid, y_valid)]

    clf.fit(train, y_train,
            eval_set=eval_set, eval_metric="auc",
            early_stopping_rounds=30)

    pred = clf.predict_proba(valid)[:, 1]
    auc = roc_auc_score(y_valid, pred)
    print("SCORE:", auc)

    return{'loss':1-auc, 'status': STATUS_OK }


space = {
         'max_depth': hp.quniform("x_max_depth", 5, 30, 1),
         'min_child_weight': hp.quniform ('x_min_child', 1, 10, 1),
         'subsample': hp.uniform ('x_subsample', 0.8, 1)
        }


trials = Trials()

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=1,
            trials=trials)

print(best)




####################################
# Run CTR Estimation for Test Set
####################################

import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


features_t = sparse.load_npz("features_t.npz")
features_tt = sparse.load_npz("features_tt.npz")

X_train = features_t
y_train = train_set.click.values

X_test = features_tt


# fit model to training data
model = XGBClassifier(n_estimators = 100,
                      max_depth = 14,
                      min_child_weight = 1,
                      subsample = 0.8166888840525918)

model.fit(X_train, y_train)

# make predictions for the test set
y_proba = model.predict_proba(X_test)

# save predictions
np.savetxt("test_predictions.csv", y_proba[:,1], delimiter=",")
