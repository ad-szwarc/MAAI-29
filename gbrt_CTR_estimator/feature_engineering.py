import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse import hstack
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer


train_set = pd.read_csv("train.csv")
validation_set = pd.read_csv("validation.csv")
test_set = pd.read_csv('test.csv')


###################
# Training Set
###################

# one-hot encoding for categorical features
useragent_lb_t = LabelBinarizer()
slotvisibility_lb_t = LabelBinarizer()
slotformat_lb_t = LabelBinarizer()

useragent_1hot_t = useragent_lb_t.fit_transform(train_set.useragent.values)
slotvisibility_1hot_t = slotvisibility_lb_t.fit_transform(train_set.slotvisibility.values)
slotformat_1hot_t = slotformat_lb_t.fit_transform(train_set.slotformat.values)


# one-hot encoding for numerical features
weekday_ohe_t = OneHotEncoder()
hour_ohe_t = OneHotEncoder()
region_ohe_t = OneHotEncoder()
city_ohe_t = OneHotEncoder()
slotwidth_ohe_t = OneHotEncoder()
slotheight_ohe_t = OneHotEncoder()
slotprice_ohe_t = OneHotEncoder()
advertiser_ohe_t = OneHotEncoder()
adexchange_ohe_t = OneHotEncoder()
usertag_num_ohe_t = OneHotEncoder()
hour_avgCTR_ohe_t = OneHotEncoder()

weekday_1hot_t = weekday_ohe_t.fit_transform(train_set.weekday.values.reshape(-1,1)).toarray()
hour_1hot_t = hour_ohe_t.fit_transform(train_set.hour.values.reshape(-1,1)).toarray()
region_1hot_t = region_ohe_t.fit_transform(train_set.region.values.reshape(-1,1)).toarray()
city_1hot_t = city_ohe_t.fit_transform(train_set.city.values.reshape(-1,1)).toarray()
slotwidth_1hot_t = slotwidth_ohe_t.fit_transform(train_set.slotwidth.values.reshape(-1,1)).toarray()
slotheight_1hot_t = slotheight_ohe_t.fit_transform(train_set.slotheight.values.reshape(-1,1)).toarray()
slotprice_1hot_t = slotprice_ohe_t.fit_transform(train_set.slotprice.values.reshape(-1,1)).toarray()
advertiser_1hot_t = advertiser_ohe_t.fit_transform(train_set.advertiser.values.reshape(-1,1)).toarray()


# replace 'nan's with 0s in 'adexchange' column
train_set.adexchange.fillna(0, inplace=True)
adexchange_1hot_t = adexchange_ohe_t.fit_transform(train_set.adexchange.values.reshape(-1,1)).toarray()


# 1-hot encoding for usertag column
train_set.usertag.fillna('0', inplace=True)
usertag_1hot_t = train_set.usertag.str.get_dummies(',')
usertag_1hot_t.to_csv('usertag_1hot_t.csv', index=False)

usertag_1hot_t = pd.read_csv('usertag_1hot_t.csv')


# number of tags feature
train_set.usertag.fillna('0', inplace=True)
usertag_list_t = [x.split(',') for x in train_set.usertag]
usertag_num_t = [len(x) for x in usertag_list_t]
usertag_num_t = np.asarray(usertag_num_t)
usertag_num_1hot_t = usertag_num_ohe_t.fit_transform(usertag_num_t.reshape(-1,1)).toarray()


# compute avgCTR for each group in 'usertag_num' feature
train_copy = train_set.copy()
train_copy['usertag_num'] = usertag_num_t

def get_avgCTR(group):
    clicks = len(group[group.click == 1])
    avgCTR = clicks/len(group)
    group['avgCTR'] = avgCTR
    return group

train_copy = train_copy.groupby('usertag_num').apply(get_avgCTR)
usertag_avgCTR_t = train_copy.avgCTR


# compute avgCTR for each group in the 'hour' column
train_copy = train_set.copy()

def get_avgCTR(group):
    clicks = len(group[group.click == 1])
    avgCTR = clicks/len(group)
    group['avgCTR'] = avgCTR
    return group

train_copy = train_copy.groupby('hour').apply(get_avgCTR)
hour_avgCTR_t = 10000000 * train_copy.avgCTR
hour_avgCTR_1hot_t = hour_avgCTR_ohe_t.fit_transform(hour_avgCTR_t.values.reshape(-1,1)).toarray()


# convert to sparse matrices
hour_avgCTR_s_t = sparse.csr_matrix(hour_avgCTR_1hot_t)
usertag_num_s_t = sparse.csr_matrix(usertag_num_1hot_t)
usertag_s_t = sparse.csr_matrix(usertag_1hot_t)
city_s_t = sparse.csr_matrix(city_1hot_t)
hour_s_t = sparse.csr_matrix(hour_1hot_t)
slotprice_s_t = sparse.csr_matrix(slotprice_1hot_t)
useragent_s_t = sparse.csr_matrix(useragent_1hot_t)
weekday_s_t = sparse.csr_matrix(weekday_1hot_t)
slotwidth_s_t = sparse.csr_matrix(slotwidth_1hot_t)
slotvisibility_s_t = sparse.csr_matrix(slotvisibility_1hot_t)
advertiser_s_t = sparse.csr_matrix(advertiser_1hot_t)
region_s_t = sparse.csr_matrix(region_1hot_t)

features_t = hstack([hour_avgCTR_s_t, usertag_num_s_t, usertag_s_t, city_s_t, hour_s_t,
                     weekday_s_t, slotwidth_s_t, slotvisibility_s_t, advertiser_s_t])

sparse.save_npz("features_t.npz", features_t)




###################
# Validation Set
###################

# one-hot encoding for categorical features
useragent_lb_v = LabelBinarizer()
slotvisibility_lb_v = LabelBinarizer()
slotformat_lb_v = LabelBinarizer()

useragent_1hot_v = useragent_lb_v.fit_transform(validation_set.useragent.values)
slotvisibility_1hot_v = slotvisibility_lb_v.fit_transform(validation_set.slotvisibility.values)
slotformat_1hot_v = slotformat_lb_v.fit_transform(validation_set.slotformat.values)


# one-hot encoding for numerical features
weekday_ohe_v = OneHotEncoder()
hour_ohe_v = OneHotEncoder()
region_ohe_v = OneHotEncoder()
city_ohe_v = OneHotEncoder()
slotwidth_ohe_v = OneHotEncoder()
slotheight_ohe_v = OneHotEncoder()
slotprice_ohe_v = OneHotEncoder()
advertiser_ohe_v = OneHotEncoder()
adexchange_ohe_v = OneHotEncoder()
usertag_num_ohe_v = OneHotEncoder()
hour_avgCTR_ohe_v = OneHotEncoder()

weekday_1hot_v = weekday_ohe_v.fit_transform(validation_set.weekday.values.reshape(-1,1)).toarray()
hour_1hot_v = hour_ohe_v.fit_transform(validation_set.hour.values.reshape(-1,1)).toarray()
region_1hot_v = region_ohe_v.fit_transform(validation_set.region.values.reshape(-1,1)).toarray()
city_1hot_v = city_ohe_v.fit_transform(validation_set.city.values.reshape(-1,1)).toarray()
slotwidth_1hot_v = slotwidth_ohe_v.fit_transform(validation_set.slotwidth.values.reshape(-1,1)).toarray()
slotheight_1hot_v = slotheight_ohe_v.fit_transform(validation_set.slotheight.values.reshape(-1,1)).toarray()
slotprice_1hot_v = slotprice_ohe_v.fit_transform(validation_set.slotprice.values.reshape(-1,1)).toarray()
advertiser_1hot_v = advertiser_ohe_v.fit_transform(validation_set.advertiser.values.reshape(-1,1)).toarray()


# replace 'nan's with 0s in 'adexchange' column
validation_set.adexchange.fillna(0, inplace=True)
adexchange_1hot_v = adexchange_ohe_v.fit_transform(validation_set.adexchange.values.reshape(-1,1)).toarray()


# 1-hot encoding for usertag column
validation_set.usertag.fillna('0', inplace=True)
usertag_1hot_v = validation_set.usertag.str.get_dummies(',')
usertag_1hot_v.to_csv('usertag_1hot_v.csv', index=False)

usertag_1hot_v = pd.read_csv('usertag_1hot_v.csv')


# number of usertags feature
validation_set.usertag.fillna('0', inplace=True)
usertag_list_v = [x.split(',') for x in validation_set.usertag]
usertag_num_v = [len(x) for x in usertag_list_v]
usertag_num_v = np.asarray(usertag_num_v)
usertag_num_1hot_v = usertag_num_ohe_v.fit_transform(usertag_num_v.reshape(-1,1)).toarray()
# add extra column to match training set
zeros = np.zeros((usertag_num_1hot_v.shape[0], 1))
usertag_num_1hot_v = np.append(usertag_num_1hot_v, zeros, axis=1)


# compute avgCTR for each group in 'usertag_num' feature
val_copy = validation_set.copy()
val_copy['usertag_num'] = usertag_num_v

def get_avgCTR(group):
    clicks = len(group[group.click == 1])
    avgCTR = clicks/len(group)
    group['avgCTR'] = avgCTR
    return group

val_copy = val_copy.groupby('usertag_num').apply(get_avgCTR)
usertag_avgCTR_v = val_copy.avgCTR


# compute avgCTR for each group in the 'hour' column
val_copy = validation_set.copy()

def get_avgCTR(group):
    clicks = len(group[group.click == 1])
    avgCTR = clicks/len(group)
    group['avgCTR'] = avgCTR
    return group

val_copy = val_copy.groupby('hour').apply(get_avgCTR)
hour_avgCTR_v = 10000000 * val_copy.avgCTR
hour_avgCTR_1hot_v = hour_avgCTR_ohe_v.fit_transform(hour_avgCTR_v.values.reshape(-1,1)).toarray()



# convert to sparse matrices
hour_avgCTR_s_v = sparse.csr_matrix(hour_avgCTR_1hot_v)
usertag_num_s_v = sparse.csr_matrix(usertag_num_1hot_v)
usertag_s_v = sparse.csr_matrix(usertag_1hot_v)
city_s_v = sparse.csr_matrix(city_1hot_v)
hour_s_v = sparse.csr_matrix(hour_1hot_v)
slotprice_s_v = sparse.csr_matrix(slotprice_1hot_v)
useragent_s_v = sparse.csr_matrix(useragent_1hot_v)
weekday_s_v = sparse.csr_matrix(weekday_1hot_v)
slotwidth_s_v = sparse.csr_matrix(slotwidth_1hot_v)
slotvisibility_s_v = sparse.csr_matrix(slotvisibility_1hot_v)
advertiser_s_v = sparse.csr_matrix(advertiser_1hot_v)
region_s_v = sparse.csr_matrix(region_1hot_v)


features_v = hstack([hour_avgCTR_s_v, usertag_num_s_v, usertag_s_v, city_s_v, hour_s_v,
                     weekday_s_v, slotwidth_s_v, slotvisibility_s_v, advertiser_s_v])

sparse.save_npz("features_v.npz", features_v)




###################
# Test Set
###################

# one-hot encoding for categorical features
useragent_lb_tt = LabelBinarizer()
slotvisibility_lb_tt = LabelBinarizer()
slotformat_lb_tt = LabelBinarizer()

useragent_1hot_tt = useragent_lb_tt.fit_transform(test_set.useragent.values)
slotvisibility_1hot_tt = slotvisibility_lb_tt.fit_transform(test_set.slotvisibility.values)
slotformat_1hot_tt = slotformat_lb_tt.fit_transform(test_set.slotformat.values)


# one-hot encoding for numerical features
weekday_ohe_tt = OneHotEncoder()
hour_ohe_tt = OneHotEncoder()
region_ohe_tt = OneHotEncoder()
city_ohe_tt = OneHotEncoder()
slotwidth_ohe_tt = OneHotEncoder()
slotheight_ohe_tt = OneHotEncoder()
slotprice_ohe_tt = OneHotEncoder()
advertiser_ohe_tt = OneHotEncoder()
adexchange_ohe_tt = OneHotEncoder()

weekday_1hot_tt = weekday_ohe_tt.fit_transform(test_set.weekday.values.reshape(-1,1)).toarray()
hour_1hot_tt = hour_ohe_tt.fit_transform(test_set.hour.values.reshape(-1,1)).toarray()
region_1hot_tt = region_ohe_tt.fit_transform(test_set.region.values.reshape(-1,1)).toarray()
city_1hot_tt = city_ohe_tt.fit_transform(test_set.city.values.reshape(-1,1)).toarray()
slotwidth_1hot_tt = slotwidth_ohe_tt.fit_transform(test_set.slotwidth.values.reshape(-1,1)).toarray()
slotheight_1hot_tt = slotheight_ohe_tt.fit_transform(test_set.slotheight.values.reshape(-1,1)).toarray()
slotprice_1hot_tt = slotprice_ohe_tt.fit_transform(test_set.slotprice.values.reshape(-1,1)).toarray()
advertiser_1hot_tt = advertiser_ohe_tt.fit_transform(test_set.advertiser.values.reshape(-1,1)).toarray()


# replace 'nan's with 0s in 'adexchange' column
test_set.adexchange.fillna(0, inplace=True)
adexchange_1hot_tt = adexchange_ohe_tt.fit_transform(test_set.adexchange.values.reshape(-1,1)).toarray()


# 1-hot encoding for usertag column
test_set.usertag.fillna('0', inplace=True)
usertag_1hot_tt = test_set.usertag.str.get_dummies(',')
usertag_1hot_tt.to_csv('usertag_1hot_tt.csv', index=False)


# convert to sparse matrices
usertag_s_tt = sparse.csr_matrix(usertag_1hot_tt)
city_s_tt = sparse.csr_matrix(city_1hot_tt)
hour_s_tt = sparse.csr_matrix(hour_1hot_tt)
slotprice_s_tt = sparse.csr_matrix(slotprice_1hot_tt)
useragent_s_tt = sparse.csr_matrix(useragent_1hot_tt)
weekday_s_tt = sparse.csr_matrix(weekday_1hot_tt)
slotwidth_s_tt = sparse.csr_matrix(slotwidth_1hot_tt)
slotvisibility_s_tt = sparse.csr_matrix(slotvisibility_1hot_tt)
advertiser_s_tt = sparse.csr_matrix(advertiser_1hot_tt)
region_s_tt = sparse.csr_matrix(region_1hot_tt)


features_tt = hstack([usertag_s_tt, city_s_tt, hour_s_tt,
                     weekday_s_tt, slotwidth_s_tt, slotvisibility_s_tt, advertiser_s_tt])

sparse.save_npz("features_tt.npz", features_tt)
