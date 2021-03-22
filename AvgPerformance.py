from os import walk
import numpy as np
import pandas as pd

master_folder = 'Data/Final_results/AlloutputFiles'
_, folders, _ = next(walk(master_folder))

for folder in folders:
    _, _, files = next(walk(f'{master_folder}/{folder}'))
    five_iter_npy_lst = []
    five_iter_rf_lst = []
    five_iter_bag_lst = []
    five_iter_xgb_lst = []
    five_iter_logr_lst = []
    five_iter_nn_lst = []
    for idx, file in enumerate(files):
        five_iter_npy_lst.append('npy'+str(idx))
        five_iter_rf_lst.append('rf' + str(idx))
        five_iter_bag_lst.append('bag' + str(idx))
        five_iter_xgb_lst.append('xgb' + str(idx))
        five_iter_logr_lst.append('logr' + str(idx))
        five_iter_nn_lst.append('nn' + str(idx))

        five_iter_npy_lst[idx] = np.load(f'{master_folder}/{folder}/{file}', allow_pickle='TRUE').item()
        five_iter_rf_lst[idx] = pd.DataFrame.from_dict(five_iter_npy_lst[idx]['cv_results_Random Forest']).loc[:, 'split0_test_gmean':]
        five_iter_bag_lst[idx] = pd.DataFrame.from_dict(five_iter_npy_lst[idx]['cv_results_Bagging']).loc[:, 'split0_test_gmean':]
        five_iter_xgb_lst[idx] = pd.DataFrame.from_dict(five_iter_npy_lst[idx]['cv_results_Boosting']).loc[:, 'split0_test_gmean':]
        five_iter_logr_lst[idx] = pd.DataFrame.from_dict(five_iter_npy_lst[idx]['cv_results_Logistic Regression']).loc[:, 'split0_test_gmean':]
        five_iter_nn_lst[idx] = pd.DataFrame.from_dict(five_iter_npy_lst[idx]['cv_results_Neural Network']).loc[:, 'split0_test_gmean':]

    rf_avg = pd.concat([five_iter_rf_lst[0].stack(), five_iter_rf_lst[1].stack(), five_iter_rf_lst[2].stack(),
                        five_iter_rf_lst[3].stack(), five_iter_rf_lst[4].stack()], axis=1).apply(lambda x: x.mean(),
                                                                                                 axis=1).unstack()
    bag_avg = pd.concat([five_iter_bag_lst[0].stack(), five_iter_bag_lst[1].stack(), five_iter_bag_lst[1].stack(),
                         five_iter_bag_lst[3].stack(), five_iter_bag_lst[4].stack()], axis=1).apply(lambda x: x.mean(),
                                                                                                    axis=1).unstack()
    xgb_avg = pd.concat([five_iter_xgb_lst[0].stack(), five_iter_xgb_lst[1].stack(), five_iter_xgb_lst[2].stack(),
                         five_iter_xgb_lst[3].stack(), five_iter_xgb_lst[4].stack()], axis=1).apply(lambda x: x.mean(),
                                                                                                    axis=1).unstack()
    logr_avg = pd.concat([five_iter_logr_lst[0].stack(), five_iter_logr_lst[1].stack(), five_iter_logr_lst[2].stack(),
                          five_iter_logr_lst[3].stack(), five_iter_logr_lst[4].stack()], axis=1).\
        apply(lambda x: x.mean(), axis=1).unstack()
    nn_avg = pd.concat([five_iter_nn_lst[0].stack(), five_iter_nn_lst[1].stack(), five_iter_nn_lst[2].stack(),
                        five_iter_nn_lst[3].stack(), five_iter_nn_lst[4].stack()], axis=1).apply(lambda x: x.mean(),
                                                                                                 axis=1).unstack()
    rf_avg.to_csv(f'{master_folder}/{folder}/{folder}_rf_avg.csv', sep='\t', index=False)
    bag_avg.to_csv(f'{master_folder}/{folder}/{folder}_bag_avg.csv', sep='\t', index=False)
    xgb_avg.to_csv(f'{master_folder}/{folder}/{folder}_xgb_avg.csv', sep='\t', index=False)
    logr_avg.to_csv(f'{master_folder}/{folder}/{folder}_logr_avg.csv', sep='\t', index=False)
    nn_avg.to_csv(f'{master_folder}/{folder}/{folder}_nn_avg.csv', sep='\t', index=False)
