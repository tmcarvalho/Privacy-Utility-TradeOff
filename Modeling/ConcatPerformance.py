import os
import shutil
from os import walk
import numpy as np
import pandas as pd

master_folder_transf = 'Data/Final_results/AlloutputFiles'
_, folders_transf, _ = next(walk(master_folder_transf))

# split files into original and transformed folders
for folder in folders_transf:
    _, _, files = next(walk(f'{master_folder_transf}/{folder}'))
    if "_" not in folder:
        shutil.move(os.path.join(f'{master_folder_transf}/{folder}'), f'{master_folder_transf}/Originals')
    else:
        shutil.move(os.path.join(f'{master_folder_transf}/{folder}'), f'{master_folder_transf}/Transformed')

# percentage difference
orig_folder = 'Data/Final_results/AlloutputFiles/Originals'
transf_folder = 'Data/Final_results/AlloutputFiles/Transformed'
_, org_folders, _ = next(walk(orig_folder))
_, transf_folders, _ = next(walk(transf_folder))


def concatenate(flag, master_folder, folder):
    # concatenate all 5 repetitions
    five_iter_npy_lst = []
    five_iter_rf_lst = []
    five_iter_bag_lst = []
    five_iter_xgb_lst = []
    five_iter_logr_lst = []
    five_iter_nn_lst = []

    _, _, files = next(walk(f'{master_folder}/{folder}'))

    for idx, file in enumerate(files):
        five_iter_npy_lst.append('npy' + str(idx))
        five_iter_rf_lst.append('rf' + str(idx))
        five_iter_bag_lst.append('bag' + str(idx))
        five_iter_xgb_lst.append('xgb' + str(idx))
        five_iter_logr_lst.append('logr' + str(idx))
        five_iter_nn_lst.append('nn' + str(idx))

        five_iter_npy_lst[idx] = np.load(f'{master_folder}/{folder}/{file}', allow_pickle='TRUE').item()
        five_iter_rf_lst[idx] = pd.DataFrame.from_dict(five_iter_npy_lst[idx]['cv_results_Random Forest'])
        five_iter_bag_lst[idx] = pd.DataFrame.from_dict(five_iter_npy_lst[idx]['cv_results_Bagging'])
        five_iter_xgb_lst[idx] = pd.DataFrame.from_dict(five_iter_npy_lst[idx]['cv_results_Boosting'])
        five_iter_logr_lst[idx] = pd.DataFrame.from_dict(
            five_iter_npy_lst[idx]['cv_results_Logistic Regression'])
        five_iter_nn_lst[idx] = pd.DataFrame.from_dict(five_iter_npy_lst[idx]['cv_results_Neural Network'])

    rf_avg = pd.concat([df for df in five_iter_rf_lst])
    bag_avg = pd.concat([df for df in five_iter_bag_lst])
    xgb_avg = pd.concat([df for df in five_iter_xgb_lst])
    logr_avg = pd.concat([df for df in five_iter_logr_lst])
    nn_avg = pd.concat([df for df in five_iter_nn_lst])

    if flag == 0:
        rf_avg.to_csv(f'{transf_folder}/{t_folder}/{t_folder}_rf_avg.csv', sep='\t', index=False)
        bag_avg.to_csv(f'{transf_folder}/{t_folder}/{t_folder}_bag_avg.csv', sep='\t', index=False)
        xgb_avg.to_csv(f'{transf_folder}/{t_folder}/{t_folder}_xgb_avg.csv', sep='\t', index=False)
        logr_avg.to_csv(f'{transf_folder}/{t_folder}/{t_folder}_logr_avg.csv', sep='\t', index=False)
        nn_avg.to_csv(f'{transf_folder}/{t_folder}/{t_folder}_nn_avg.csv', sep='\t', index=False)

    else:
        return rf_avg, bag_avg, xgb_avg, logr_avg, nn_avg


# split files into original and transformed folders
metrics = ['gmean', 'acc', 'bal_acc', 'f1', 'f1_weighted']
for t_folder in transf_folders:
    for o_folder in org_folders:
        if str(o_folder + '_') in t_folder:
            o_rf, o_bag, o_xgb, o_logr, o_nn = concatenate(1, orig_folder, o_folder)
            t_rf, t_bag, t_xgb, t_logr, t_nn = concatenate(1, transf_folder, t_folder)

            # for each CV calculate the percentage difference
            for metric in metrics:
                # 100 * (Sc - Sb) / Sb
                t_rf['mean_test_' + metric + '_perdif'] = 100 * (t_rf['mean_test_' + metric] - o_rf[
                    'mean_test_' + metric]) / o_rf['mean_test_' + metric]

                t_bag['mean_test_' + metric + '_perdif'] = 100 * (t_bag['mean_test_' + metric] - o_bag[
                    'mean_test_' + metric]) / o_bag['mean_test_' + metric]

                t_xgb['mean_test_' + metric + '_perdif'] = 100 * (t_xgb['mean_test_' + metric] - o_xgb[
                    'mean_test_' + metric]) / o_xgb['mean_test_' + metric]

                t_logr['mean_test_' + metric + '_perdif'] = 100 * (t_logr['mean_test_' + metric] - o_logr[
                    'mean_test_' + metric]) / o_logr['mean_test_' + metric]

                t_nn['mean_test_' + metric + '_perdif'] = 100 * (t_nn['mean_test_' + metric] - o_nn[
                    'mean_test_' + metric]) / o_nn['mean_test_' + metric]

            t_rf.to_csv(f'{transf_folder}/{t_folder}/{t_folder}_rf_avg.csv', sep='\t', index=False)
            t_bag.to_csv(f'{transf_folder}/{t_folder}/{t_folder}_bag_avg.csv', sep='\t', index=False)
            t_xgb.to_csv(f'{transf_folder}/{t_folder}/{t_folder}_xgb_avg.csv', sep='\t', index=False)
            t_logr.to_csv(f'{transf_folder}/{t_folder}/{t_folder}_logr_avg.csv', sep='\t', index=False)
            t_nn.to_csv(f'{transf_folder}/{t_folder}/{t_folder}_nn_avg.csv', sep='\t', index=False)


# concatenate the 5 iterations of original datasets
for folder in org_folders:
    concatenate(0, orig_folder, folder)

# remove csv files
# _, transf_folders, _ = next(walk(transf_folder))
# for t_folder in transf_folders:
#     _, _, files = next(walk(f'{transf_folder}/{t_folder}'))
#     for file in files:
#         if file.endswith(".csv"):
#             os.remove(os.path.join(f'{transf_folder}/{t_folder}', file))
