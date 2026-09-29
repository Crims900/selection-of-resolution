# Read ari_1000_list.txt and ari_list.csv, then draw box plots.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time

RESOLUTIONS = [round(float(res), 2) for res in np.linspace(1.03, 1.12, 10)]

base_dir = "/Data/Programs/SpaGCN_stabilization/stabilization_part/"
result_dir = base_dir + "eps_plots/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
index_list = ["MERFISH_0.04","MERFISH_0.09","MERFISH_0.14","MERFISH_0.19","MERFISH_0.24",
              "E9.5_E1S1.MOSTA","E9.5_E2S1.MOSTA","E9.5_E2S2.MOSTA","E9.5_E2S3.MOSTA","E9.5_E2S4.MOSTA",
              "151507", "151508", "151509", "151510", "151669", "151670","151671", "151672", "151673", "151674", "151675","151676",
              "Slice_1","Slice_2","Slice_3","BZ5","BZ14","BZ97","STARmap_BY3_1k"]
for index in index_list:

    time1 = time.time()
    # Read the ari_1000_list.txt file.
    with open(base_dir + index + "/" + 'ari_1000_list.txt', 'r') as f:
        ari_list = f.readlines()
        ari_list = [float(i) for i in ari_list]



    # Use subplots: two-thirds width on the left and one-third on the right.
    plt.figure(figsize=(10, 5))
    grid = plt.GridSpec(1, 3, wspace=0.4, hspace=0.3)
    plt.subplot(grid[0:2])
    # Draw the histogram.
    # plt.hist(ari_list, bins=20, color='skyblue', alpha=0.7)
    plt.hist(ari_list, bins=20,  alpha=0.7, edgecolor='black',)
    plt.xlabel('ARI')
    plt.ylabel('Frequency')
    plt.grid(False)
    plt.subplot(grid[2])
    # Draw the box plot.
    plt.boxplot(ari_list)
    plt.ylabel('ARI')
    # Remove x-axis ticks.
    plt.xticks([])
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.1)
    plt.savefig(result_dir + index + '_ari_1000.eps', dpi=600, format='eps')
    # plt.show()
    plt.close()
    time2 = time.time()
    print(index + " cost time: " + str(time2 - time1))


# Draw ten box plots from ari_list.csv in one row.
for index in index_list:
    # Read the ari_list.csv file.
    ari_list_csv = pd.read_csv(base_dir + index + '/ari_list.csv',index_col=0)
    plt.figure(figsize=(20, 10))
    # plt.boxplot([ari_list_csv.iloc[i] for i in range(5)], patch_artist=True, showmeans=True, meanline=True, showfliers=False)
    plt.boxplot([ari_list_csv.iloc[:, i] for i in range(10)], showfliers=False)
    plt.ylabel('ARI')
    plt.xlabel("resolution")
    plt.xticks(list(range(1, len(RESOLUTIONS) + 1)), [f"{res:.2f}" for res in RESOLUTIONS])
    plt.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.05, hspace=0.13)
    plt.savefig(result_dir + index + '_ari_random_50.eps', dpi=600, format='eps')
    # plt.show()
    plt.close()
