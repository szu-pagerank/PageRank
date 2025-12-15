#输出5种算法5个比例下每个节点与真实值的绝对误差箱体图
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import sys


args = sys.argv
ground_truth_path = args[1]

ApproxRank1 = args[2]
ApproxRank2 = args[3]
ApproxRank3 = args[4]
ApproxRank4 = args[5]
ApproxRank5 = args[6]

LPRAP1 = args[7]
LPRAP2 = args[8]
LPRAP3 = args[9]
LPRAP4 = args[10]
LPRAP5 = args[11]

DSPI1 = args[12]
DSPI2 = args[13]
DSPI3 = args[14]
DSPI4 = args[15]
DSPI5 = args[16]

# CUR
CUR_PR1 = args[17]
CUR_PR2 = args[18]
CUR_PR3 = args[19]
CUR_PR4 = args[20]
CUR_PR5 = args[21]

# T2
T1 = args[22]
T2 = args[23]
T3 = args[24]
T4 = args[25]
T5 = args[26]

dataset = args[27]
output = args[28]


ground_truth_data = {}
if dataset == "orkut":
    with open(ground_truth_path, 'r') as file:
        lines = file.readlines()
        ground_truth_data = {int(line.split()[0]): float(line.split()[1]) for line in lines}
elif dataset == "friendster":
    with open(ground_truth_path, 'r') as file:
        lines = file.readlines()
        ground_truth_data = {int(line.split()[0]): float(line.split()[1]) for line in lines}
elif dataset == "uk2007":
    with open(ground_truth_path, 'r') as file:
        lines = file.readlines()
        ground_truth_data = {int(line.split()[0]): float(line.split()[1]) for line in lines}


def calculate_error(file_path, ground_truth_data):
    result = {}
    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split()
            node = int(data[0])
            pagerank = float(data[1])
            result[node] = pagerank

    errors = []
    relative_error_sum = 0

    for node, true_pagerank in ground_truth_data.items():
        if node in result:
            absolute_error = abs(true_pagerank - result[node])
            relative_error = absolute_error / true_pagerank
        else:
            absolute_error = true_pagerank
            relative_error = 1

        errors.append(absolute_error)
        relative_error_sum += relative_error

    avg_relative_error = relative_error_sum / len(errors)
    total_absolute_error = sum(errors)

    print(f"Algorithm: {file_path}")
    print(f"  Average relative error: {avg_relative_error:.6f}")
    print(f"  Total absolute error: {total_absolute_error:.6f}")
    print()

    return errors


print("Calculating errors for all algorithms...")

# ApproxRank
ApproxRank1_error = calculate_error(ApproxRank1, ground_truth_data)
ApproxRank2_error = calculate_error(ApproxRank2, ground_truth_data)
ApproxRank3_error = calculate_error(ApproxRank3, ground_truth_data)
ApproxRank4_error = calculate_error(ApproxRank4, ground_truth_data)
ApproxRank5_error = calculate_error(ApproxRank5, ground_truth_data)

# LPRAP
LPRAP1_error = calculate_error(LPRAP1, ground_truth_data)
LPRAP2_error = calculate_error(LPRAP2, ground_truth_data)
LPRAP3_error = calculate_error(LPRAP3, ground_truth_data)
LPRAP4_error = calculate_error(LPRAP4, ground_truth_data)
LPRAP5_error = calculate_error(LPRAP5, ground_truth_data)

# DSPI
DSPI1_error = calculate_error(DSPI1, ground_truth_data)
DSPI2_error = calculate_error(DSPI2, ground_truth_data)
DSPI3_error = calculate_error(DSPI3, ground_truth_data)
DSPI4_error = calculate_error(DSPI4, ground_truth_data)
DSPI5_error = calculate_error(DSPI5, ground_truth_data)

# CUR
CUR_PR1_error = calculate_error(CUR_PR1, ground_truth_data)
CUR_PR2_error = calculate_error(CUR_PR2, ground_truth_data)
CUR_PR3_error = calculate_error(CUR_PR3, ground_truth_data)
CUR_PR4_error = calculate_error(CUR_PR4, ground_truth_data)
CUR_PR5_error = calculate_error(CUR_PR5, ground_truth_data)

# T2
T1_error = calculate_error(T1, ground_truth_data)
T2_error = calculate_error(T2, ground_truth_data)
T3_error = calculate_error(T3, ground_truth_data)
T4_error = calculate_error(T4, ground_truth_data)
T5_error = calculate_error(T5, ground_truth_data)


boxprops = dict(linewidth=1)
flierprops = dict(marker='o', markersize=8, markerfacecolor='red', markeredgecolor='red')
medianprops = dict(linestyle='-', linewidth=1, color='firebrick')

colors = ['Brown', 'orange', 'Cyan', 'blue', 'lightpink', 'white',
          'Brown', 'orange', 'Cyan', 'blue', 'lightpink', 'white',
          'Brown', 'orange', 'Cyan', 'blue', 'lightpink', 'white',
          'Brown', 'orange', 'Cyan', 'blue', 'lightpink', 'white',
          'Brown', 'orange', 'Cyan', 'blue', 'lightpink']

error = []

fig, ax = plt.subplots(figsize=(12, 3.5))


bp = ax.boxplot([ApproxRank1_error, LPRAP1_error, DSPI1_error, CUR_PR1_error, T1_error, error,
                 ApproxRank2_error, LPRAP2_error, DSPI2_error, CUR_PR2_error, T2_error, error,
                 ApproxRank3_error, LPRAP3_error, DSPI3_error, CUR_PR3_error, T3_error, error,
                 ApproxRank4_error, LPRAP4_error, DSPI4_error, CUR_PR4_error, T4_error, error,
                 ApproxRank5_error, LPRAP5_error, DSPI5_error, CUR_PR5_error, T5_error],
                showfliers=False, boxprops=boxprops, flierprops=flierprops, medianprops=medianprops, widths=0.5)


for box, color in zip(bp['boxes'], colors):
    box.set(color=color)


labels = ['', '', '1.0', '', '', '',
          '', '', '0.7', '', '', '',
          '', '', '0.5', '', '', '',
          '', '', '0.3', '', '', '',
          '', '', '0.1', '', '']

ax.set_xticklabels(labels, weight='bold')

# Friendster
if dataset == "friendster":
    special_x = [4, 10, 16, 22, 28]
    special_label = ['0.3', '0.15', '0.1', '0.05', '0.01']
    for x, label in zip(special_x, special_label):
        plt.annotate(label, xy=(x, 2.3e-8), xytext=(x, 3e-8),
                    arrowprops=dict(facecolor='black', arrowstyle='->'))


ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=True))
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_weight('bold')
offset_text = ax.yaxis.get_offset_text()
offset_text.set_size(14)
offset_text.set_weight('bold')

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

ax.set_ylabel('Error', weight='bold', fontsize=17)
ax.set_xlabel('#sampled edges (%)', weight='bold', fontsize=17)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']


plt.savefig(output, bbox_inches="tight", dpi=300)
plt.show()

