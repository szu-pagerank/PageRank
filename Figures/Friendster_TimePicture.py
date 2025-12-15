import matplotlib.pyplot as plt
import numpy as np
import sys
def create_time_chart(output_path):
    num_experiments = 29

    # input your Friendster runtime data
    data = [313.11, 102772.37, 3259.73, 604.55, 264.80, 0,
            289.00, 78980.33, 3168.84, 423.80, 242.68, 0,
            254.31, 48004.82, 3257.04, 305.10, 248.35, 0,
            362.04, 31502.62, 3077.95, 212.41, 247.87, 0,
            229.10, 8114.21, 3174.42, 198.17, 214.65
            ]

    fig, ax = plt.subplots(figsize=(10.5, 3.5))

    bar_positions = np.arange(num_experiments)
    bar_width = 0.8

    colors = ['Brown', 'orange', 'Cyan', 'blue', 'lightpink', 'white',
              'Brown', 'orange', 'Cyan', 'blue', 'lightpink', 'white',
              'Brown', 'orange', 'Cyan', 'blue', 'lightpink', 'white',
              'Brown', 'orange', 'Cyan', 'blue', 'lightpink', 'white',
              'Brown', 'orange', 'Cyan', 'blue', 'lightpink']
    ax.bar(bar_positions, data, width=bar_width, color=colors)

    plt.ylabel('Time (seconds)', weight='bold', fontsize=19)
    ax.set_xlabel('#sampled edges (%)', weight='bold', fontsize=19)
    ax.set_yscale('log')

    labels = ['', '', '1.0', '', '', '',
              '', '', '0.7', '', '', '',
              '', '', '0.5', '', '', '',
              '', '', '0.3', '', '', '',
              '', '', '0.1', '', '']

    # Friendster label
    special_x = [4, 10, 16, 22, 28]
    special_label = ['0.3', '0.15', '0.1', '0.05', '0.01']
    for x, label in zip(special_x, special_label):
        plt.annotate(label, xy=(x - 1, 650), xytext=(x - 1, 2000), arrowprops=dict(facecolor='black', arrowstyle='->'))

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(labels, weight='bold')

    ax.set_xlim([-bar_width / 2, num_experiments - 1 + bar_width / 2])

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14, weight='bold')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

    plt.savefig(output_path, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    output_path = sys.argv[1]
    create_time_chart(output_path)
