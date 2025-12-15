#该数据集5种算法下5个采样比例的运行时间，需要自行修改data[]
import matplotlib.pyplot as plt
import numpy as np
import sys
def create_time_chart(output_path):
    num_experiments = 29

    # input your orkut runtime data
    data = [6.83, 4451.59, 183.65, 11.48, 3.29, 0,
            3.33, 3114.51, 173.69, 6.78, 3.48, 0,
            4.36, 2105.62, 191.83, 7.61, 2.87, 0,
            4.84, 1280.42, 171.49, 4.72, 2.87, 0,
            3.87, 372.89, 170.85, 4.33, 4.81
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
