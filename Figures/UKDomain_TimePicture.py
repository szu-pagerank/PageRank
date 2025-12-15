import matplotlib.pyplot as plt
import numpy as np
import sys
def create_time_chart(output_path):
    num_experiments = 29

    #input your UKDomain runtime data
    data = [355.55, 124021.85, 5066.83, 707.34, 117.28, 0,
           200.00, 90398.59, 5028.10, 609.61, 124.05, 0,
           331.64, 64709.47, 4891.52, 360.45, 105.34, 0,
           268.69, 43906.08, 4930.50, 194.64, 102.55, 0,
           133.63, 15753.61, 4909.17, 129.28, 92.62
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
