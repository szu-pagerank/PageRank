# PageRank
## Intruduction
This is an implementation of the CUR_Trans and $T^2$-Approx algorithms proposed in the paper “Efficient and Accurate PageRank Approximation on Large Graphs” published in the 2025 Proceedings of the ACM on Management of Data, and includes the competitors which are sampling-based PageRank estimations.

```bash
@article{DBLP:journals/pacmmod/WuWQCL24,
  author       = {Siyue Wu and
                  Dingming Wu and
                  Junyi Quan and
                  Tsz Nam Chan and
                  Kezhong Lu},
  title        = {Efficient and Accurate PageRank Approximation on Large Graphs},
  journal      = {Proc. {ACM} Manag. Data},
  volume       = {2},
  number       = {4},
  pages        = {196:1--196:26},
  year         = {2024}
}
```






## Contents
### Algorithms
1.CUR.py: The algorithm CUR-Trans proposed in this paper and its variant SVD-Trans.

2.T2.py:  The algorithm $T^2$-Approx proposed in this paper.

3.Competitors.py: The comparison algorithm in this paper.

### Figures
1.time picture: Orkut_TimePicture.py,Friendster_TimePicture.py,UKDomain_TimePicture.py

2.error boxplot: Error_Boxplot.py

### Poster.pdf
Display poster of main paper content.

### Supplementary Material.pdf
Proofs and experiments not included in the paper due to limited page.











## Required environment
Since the source code of the competitors are not available, we have implemented all the competitors. Specifically, algorithms DSPI and LPRAP compute PageRank values based on graph operations. They are implemented using  Networkit (10.1). Algorithms ApproxRank, SVD-Trans, CUR-Trans, and $T^2$-Approx perform matrix iterations. They are implemented using  NumPy (1.20.1) and SciPy (1.8.1).

All experiments are running on a machine with 4 Intel Xeon E7-4830 CPUs (56 cores, 2.0 GHz) and 2 TB main memory with Python (3.8.8). 









## Dataset
We conduct evaluations on three real large graph datasets, Friendster and Orkut and UKDomain. These dataset can be download at:

Friendster: http://konect.cc/networks/friendster/

Orkut: http://konect.cc/networks/orkut-groupmemberships/

UKDomain: http://konect.cc/networks/dimacs10-uk-2007-05/









## Proposed algorithms
### CUR_Trans

```bash
# args[1]: source code file name
# args[2]: algorithm name
# args[3]: graph path
# args[4]: output path
# args[5]: sampling ratio of col
# args[6]: the number of nodes in the graph
# args[7]: sampling ratio of row
python CUR.py CUR_Trans /your_graph_path /your_output_path 0.1 1000 0.1
```

### $T^2$-Approx

```bash
# args[1]: source code file name
# args[2]: graph path
# args[3]: output path
# args[4]: sampling ratio
python T2.py /your_graph_path /your_output_path 0.1
```

### Other variants of CUR_Trans and $T^2$-Approx 

> $T^2$-Approx involve sampling operations on the transition matrix $T$, and we can use different sampling distributions. If you want to sample from matrix using a different sampling probability distribution, you need to uncomment the code below.

``` python
# Step I: Calculate the distribution we want. We propose four widely used distributions on matrix smapling.

# (1)F-norm of column and row (widely used and applied in our paper):
col_sums = A.power(2).sum(axis=0)
row_sums = A.power(2).sum(axis=1)
p = np.multiply(col_sums, row_sums)
probabilities = p / sum(p)

# (2)F-norm of column: 
# total_sums = A.power(2).sum()
# col_sums = A.power(2).sum(axis=0)
# col_distribution = col_sums / total_sums

# (3)F-norm of row: 
# total_sums = A.power(2).sum()
# row_sums = A.power(2).sum(axis=1)
# row_distribution = row_sums / total_sums

# (4)uniform distribution.

# Step II: Use different distributions.
# (1)F-norm of column and row:
sampled_index = np.random.choice(A.shape[1], size=c, replace=False, p=probabilities)

# (2)F-norm of column: 
# sampled_index = np.random.choice(A.shape[1], size=c, replace=False, p=col_distribution)

# (3)F-norm of row: 
# sampled_index = np.random.choice(A.shape[1], size=c, replace=False, p=row_distribution)

# (4)uniform distribution:
# sampled_index = np.random.choice(A.shape[1], size=c, replace=False, p=None)
```

> The low-rank approximation algorithm used in CUR_Trans and the Monte Carlo matrix multiplication algorithm used in $T^2$-Approx both recommend scaling each sampled element to obtain better approximation results. However, this is not applicable to the PageRank scenario, as detailed in the paper. Nevertheless, this project provides scaling operations for others to test and study. If you need to scale the sampled elements, you need to uncomment the following code.

``` python
# # Scaling each element in R.
# for i in range(R.shape[0]):
#     start_idx = R.indptr[i]
#     end_idx = R.indptr[i + 1]
#     for j in range(start_idx, end_idx):
#         col_idx = R.indices[j]
#         value = R.data[j]
#         R[i, col_idx] = value / row_distribution[sampled_row_index[i]]
#
# # Scaling each element in C.
# C = sp.sparse.csc_array(C)
# for j in range(C.shape[1]):
#     start_idx = C.indptr[j]
#     end_idx = C.indptr[j + 1]
#     for i in range(start_idx, end_idx):
#         row_idx = C.indices[i]
#         value = C.data[i]
#         C[row_idx, j] = value / col_distribution[sampled_col_index[j]]
```

> CUR-Trans use CUR low rank approximation method to get a smaller iteration matrix of transition matrix $T$ for PageRank, and other low rank approximation methods can replace CUR. We provide SVD as an example and propose SVD-Trans.

``` bash
# args[1]: algorithm name
# args[2]: graph path
# args[3]: output path
# args[4]: sampling ratio of col
# args[5]: the number of nodes in the graph
python CUR.py SVD_Trans /your_graph_path /your_output_path 0.1 1000
```

## Competitors 

``` bash
# args[1]: source code file name
# args[2]: algorithm name
# args[3]: graph path
# args[4]: output path

# Use Networkit to calculate the PageRank of the original graph as the ground truth of the experiment.
python Competitors.py GroundTruth /your_graph_path /your_output_path

# --------------------competitors in the paper

# DSPI：
# args[5]: alpha,
# args[6]: theta, alpha and theta together determine the sampling probability of elements.
python Competitors.py DSPI /your_graph_path /your_output_path 0.1 0.1

# ApproxRank:
# args[5]: sampling_ratio, 
# args[6]: node_num, the number of vertices in the subgraph.
python Competitors.py ApproxRank /your_graph_path /your_output_path 0.1 1000

# LPRAP：
# args[5]:sampling_num, the number of vertices in the subgraph.
# args[6]:edges_ration, the sampling ratio of edges.
# args[7]:T, purning threshold.
python Competitors.py LPRAP /your_graph_path /your_output_path 100 0.1 0.1

# ---------------------- other competitors

# LocalPR: LPRAP is an optimized version of LocalPR, so the article only compares LPRAP and not LocalPR. 
# args[5]: sampling_num, the number of vertices in the subgraph.
# args[6]: edges ration, the sampling ratio of edges.
python Competitors.py LocalPR /your_graph_path /your_output_path 100 0.1

# PER_PR: DSPI performs biased sampling on the elements in the matrix, and we have also implemented a uniform sampling version.
# args[5]: theta, the sampling ratio of sparsifying edge
python Competitors.py PER_PR /your_graph_path /your_output_path 0.1
```






## Detailed Steps for Using the Code

#Step 0: Download the datasets.


#Step 1: Use Networkit to calculate the PageRank of the original graph as the ground truth of the experiment.
# args[1]: source code file name
# args[2]: algorithm name
# args[3]: graph path
# args[4]: output path
# Orkut/Friendster/UKdomain Groundtruth 
python Competitors.py GroundTruth /your_graph_path /your_output_path


#Step 2: Run the following command to generate the PageRank result files of five algorithms with different sampling ratios.
#You can execute the following example command in this step to conveniently preserve the output log.
nohup python CUR.py CUR_Trans /your_graph_path /your_output_path 0.00018 3072441 0.0065 > your_output_path.log

#Competitors
#ApproxRank
# args[1]: source code file name
# args[2]: algorithm name
# args[3]: graph path
# args[4]: output path
# args[5]: sampling_ratio
# args[6]: node_num

#Orkut ApproxRank
#sampled edges (%)  0.1
python Competitors.py ApproxRank /your_graph_path /your_output_path 0.02 3072441
#sampled edges (%)  0.3
python Competitors.py ApproxRank /your_graph_path /your_output_path 0.045 3072441
#sampled edges (%)  0.5
python Competitors.py ApproxRank /your_graph_path /your_output_path 0.06 3072441
#sampled edges (%)  0.7
python Competitors.py ApproxRank /your_graph_path /your_output_path 0.075 3072441
#sampled edges (%)  1.0
python Competitors.py ApproxRank /your_graph_path /your_output_path 0.09 3072441

#Friendster ApproxRank
#sampled edges (%)  0.1
python Competitors.py ApproxRank /your_graph_path /your_output_path 0.02 65608366
#sampled edges (%)  0.3
python Competitors.py ApproxRank /your_graph_path /your_output_path 0.04 65608366
#sampled edges (%)  0.5
python Competitors.py ApproxRank /your_graph_path /your_output_path 0.055 65608366
#sampled edges (%)  0.7
python Competitors.py ApproxRank /your_graph_path /your_output_path 0.07 65608366
#sampled edges (%)  1.0
python Competitors.py ApproxRank /your_graph_path /your_output_path 0.09 65608366

#UKDomain ApproxRank
#sampled edges (%)  0.1
python Competitors.py ApproxRank /your_graph_path /your_output_path 0.02 105153952
#sampled edges (%)  0.3
python Competitors.py ApproxRank /your_graph_path /your_output_path 0.04 105153952
#sampled edges (%)  0.5
python Competitors.py ApproxRank /your_graph_path /your_output_path 0.06 105153952
#sampled edges (%)  0.7
python Competitors.py ApproxRank /your_graph_path /your_output_path 0.07 105153952
#sampled edges (%)  1.0
python Competitors.py ApproxRank /your_graph_path /your_output_path 0.09 105153952

#LPRAP
# args[1]: source code file name
# args[2]: algorithm name
# args[3]: graph path
# args[4]: output path
# args[5]:sampling_num, the number of vertices in the subgraph.
# args[6]:edges_ration, the sampling ratio of edges.
# args[7]:T, purning threshold.
#Orkut LPRAP
#sampled edges (%)  0.1
python Competitors.py LPRAP /your_graph_path /your_output_path 1000 0.001 1e-8
#sampled edges (%)  0.3
python Competitors.py LPRAP /your_graph_path /your_output_path 1000 0.003 1e-8
#sampled edges (%)  0.5
python Competitors.py LPRAP /your_graph_path /your_output_path 1000 0.005 1e-8
#sampled edges (%)  0.7
python Competitors.py LPRAP /your_graph_path /your_output_path 1000 0.007 1e-8
#sampled edges (%)  1.0
python Competitors.py LPRAP /your_graph_path /your_output_path 1000 0.01 1e-8

#Friendster LPRAP
#sampled edges (%)  0.1
python Competitors.py LPRAP /your_graph_path /your_output_path 1000 0.001 1e-9
#sampled edges (%)  0.3
python Competitors.py LPRAP /your_graph_path /your_output_path 1000 0.003 1e-9
#sampled edges (%)  0.5
python Competitors.py LPRAP /your_graph_path /your_output_path 1000 0.005 1e-9
#sampled edges (%)  0.7
python Competitors.py LPRAP /your_graph_path /your_output_path 1000 0.007 1e-9
#sampled edges (%)  1.0
python Competitors.py LPRAP /your_graph_path /your_output_path 1000 0.01 1e-9

#UKDomain LPRAP
#sampled edges (%)  0.1
python Competitors.py LPRAP /your_graph_path /your_output_path 1000 0.001 1e-10
#sampled edges (%)  0.3
python Competitors.py LPRAP /your_graph_path /your_output_path 1000 0.003 1e-10
#sampled edges (%)  0.5
python Competitors.py LPRAP /your_graph_path /your_output_path 1000 0.005 1e-10
#sampled edges (%)  0.7
python Competitors.py LPRAP /your_graph_path /your_output_path 1000 0.007 1e-10
#sampled edges (%)  1.0
python Competitors.py LPRAP /your_graph_path /your_output_path 1000 0.01 1e-10

# DSPI：
# args[1]: source code file name
# args[2]: algorithm name
# args[3]: graph path
# args[4]: output path
# args[5]: alpha,
# args[6]: theta, alpha and theta together determine the sampling probability of elements.

# Orkut DSPI 
#sampled edges (%)  0.1
python Competitors.py DSPI /your_graph_path /your_output_path 130000 0.999
#sampled edges (%)  0.3
python Competitors.py DSPI /your_graph_path /your_output_path 20000 0.999
#sampled edges (%)  0.5
python Competitors.py DSPI /your_graph_path /your_output_path 8000 0.999
#sampled edges (%)  0.7
python Competitors.py DSPI /your_graph_path /your_output_path 4000 0.999
#sampled edges (%)  1.0
python Competitors.py DSPI /your_graph_path /your_output_path 1500 0.999

# Friendster DSPI 
#sampled edges (%)  0.1
python Competitors.py DSPI /your_graph_path /your_output_path 35000 0.999
#sampled edges (%)  0.3
python Competitors.py DSPI /your_graph_path /your_output_path 7000 0.999
#sampled edges (%)  0.5
python Competitors.py DSPI /your_graph_path /your_output_path 2500 0.999
#sampled edges (%)  0.7
python Competitors.py DSPI /your_graph_path /your_output_path 1000 0.999
#sampled edges (%)  1.0
python Competitors.py DSPI /your_graph_path /your_output_path 500 0.999

# UKDomain DSPI 
#sampled edges (%)  0.1
python Competitors.py DSPI /your_graph_path /your_output_path 35000 0.999
#sampled edges (%)  0.3
python Competitors.py DSPI /your_graph_path /your_output_path 10000 0.999
#sampled edges (%)  0.5
python Competitors.py DSPI /your_graph_path /your_output_path 4000 0.999
#sampled edges (%)  0.7
python Competitors.py DSPI /your_graph_path /your_output_path 1500 0.999
#sampled edges (%)  1.0
python Competitors.py DSPI /your_graph_path /your_output_path 1000 0.999

#The data loading phase of this code is indeed quite time-consuming. Therefore, we utilize the tqdm library to display a progress bar. 
#You can directly run the CUR.py file to conveniently monitor the program's execution progress.
# CUR
# args[1]: source code file name
# args[2]: algorithm name
# args[3]: graph path
# args[4]: output path
# args[5]: sampling ratio of col
# args[6]: the number of nodes in the graph
# args[7]: sampling ratio of row

#Orkut CUR
#sampled edges (%)  0.1
python CUR.py CUR_Trans /your_graph_path /your_output_path 0.00018 3072441 0.0065
#sampled edges (%)  0.3
python CUR.py CUR_Trans /your_graph_path /your_output_path 0.00048 3072441 0.015
#sampled edges (%)  0.5
python CUR.py CUR_Trans /your_graph_path /your_output_path 0.0009 3072441 0.025
#sampled edges (%)  0.7
python CUR.py CUR_Trans /your_graph_path /your_output_path 0.0014 3072441 0.035
#sampled edges (%)  1.0
python CUR.py CUR_Trans /your_graph_path /your_output_path 0.0022 3072441 0.05

#Friendster CUR
#sampled edges (%)  0.01
python CUR.py CUR_Trans /your_graph_path /your_output_path 0.00003 65608366 0.0025
#sampled edges (%)  0.05
python CUR.py CUR_Trans /your_graph_path /your_output_path 0.00015 65608366 0.01
#sampled edges (%)  0.1
python CUR.py CUR_Trans /your_graph_path /your_output_path 0.0003 65608366 0.02
#sampled edges (%)  0.15
python CUR.py CUR_Trans /your_graph_path /your_output_path 0.0005 65608366 0.035
#sampled edges (%)  0.3
python CUR.py CUR_Trans /your_graph_path /your_output_path 0.0008 65608366 0.05

#UKdomain CUR
#sampled edges (%)  0.1
python CUR.py CUR_Trans /your_graph_path /your_output_path 0.00002 105153952 0.012
#sampled edges (%)  0.3
python CUR.py CUR_Trans /your_graph_path /your_output_path 0.00005 105153952 0.03
#sampled edges (%)  0.5
python CUR.py CUR_Trans /your_graph_path /your_output_path 0.0001 105153952 0.055
#sampled edges (%)  0.7
python CUR.py CUR_Trans /your_graph_path /your_output_path 0.00015 105153952 0.08
#sampled edges (%)  1.0
python CUR.py CUR_Trans /your_graph_path /your_output_path 0.0002 105153952 0.09

#You can directly run the T2.py file to conveniently monitor the program's execution progress.
# T2-Approx
# args[1]: source code file name
# args[2]: graph path
# args[3]: output path
# args[4]: sampling ratio
# Orkut T2-Approx  
#sampled edges (%)  0.1
python T2.py /your_graph_path /your_output_path 0.002
#sampled edges (%)  0.3
python T2.py /your_graph_path /your_output_path 0.004
#sampled edges (%)  0.5
python T2.py /your_graph_path /your_output_path 0.007
#sampled edges (%)  0.7
python T2.py /your_graph_path /your_output_path 0.01
#sampled edges (%)  1.0
python T2.py /your_graph_path /your_output_path 0.015

# Friendster T2-Approx  
#sampled edges (%)  0.1
python T2.py /your_graph_path /your_output_path 0.0025
#sampled edges (%)  0.3
python T2.py /your_graph_path /your_output_path 0.008
#sampled edges (%)  0.5
python T2.py /your_graph_path /your_output_path 0.0115
#sampled edges (%)  0.7
python T2.py /your_graph_path /your_output_path 0.016
#sampled edges (%)  1.0
python T2.py /your_graph_path /your_output_path 0.024

# UKDomain T2-Approx  
#sampled edges (%)  0.1
python T2.py /your_graph_path /your_output_path 0.004
#sampled edges (%)  0.3
python T2.py /your_graph_path /your_output_path 0.01
#sampled edges (%)  0.5
python T2.py /your_graph_path /your_output_path 0.015
#sampled edges (%)  0.7
python T2.py /your_graph_path /your_output_path 0.02
#sampled edges (%)  1.0
python T2.py /your_graph_path /your_output_path 0.025


#Step 3: Run the following command to generate bar charts of PageRank computation time for five algorithms at different ratios.
#You need to change the data[] in the code to your runtime data!
# args[1]: source code file name
# args[2]: output path
python Orkut_TimePicture.py your_output_path.pdf
python Friendster_TimePicture.py your_output_path.pdf
python UKDomain_TimePicture.py your_output_path.pdf


#Step 4: Run the following command to generate box plots of the absolute errors for each node across five algorithms at different ratios.
# args[1]: source code file name
# args[2]: groundtruth path
# args[3-7]: your_ApproxRank_path1-5
# args[8-12]: your_LPRAP_path1-5
# args[13-17]: your_DSPI_path1-5
# args[18-22]: your_CUR_path1-5
# args[23-27]: your_T2-Approx_path1-5
# args[28]: dataset name, ["orkut", "friendster", "uk2007"]
# args[29]: your_output_path
nohup python Error_Boxplot.py \
groundtruth path \
your_ApproxRank_path1 \
your_ApproxRank_path2 \
your_ApproxRank_path3 \
your_ApproxRank_path4 \
your_ApproxRank_path5 \
your_LPRAP_path1 \
your_LPRAP_path2 \
your_LPRAP_path3 \
your_LPRAP_path4 \
your_LPRAP_path5 \
your_DSPI_path1 \
your_DSPI_path2 \
your_DSPI_path3 \
your_DSPI_path4 \
your_DSPI_path5 \
your_CUR_path1 \
your_CUR_path2 \
your_CUR_path3 \
your_CUR_path4 \
your_CUR_path5 \
your_T2-Approx_path1 \
your_T2-Approx_path2 \
your_T2-Approx_path3 \
your_T2-Approx_path4 \
your_T2-Approx_path5 \
dataset name \
your_output_path.pdf > your_output_log.log 
