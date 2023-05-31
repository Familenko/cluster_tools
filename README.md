# Cluster_assesment
 Check and build one of cluster algorithm (Aglomerative, K-mean,DBscan)

This class help to identify best amoun of clusters required for raw data to be divided for
For this purpose used Aglomerative, K-mean,DBscan clusterisations and required following step:

    - algorithm may required MinMax or StandardScaler data preprocessing

    - main method is knife wich is building diagram wich includes silhuet coefficient for ich particular element of data
    and represent it in different cluster. On diagram is also draph average silhuete score line. Respectly to this to
    select correct amount of cluster it is required to visualy check silhuet of all 'knifes', they shape, length, leakege presents
    and corresponding to average silhuete score line.

    Best amount of cluster should show on diagram equal to ich other shape, length
    of knifes without any leakega with the biggest amount of elements higher than average silhuete score line
    Average silhuete score is close to 1.0

    - next method is distance wich is building diagram wich is includes silhuet and distance_threshold,
    both parameter required to be on maximum

    or

    - next method is mount wich is building diagram wich is includes squered_distances and gap

    or

    - next method is outliers wich is building diagram wich is showing amoun of outliers

To build required clusterisation used metod build_knife and build_simple (build on PCA preprocessed data)
Final dataframe posible to return by cluster.result or cluster.result_simple argument

It is posible to check data using PCA by method simple_check with different mode and parameters

Ordianry method for cluster assesment build informative diagram:

    linkage - Check the optimal amount of cluster by scipy.cluster.hierarchy
    dendrogram - By using this diagram possible make assesment of choosen amount of cluster on actual data
    pie,heat,feature_corr - Build feature correlation diagram for 'knife' or 'simple' algorithm (use after build method)
