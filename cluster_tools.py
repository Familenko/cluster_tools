import timeit
from tqdm import tqdm

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN

from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.pipeline import make_pipeline

class Clusterer():

    '''
    This class help to identify best amoun of clusters required for raw data to be divided for
    For this purpose used Aglomerative clusterisation and required following step:

        - present algorithm is very sensetive and required MinMax data preprocessing
        wich can be done by method minmax

        - main method is knife wich is building diagram wich includes silhuet coefficient for ich particular element of data
        and represent it in different cluster. On diagram is also draph average silhuete score line. Respectly to this to
        select correct amount of cluster it is required to visualy check silhuet of all 'knifes', they shape, length, leakege presents
        and corresponding to average silhuete score line.

        Best amount of cluster should show on diagram equal to ich other shape, length
        of knifes without any leakega with the biggest amount of elements higher than average silhuete score line
        Average silhuete score is close to 1.0

        - next method is distance wich is building diagram wich is includes silhuet and distance_threshold,
        both parameter required to be on maximum

    To build required clusterisation used metod build_knife and build_simple (build on PCA preprocessed data)
    Final dataframe posible to return by agglo.result or agglo.result_simple argument

    It is posible to check data using PCA by method simple_check with different mode and parameters

    Ordianry method for cluster assesment build informative diagram:

        linkage - Check the optimal amount of cluster by scipy.cluster.hierarchy
        dendrogram - By using this diagram possible make assesment of choosen amount of cluster on actual data
        pie,heat,feature_corr - Build feature correlation diagram for 'knife' or 'simple' algorithm (use after build method)
    '''
    def __init__(self,X):
        
        self.df = X
        self.X = X

    def elapsed_time_decorator(func):
        def wrapper(*args, **kwargs):
            start_time = timeit.default_timer()
            result = func(*args, **kwargs)
            elapsed_time = round(timeit.default_timer() - start_time, 2)
            print("Elapsed time:", elapsed_time)
            return result
        return wrapper

    @elapsed_time_decorator
    def agglo_distance(self,min_d=1,max_d=5,range_d=10,save=None):

        # DESCRIPTION:

        #     Build diagram with distance correlation of data in different cluster

        # ARGUMENTS:

        #     min_d - minimum level of distance
        #     max_d - maximum level of distance
        #     range_d - amount of tested distances

        distance_threshold_list = list(np.linspace(min_d,max_d,range_d))

        n_claster_dis = []
        distance_threshold = []
        silhouette_dis = []
    
        for distance_threshold_n in tqdm(distance_threshold_list, desc="Checking distance"):
            
            clusterer = AgglomerativeClustering(n_clusters=None,distance_threshold=distance_threshold_n)
            cluster_labels = clusterer.fit_predict(self.X)
            
            distance_threshold.append(distance_threshold_n)
            n_claster_dis.append(len(np.unique(cluster_labels)))
            silhouette_dis.append(round(silhouette_score(self.X, cluster_labels),3),)
            
    
        plt.figure(figsize=(12,5),dpi=200)
        sns.scatterplot(x=n_claster_dis,y=distance_threshold,hue=silhouette_dis,s=100)
        plt.title("Distance plot")
        plt.xlabel("Number of claster")
        plt.ylabel("Distance_threshold")
        plt.xticks(range(min(n_claster_dis), max(n_claster_dis)+1))
        plt.yticks(np.arange(0, max(distance_threshold)+1, step=1))
        plt.legend(title='Silhouette', loc='upper right')

        if save =='save':
            plt.savefig('my_plot.png')
        
        plt.show()

        df = pd.DataFrame({
            'n_clusters':n_claster_dis,
            'distance_threshold':distance_threshold,
            'silhouette':silhouette_dis
            })

        return df.transpose()

    @elapsed_time_decorator
    def agglo_knife(self,min_k,max_k,step=1,knife=True):

        # DESCRIPTION:

        #     Build selected range of clusters and represent knife metric to ich of them
        #     'Knifes' should be similar to ich other and have good shape without leakeges

        # ARGUMENTS:

        #     min_k - minimum amount of cluster
        #     max_k - maximum amount of cluster

        range_n_clusters = list(range(min_k,max_k,step))

        n_clus = []
        silhouette=[]        

        for n_clusters in tqdm(range_n_clusters, desc="Checking knifes"):
            fig, ax1 = plt.subplots(1, 1)
            fig.set_size_inches(7, 4)

            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(self.X)

            n_clus.append(n_clusters)
            silhouette.append(round(silhouette_score(self.X, cluster_labels),3),)

            print(
                "n_clusters =",
                n_clusters,
                "average silhouette_score =",
                round(silhouette_score(self.X, cluster_labels),3),)

            if knife:

                sample_silhouette_values = silhouette_samples(self.X, cluster_labels)

                y_lower = 10
                for i in range(n_clusters):
                    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
                    ith_cluster_silhouette_values.sort()
                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i
                    color = cm.nipy_spectral(float(i) / n_clusters)
                    ax1.fill_betweenx(
                        np.arange(y_lower, y_upper),
                        0,
                        ith_cluster_silhouette_values,
                        facecolor=color,
                        edgecolor=color,
                        alpha=0.7,)

                    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                    y_lower = y_upper + 10

                ax1.set_title("Silhouette plot for n_clusters = %d" % n_clusters)
                ax1.set_xlabel("Silhouette coefficient values")
                ax1.set_ylabel("Cluster label")
                ax1.axvline(x=silhouette_score(self.X, cluster_labels), color="red", linestyle="--")
                ax1.set_yticks([])
                ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

                plt.show()

        df = pd.DataFrame({
            'silhouette':silhouette},
            index=range(min_k,max_k,step))

        return df.transpose()

    @elapsed_time_decorator
    def kmean_mount(self,min_n,max_n,save=None):

        # DESCRIPTION:

        #     Build diagram with squared distance correlation of data in different cluster

        # ARGUMENTS:

        #     min_n - minimum cluster
        #     max_n - maximum cluster

        range_n_clusters = list(range(min_n,max_n))

        n_clus_mount=[]
        ssd_mount=[]
        silhouette_mount=[]

        for n_clusters in tqdm(range_n_clusters):

            clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=10)
            cluster_labels = clusterer.fit_predict(self.X)

            n_clus_mount.append(n_clusters)
            ssd_mount.append(round(clusterer.inertia_,1))
            silhouette_mount.append(round(silhouette_score(self.X, cluster_labels),3),)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

        ax1.plot(range(min_n,max_n), ssd_mount, 'o--')
        ax1.set_xlabel("K Value")
        ax1.set_ylabel("Sum of Squared Distances")

        pd.Series(ssd_mount).diff().plot(kind='bar', ax=ax2)
        ax2.set_xlabel("K Value")
        ax2.set_ylabel("Change in Sum of Squared Distances")
        plt.tight_layout()
        if save =='save':
            plt.savefig('my_plot.png')
        plt.show()

        df = pd.DataFrame({
            'n_clusters':n_clus_mount,
            'ssd':ssd_mount,
            'silhouette':silhouette_mount,
            'diff':pd.Series(ssd_mount).diff()
            })

        return df.transpose()

    @elapsed_time_decorator
    def kmean_knife(self,min_n,max_n,step=1,knife=True):

        # DESCRIPTION:

        #     Build selected range of clusters and represent knife metric to ich of them
        #     'Knifes' should be similar to ich other and have good shape without leakeges

        # ARGUMENTS:

        #     min_n - minimum amount of cluster
        #     max_n - maximum amount of cluster

        range_n_clusters = list(range(min_n,max_n,step))

        n_clus_knife = []
        ssd_knife=[]
        silhouette_knife=[]

        for n_clusters in tqdm(range_n_clusters):
            fig, ax1 = plt.subplots(1, 1)
            fig.set_size_inches(7, 4)

            clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=10)
            cluster_labels = clusterer.fit_predict(self.X)

            n_clus_knife.append(n_clusters)
            ssd_knife.append(round(clusterer.inertia_,1))
            silhouette_knife.append(round(silhouette_score(self.X, cluster_labels),3),)

            print(
                "n_clusters =",
                n_clusters,
                'ssd =',
                round(clusterer.inertia_,1),
                "average silhouette_score =",
                round(silhouette_score(self.X, cluster_labels),3),)

            if knife:

                sample_silhouette_values = silhouette_samples(self.X, cluster_labels)

                y_lower = 10
                for i in range(n_clusters):
                    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
                    ith_cluster_silhouette_values.sort()
                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i
                    color = cm.nipy_spectral(float(i) / n_clusters)
                    ax1.fill_betweenx(
                        np.arange(y_lower, y_upper),
                        0,
                        ith_cluster_silhouette_values,
                        facecolor=color,
                        edgecolor=color,
                        alpha=0.7,)

                    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                    y_lower = y_upper + 10

                ax1.set_title("Silhouette plot for n_clusters = %d" % n_clusters)
                ax1.set_xlabel("Silhouette coefficient values")
                ax1.set_ylabel("Cluster label")
                ax1.axvline(x=silhouette_score(self.X, cluster_labels), color="red", linestyle="--")
                ax1.set_yticks([])
                ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

                plt.show()

        df = pd.DataFrame({
            'ssd':ssd_knife,
            'silhouette':silhouette_knife,
            'diff':pd.Series(ssd_knife).diff()
            },index=range_n_clusters)

        return df.transpose()

    @elapsed_time_decorator
    def dbscan_outliers(self,percent=1,save=None):

        # DESCRIPTION:

        #     Build diagram to show amount of outliers based on tested parameter (eps or dot) in knife method

        # ARGUMENTS:

        #     percent - build hlines on diagram for better visualization
        
        sns.lineplot(x=self.check_range,y=self.outlier_percent)
        plt.ylabel("Percentage of Points Classified as Outliers")
        plt.xlabel("Check Value")
        plt.hlines(y=percent,xmin=self.xmin,xmax=self.xmax,colors='red',ls='--')
        plt.grid(alpha=0.2)
        if save =='save':
            plt.savefig('my_plot.png')
        plt.show()

    @elapsed_time_decorator
    def dbscan_knife(self,mod = 'eps',min_eps=0.01,max_eps=1,range_eps=10,min_sample = 1,max_sample = 5,epsindot=0.5,dotineps=5,knife=True):

        # DESCRIPTION:

        #     Build selected range of clusters and represent knife metric to ich of them
        #     'Knifes' should be similar to ich other and have good shape without leakeges

        # ARGUMENTS:

        #     mod - different way for assesment density dependts on eps or dot
        #     min_eps - minimum parameter for eps in case of use mod='eps'
        #     max_eps - maximum parameter for eps in case of use mod='eps'
        #     range_eps - amount of eps parameters tested for eps in case of use mod='eps'
        #     min_sample - minimum amount of dots in area in case of use mod='dot'
        #     max_sample - maximum amount of dots in area in case of use mod='dot'
        #     epsindot - static parameter for eps in case of use mod='dot'
        #     dotineps - static parameter for dot in case of use mod='eps'

        check_values = []
        silhouette=[]
        self.outlier_percent = []
        amount_of_clusters = []
        
        if mod == 'eps':
            self.xmin = min_eps
            self.xmax = max_eps
            
        if mod == 'dot':
            self.xmin = min_sample
            self.xmax = max_sample            
        
        if mod == 'eps':
            self.check_range = np.linspace(min_eps,max_eps,range_eps)
            
        if mod == 'dot':
            self.check_range = range(min_sample,max_sample+1)

        for check_n in tqdm(self.check_range):
            fig, ax1 = plt.subplots(1, 1)
            fig.set_size_inches(7, 4)

            if mod == 'eps':
                clusterer = DBSCAN(eps=check_n,min_samples=dotineps)
                
            if mod == 'dot':
                clusterer = DBSCAN(min_samples=check_n,eps=epsindot)

            try:
                
                cluster_labels = clusterer.fit_predict(self.X)
                
                perc_outliers = 100 * np.sum(clusterer.labels_ == -1) / len(clusterer.labels_)
                self.outlier_percent.append(perc_outliers)
                
                amount_of_clusters.append(str(len(set(cluster_labels))))
                check_values.append(str(check_n))
                silhouette.append(str(round(silhouette_score(self.X, cluster_labels),3),))

                print(
                    "check_n =",
                    check_n,
                    "average silhouette_score =",
                    round(silhouette_score(self.X, cluster_labels),3),
                    'outliers =',
                    perc_outliers
                    )

                if knife:

                    sample_silhouette_values = silhouette_samples(self.X, cluster_labels)

                    y_lower = 10
                    for i in np.unique(cluster_labels):
                        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
                        ith_cluster_silhouette_values.sort()
                        size_cluster_i = ith_cluster_silhouette_values.shape[0]
                        y_upper = y_lower + size_cluster_i
                        color = cm.nipy_spectral(float(i) / len(np.unique(cluster_labels)))
                        ax1.fill_betweenx(
                            np.arange(y_lower, y_upper),
                            0,
                            ith_cluster_silhouette_values,
                            facecolor=color,
                            edgecolor=color,
                            alpha=0.7,)

                        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                        y_lower = y_upper + 10

                    ax1.set_title("Silhouette plot for {} clusters".format(len(set(cluster_labels))))
                    ax1.set_xlabel("Silhouette coefficient values")
                    ax1.set_ylabel("Cluster label")
                    ax1.axvline(x=silhouette_score(self.X, cluster_labels), color="red", linestyle="--")
                    ax1.set_yticks([])
                    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

                    plt.show()

            except ValueError:

                continue

        df = pd.DataFrame({
            'n_clusters':amount_of_clusters,
            'silhouette':silhouette,
            'outliers':self.outlier_percent,
            },index=self.check_range)

        return df.transpose()

    @elapsed_time_decorator
    def build_clusterer(self,clusterer='kmean',param='n_clusters',value=2,target='cluster',alpha=0.5):

        # DESCRIPTION:

        #     Build selected amount of cluster

        # ARGUMENTS:

        #     n_clusters - amount of cluster
        #     target - parameter hue for scatterplot to vizualize clusters on PCA data
        #     alpha - parameter alpha for scatterplot
        
        if clusterer == 'agglo':
            if param == 'n_clusters':
                self.clusterer = AgglomerativeClustering(n_clusters=value,distance_threshold=None)
            elif param == 'distance_threshold':
                self.clusterer = AgglomerativeClustering(distance_threshold=value,n_clusters=None)

        elif clusterer == 'kmean':
            if param == 'n_clusters':
                self.clusterer = KMeans(n_clusters=value)

        elif clusterer == 'dbscan':
            if param == 'eps':
                self.clusterer = DBSCAN(eps=value)
            elif param == 'min_samples':
                self.clusterer = DBSCAN(min_samples=value)

        cluster_labels = self.clusterer.fit_predict(self.X)

        self.result = self.df
        self.result['cluster'] = cluster_labels

    def get_pipe(self):

        self.build_pipe = make_pipeline(self.scaler,self.clusterer)

        return self.build_pipe

    def preprocessing(self,scaler='StandardScaler'):

        # DESCRIPTION:

        #     Preprocess the data with MinMaxScaler
        
        if scaler == 'StandardScaler' : scaler = StandardScaler()
        if scaler == 'MinMaxScaler'   : scaler = MinMaxScaler()

        scaled_data = scaler.fit_transform(self.X)
        self.X = pd.DataFrame(scaled_data)

        self.scaler = scaler
        
    @elapsed_time_decorator
    def simple_check(self,mode='origin',target='cluster',alpha=0.5):

        # DESCRIPTION:

        #     Check data with PCA preprocessing including DataFrame builded in build_knife and build_simple method

        # ARGUMENTS:

        #     mode - tested mode, data from wich DataFrame will be taken
        #     target - hue for scatterplot diagram in case of using cluster = 'origin'
        #     alpha - alpha for scatterplot

        if mode == 'origin':
            X = self.df
            df = self.df
            target = self.df[target]

        if mode == 'build':
            X = self.result
            df = self.result

        if target == 'outliers':
            target = np.where(df['cluster'] == -1, 'outliers', 'normal')

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X)
        X = pd.DataFrame(principal_components)

        print(f'pca.explained_variance_ratio_ = {pca.explained_variance_ratio_}')
        print(f'np.sum(pca.explained_variance_ratio_ = {np.sum(pca.explained_variance_ratio_)}')

        plt.figure(figsize=(8,6))
        sns.scatterplot(x=X[0],y=X[1],data=df,hue=target,alpha=alpha)
        plt.xlabel('First principal component')
        plt.ylabel('Second Principal Component')
        plt.show()

    @elapsed_time_decorator
    def linkage(self,save=None):

        # DESCRIPTION:

        #     Check the optimal amount of cluster by scipy.cluster.hierarchy

        Z = linkage(self.X, 'ward')

        fig = plt.figure(figsize=(10, 5))
        dn = dendrogram(Z, color_threshold=np.sqrt(len(self.X.columns)))
        plt.xticks(rotation=90);

        if save =='save':
            plt.savefig('my_plot.png')

        plt.show()

    def pie(self,bins=20):

        # DESCRIPTION:

        #     Build distribution diagram

        if len(self.result['cluster'].value_counts()) < 10:

            cluster_counts = self.result['cluster'].value_counts()
            plt.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%')

        else:

            sns.displot(data=self.result, x='cluster', kde=True, color='green', bins=bins)

        plt.show()

    def heat(self,sh=12,vi=4,save=None):

        # DESCRIPTION:

        #     Build feature correlation diagram for 'knife' or 'simple' algorithm

        # ARGUMENTS:

        #     cluster - tested mode
        #     sh - height of diagram
        #     vi - length of diagram

        cat_means = self.result.groupby('cluster').mean()

        scaler = MinMaxScaler()
        data = scaler.fit_transform(cat_means)
        scaled_means = pd.DataFrame(data,cat_means.index,cat_means.columns)

        if scaled_means.reset_index().iloc[0]['cluster']==-1:

            ax=plt.figure(figsize=(sh,vi),dpi=200)
            ax=sns.heatmap(scaled_means.iloc[1:],annot=True,cmap='Greens')

        else:

            ax=plt.figure(figsize=(sh,vi),dpi=200)
            ax=sns.heatmap(scaled_means,annot=True,cmap='Greens')

        if save =='save':
            plt.savefig('my_plot.png')

        return ax

    def dendrogram(self,k,save=None):

        # DESCRIPTION:

        #     By using this diagram possible make assesment of choosen amount of cluster on actual data

        # ARGUMENTS:

        #     k - choosen amount of cluster

        Z = linkage(self.X, 'ward')

        clusters = fcluster(Z, k, criterion='maxclust')

        plt.figure(figsize=(10, 8))
        plt.title('Dendrogram')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')
        dendrogram(Z, leaf_rotation=90, leaf_font_size=8., labels=np.arange(1, len(self.X)+1), color_threshold=Z[-k+1, 2])
        plt.axhline(y=Z[-k+1, 2], color='r', linestyle='--')
        plt.xticks(rotation=90);
        if save =='save':
            plt.savefig('my_plot.png')
        plt.show()


