class agglo():

    def __init__(self,X):
        
        self.df = X
        self.X = X

    def knife(self,min_k,max_k):
        
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_samples, silhouette_score
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np

        range_n_clusters = list(range(min_k,max_k))

        self.n_clus = []
        self.silhouette=[]        

        for n_clusters in range_n_clusters:
            fig, ax1 = plt.subplots(1, 1)
            fig.set_size_inches(7, 4)

            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(self.X)

            self.n_clus.append(n_clusters)
            self.silhouette.append(round(silhouette_score(self.X, cluster_labels),3),)

            print(
                "n_clusters =",
                n_clusters,
                "average silhouette_score =",
                round(silhouette_score(self.X, cluster_labels),3),)

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

    def knife_df(self):

        import pandas as pd
        
        df = pd.DataFrame({
            'n_clusters':self.n_clus,
            'silhouette':self.silhouette
            })

        return df
    
    def distance(self,min_d=1,max_d=5,range_d=10,save=None):
        
        import seaborn as sns
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.metrics import silhouette_score
        from sklearn.cluster import AgglomerativeClustering

        distance_threshold_list = list(np.linspace(min_d,max_d,range_d))

        self.n_claster_dis = []
        self.distance_threshold = []
        self.silhouette_dis = []
    
        for distance_threshold_n in distance_threshold_list:
            
            clusterer = AgglomerativeClustering(n_clusters=None,distance_threshold=distance_threshold_n)
            cluster_labels = clusterer.fit_predict(self.X)
            
            self.distance_threshold.append(distance_threshold_n)
            self.n_claster_dis.append(len(np.unique(cluster_labels)))
            self.silhouette_dis.append(round(silhouette_score(self.X, cluster_labels),3),)
            
    
        plt.figure(figsize=(12,5),dpi=200)
        sns.scatterplot(x=self.n_claster_dis,y=self.distance_threshold,hue=self.silhouette_dis,s=100)
        plt.title("Distance plot")
        plt.xlabel("Number of claster")
        plt.ylabel("Distance_threshold")
        plt.xticks(range(min(self.n_claster_dis), max(self.n_claster_dis)+1))
        plt.yticks(np.arange(0, max(self.distance_threshold)+1, step=1))
        plt.legend(title='Silhouette', loc='upper right')

        if save =='save':
            plt.savefig('my_plot.png')
        
        plt.show()

    def distance_df(self):
        
        import pandas as pd

        df = pd.DataFrame({
            'n_clusters':self.n_claster_dis,
            'distance_threshold':self.distance_threshold,
            'silhouette':self.silhouette_dis
            })

        return df

    def linkage(self,save=None):

        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram, linkage

        Z = linkage(self.X, 'ward')

        fig = plt.figure(figsize=(10, 5))
        dn = dendrogram(Z, color_threshold=np.sqrt(len(self.X.columns)))
        plt.xticks(rotation=90);

        if save =='save':
            plt.savefig('my_plot.png')

        plt.show()

    def minmax(self):

        from sklearn.preprocessing import MinMaxScaler
        import pandas as pd
        
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(self.X)
        self.X = pd.DataFrame(scaled_data,self.X.index)
        
    def dendrogram(self,k,save=None):
        
        import numpy as np
        from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
        import matplotlib.pyplot as plt

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
        
    def build_knife(self,n,target='cluster',alpha=0.5):
        
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.decomposition import PCA
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        
        clusterer = AgglomerativeClustering(n_clusters=n)
        cluster_labels = clusterer.fit_predict(self.X)
        
        cluster_df = pd.DataFrame(cluster_labels, index=self.df.index, columns=['cluster'])
        self.result = pd.concat([self.df, cluster_df], axis=1)

        X = self.result

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X)
        X = pd.DataFrame(principal_components)

        print(f'pca.explained_variance_ratio_ = {pca.explained_variance_ratio_}')
        print(f'np.sum(pca.explained_variance_ratio_ = {np.sum(pca.explained_variance_ratio_)}')

        plt.figure(figsize=(8,6))
        sns.scatterplot(x=X[0],y=X[1],data=self.result,hue=target,alpha=alpha)
        plt.xlabel('First principal component')
        plt.ylabel('Second Principal Component')
        plt.show()

    def pie(self,cluster,save=None):

        import seaborn as sns
        import matplotlib.pyplot as plt

        if cluster == 'knife':
            df = self.result

        if cluster == 'simple':
            df = self.result_simple

        corr_df = df.corr()

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 6), dpi=200)

        sns.barplot(x=corr_df['cluster'].sort_values().iloc[1:-1].index, y=corr_df['cluster'].sort_values().iloc[1:-1].values, ax=ax1)
        ax1.set_title("Feature Correlation to Cluster")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)

        if len(df['cluster'].value_counts()) < 10:
            cluster_counts = df['cluster'].value_counts()
            ax2.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%')
        else:
            sns.displot(data=df, x='cluster', kde=True, color='green', bins=20, ax=ax2)

        if save =='save':
            plt.savefig('my_plot.png')

        plt.show()

    def heat(self,cluster,sh=12,vi=4,save=None): 

        from sklearn.preprocessing import MinMaxScaler
        import seaborn as sns
        import pandas as pd
        import matplotlib.pyplot as plt

        if cluster == 'knife':
            cat_means = self.result.groupby('cluster').mean()

        if cluster == 'simple':
            cat_means = self.result_simple.groupby('cluster').mean()

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

    def feature_corr(self,cluster,sh=12,vi=4,dpi=200,target='cluster',save=None):

        import seaborn as sns
        import matplotlib.pyplot as plt

        if cluster == 'knife':
            df  = self.result
            corr_df = df.corr()
        if cluster == 'simple':
            df  = self.result_simple
            corr_df = df.corr()

        x = corr_df.abs().sort_values(target).index[-2]
        y = corr_df.abs().sort_values(target).index[-3]

        ax=plt.figure(figsize=(sh,vi),dpi=dpi)
        sns.scatterplot(x=x, y=y, data=df, hue='cluster')
        if save =='save':
            plt.savefig('my_plot.png')
        plt.show()

    def simple_check(self,cluster='origin',target='cluster',alpha=0.5):

        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        if cluster == 'origin':
            X = self.df
            df=self.df
            target = self.df.columns[0]

        if cluster == 'knife':
            X = self.result
            df=self.result

        if cluster == 'simple':
            X = self.result_simple
            df=self.result_simple

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

    def build_simple(self,n_clusters,target='cluster',alpha=0.5):

        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.decomposition import PCA
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        X = self.X

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X)
        X = pd.DataFrame(principal_components)

        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(X)
        
        cluster_df = pd.DataFrame(cluster_labels, index=self.X.index, columns=['cluster'])
        self.result_simple = pd.concat([self.df, cluster_df], axis=1)

        print(f'pca.explained_variance_ratio_ = {pca.explained_variance_ratio_}')
        print(f'np.sum(pca.explained_variance_ratio_ = {np.sum(pca.explained_variance_ratio_)}')

        plt.figure(figsize=(8,6))
        sns.scatterplot(x=X[0],y=X[1],data=self.result_simple,hue=target,alpha=alpha)
        plt.xlabel('First principal component')
        plt.ylabel('Second Principal Component')
        plt.show()


class k_mean():

    def __init__(self,X):

        self.df = X
        self.X = X

    def knife(self,min_n,max_n):
        
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_samples, silhouette_score
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np

        range_n_clusters = list(range(min_n,max_n))

        self.n_clus_knife = []
        self.ssd_knife=[]
        self.silhouette_knife=[]

        for n_clusters in range_n_clusters:
            fig, ax1 = plt.subplots(1, 1)
            fig.set_size_inches(7, 4)

            clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=10)
            cluster_labels = clusterer.fit_predict(self.X)

            self.n_clus_knife.append(n_clusters)
            self.ssd_knife.append(round(clusterer.inertia_,1))
            self.silhouette_knife.append(round(silhouette_score(self.X, cluster_labels),3),)

            print(
                "n_clusters =",
                n_clusters,
                'ssd =',
                round(clusterer.inertia_,1),
                "average silhouette_score =",
                round(silhouette_score(self.X, cluster_labels),3),)

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

    def knife_df(self):

        import pandas as pd
        
        df = pd.DataFrame({
            'n_clusters':self.n_clus_knife,
            'ssd':self.ssd_knife,
            'silhouette':self.silhouette_knife,
            'diff':pd.Series(self.ssd_knife).diff()
            })

        return df

    def mount(self,min_n,max_n,save=None):

        import matplotlib.pyplot as plt
        import pandas as pd
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        range_n_clusters = list(range(min_n,max_n))

        self.n_clus_mount=[]
        self.ssd_mount=[]
        self.silhouette_mount=[]

        for n_clusters in range_n_clusters:

            clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=10)
            cluster_labels = clusterer.fit_predict(self.X)

            self.n_clus_mount.append(n_clusters)
            self.ssd_mount.append(round(clusterer.inertia_,1))
            self.silhouette_mount.append(round(silhouette_score(self.X, cluster_labels),3),)


        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

        ax1.plot(range(min_n,max_n), self.ssd_mount, 'o--')
        ax1.set_xlabel("K Value")
        ax1.set_ylabel("Sum of Squared Distances")

        pd.Series(self.ssd_mount).diff().plot(kind='bar', ax=ax2)
        ax2.set_xlabel("K Value")
        ax2.set_ylabel("Change in Sum of Squared Distances")
        plt.tight_layout()
        if save =='save':
            plt.savefig('my_plot.png')
        plt.show()

    def mount_df(self):

        import pandas as pd
        
        df = pd.DataFrame({
            'n_clusters':self.n_clus_mount,
            'ssd':self.ssd_mount,
            'silhouette':self.silhouette_mount,
            'diff':pd.Series(self.ssd_mount).diff()
            })

        return df

    def standard(self):

        from sklearn.preprocessing import StandardScaler
        import pandas as pd
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.X)
        self.X = pd.DataFrame(scaled_data,self.X.index)

    def dendrogram(self,k,save=None):
        
        import numpy as np
        from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
        import matplotlib.pyplot as plt

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

    def build_knife(self,n,target='cluster',alpha=0.5):
        
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        clusterer = KMeans(n_clusters=n)
        cluster_labels = clusterer.fit_predict(self.X)
        
        cluster_df = pd.DataFrame(cluster_labels, index=self.df.index, columns=['cluster'])
        self.result = pd.concat([self.df, cluster_df], axis=1)

        X = self.result

        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X)
        X = pd.DataFrame(principal_components)

        print(f'pca.explained_variance_ratio_ = {pca.explained_variance_ratio_}')
        print(f'np.sum(pca.explained_variance_ratio_ = {np.sum(pca.explained_variance_ratio_)}')

        plt.figure(figsize=(8,6))
        sns.scatterplot(x=X[0],y=X[1],data=self.result,hue=target,alpha=alpha)
        plt.xlabel('First principal component')
        plt.ylabel('Second Principal Component')
        plt.show()

    def build_simple(self,n_clusters,target='cluster',alpha=0.5):

        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        X = self.X

        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X)
        X = pd.DataFrame(principal_components)

        clusterer = KMeans(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(X)
        
        cluster_df = pd.DataFrame(cluster_labels, index=self.X.index, columns=['cluster'])
        self.result_simple = pd.concat([self.df, cluster_df], axis=1)

        print(f'pca.explained_variance_ratio_ = {pca.explained_variance_ratio_}')
        print(f'np.sum(pca.explained_variance_ratio_ = {np.sum(pca.explained_variance_ratio_)}')

        plt.figure(figsize=(8,6))
        sns.scatterplot(x=X[0],y=X[1],data=self.result_simple,hue=target,alpha=alpha)
        plt.xlabel('First principal component')
        plt.ylabel('Second Principal Component')
        plt.show()

    def linkage(self,save=None):

        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram, linkage

        Z = linkage(self.X, 'ward')

        fig = plt.figure(figsize=(10, 5))
        dn = dendrogram(Z, color_threshold=np.sqrt(len(self.X.columns)))
        plt.xticks(rotation=90);
        if save =='save':
            plt.savefig('my_plot.png')

        plt.show()

    def pie(self,cluster,save=None):

        import seaborn as sns
        import matplotlib.pyplot as plt

        if cluster == 'knife':
            df = self.result

        if cluster == 'simple':
            df = self.result_simple

        corr_df = df.corr()

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 6), dpi=200)

        sns.barplot(x=corr_df['cluster'].sort_values().iloc[1:-1].index, y=corr_df['cluster'].sort_values().iloc[1:-1].values, ax=ax1)
        ax1.set_title("Feature Correlation to Cluster")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)

        if len(df['cluster'].value_counts()) < 10:
            cluster_counts = df['cluster'].value_counts()
            ax2.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%')
        else:
            sns.displot(data=df, x='cluster', kde=True, color='green', bins=20, ax=ax2)

        if save =='save':
            plt.savefig('my_plot.png')

        plt.show()

    def heat(self,cluster,sh=12,vi=4,save=None): 

        from sklearn.preprocessing import MinMaxScaler
        import seaborn as sns
        import pandas as pd
        import matplotlib.pyplot as plt


        if cluster == 'knife':
            cat_means = self.result.groupby('cluster').mean()

        if cluster == 'simple':
            cat_means = self.result_simple.groupby('cluster').mean()

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

    def feature_corr(self,cluster,sh=12,vi=4,dpi=200,target='cluster',save=None):

        import seaborn as sns
        import matplotlib.pyplot as plt

        if cluster == 'knife':
            df  = self.result
            corr_df = df.corr()
        if cluster == 'simple':
            df  = self.result_simple
            corr_df = df.corr()

        x = corr_df.abs().sort_values(target).index[-2]
        y = corr_df.abs().sort_values(target).index[-3]

        ax=plt.figure(figsize=(sh,vi),dpi=dpi)
        sns.scatterplot(x=x, y=y, data=df, hue='cluster')
        if save =='save':
            plt.savefig('my_plot.png')
        plt.show()

    def simple_check(self,cluster='origin',target='cluster',alpha=0.5):

        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        if cluster == 'origin':
            X = self.df
            df=self.df
            target = self.df.columns[0]

        if cluster == 'knife':
            X = self.result
            df=self.result

        if cluster == 'simple':
            X = self.result_simple
            df=self.result_simple

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


class dbscan():

    def __init__(self,X):
        
        import pandas as pd
        
        self.df = X
        self.X = X

    def knife(self,mod = 'eps',min_eps=0.01,max_eps=1,range_eps=10,min_sample = 1,max_sample = 5,epsindot=0.5,dotineps=5):
        
        from sklearn.metrics import silhouette_samples, silhouette_score
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
        from sklearn.cluster import DBSCAN

        self.check_values = []
        self.silhouette=[]
        self.outlier_percent = []
        self.amount_of_clusters = []
        
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

        for check_n in self.check_range:
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
                
                self.amount_of_clusters.append(str(len(set(cluster_labels))))
                self.check_values.append(str(check_n))
                self.silhouette.append(str(round(silhouette_score(self.X, cluster_labels),3),))

                print(
                    "check_n =",
                    check_n,
                    "average silhouette_score =",
                    round(silhouette_score(self.X, cluster_labels),3),)

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
            
    def build_knife(self,eps=0.5,min_samples=5,target='cluster',alpha=0.5):
        
        from sklearn.cluster import DBSCAN
        from sklearn.decomposition import PCA
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        clusterer = DBSCAN(eps=eps,min_samples=min_samples)
        cluster_labels = clusterer.fit_predict(self.X)
        
        cluster_df = pd.DataFrame(cluster_labels, index=self.df.index, columns=['cluster'])
        self.result = pd.concat([self.df, cluster_df], axis=1)

        X = self.result

        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X)
        X = pd.DataFrame(principal_components)

        print(f'pca.explained_variance_ratio_ = {pca.explained_variance_ratio_}')
        print(f'np.sum(pca.explained_variance_ratio_ = {np.sum(pca.explained_variance_ratio_)}')

        plt.figure(figsize=(8,6))
        sns.scatterplot(x=X[0],y=X[1],data=self.result,hue=target,alpha=alpha)
        plt.xlabel('First principal component')
        plt.ylabel('Second Principal Component')
        plt.show()

    def build_simple(self,eps=0.5,min_samples=5,target='cluster',alpha=0.5):

        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import DBSCAN
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        X = self.X

        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X)
        X = pd.DataFrame(principal_components)

        clusterer = DBSCAN(eps=eps,min_samples=min_samples)
        cluster_labels = clusterer.fit_predict(X)
        
        cluster_df = pd.DataFrame(cluster_labels, index=self.X.index, columns=['cluster'])
        self.result_simple = pd.concat([self.df, cluster_df], axis=1)

        print(f'pca.explained_variance_ratio_ = {pca.explained_variance_ratio_}')
        print(f'np.sum(pca.explained_variance_ratio_ = {np.sum(pca.explained_variance_ratio_)}')

        plt.figure(figsize=(8,6))
        sns.scatterplot(x=X[0],y=X[1],data=self.result_simple,hue=target,alpha=alpha)
        plt.xlabel('First principal component')
        plt.ylabel('Second Principal Component')
        plt.show()

    def simple_check(self,cluster='origin',target='cluster',alpha=0.5):

        from sklearn.decomposition import PCA
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        if cluster == 'origin':
            X = self.df
            df=self.df
            target = self.df.columns[0]

        if cluster == 'knife':
            X = self.result
            df=self.result

        if cluster == 'simple':
            X = self.result_simple
            df=self.result_simple

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

    def knife_df(self):

        import pandas as pd

        df = pd.DataFrame({
            'check_values':self.check_values[len(self.silhouette)*-1:],
            'silhouette':self.silhouette,
            'n_clusters':self.amount_of_clusters[len(self.silhouette)*-1:],
            'outliers':self.outlier_percent[len(self.silhouette)*-1:]
            })

        return df
    
    def outliers(self,percent=1,save=None):
        
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        sns.lineplot(x=self.check_range,y=self.outlier_percent)
        plt.ylabel("Percentage of Points Classified as Outliers")
        plt.xlabel("Check Value")
        plt.hlines(y=percent,xmin=self.xmin,xmax=self.xmax,colors='red',ls='--')
        plt.grid(alpha=0.2)
        if save =='save':
            plt.savefig('my_plot.png')
        plt.show()
            
    def standard(self):

        from sklearn.preprocessing import StandardScaler
        import pandas as pd
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.X)
        self.X = pd.DataFrame(scaled_data,self.X.index)

    def linkage(self,save=None):

        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram, linkage

        Z = linkage(self.X, 'ward')

        fig = plt.figure(figsize=(10, 5))
        dn = dendrogram(Z, color_threshold=np.sqrt(len(self.X.columns)))
        plt.xticks(rotation=90);
        if save =='save':
            plt.savefig('my_plot.png')

        plt.show()

    def pie(self,cluster,save=None):

        import seaborn as sns
        import matplotlib.pyplot as plt

        if cluster == 'knife':
            df = self.result

        if cluster == 'simple':
            df = self.result_simple

        corr_df = df.corr()

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 6), dpi=200)

        sns.barplot(x=corr_df['cluster'].sort_values().iloc[1:-1].index, y=corr_df['cluster'].sort_values().iloc[1:-1].values, ax=ax1)
        ax1.set_title("Feature Correlation to Cluster")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)

        if len(df['cluster'].value_counts()) < 10:
            cluster_counts = df['cluster'].value_counts()
            ax2.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%')
        else:
            sns.displot(data=df, x='cluster', kde=True, color='green', bins=20, ax=ax2)

        if save =='save':
            plt.savefig('my_plot.png')

        plt.show()

    def heat(self,cluster,sh=12,vi=4,save=None): 

        from sklearn.preprocessing import MinMaxScaler
        import seaborn as sns
        import pandas as pd
        import matplotlib.pyplot as plt


        if cluster == 'knife':
            cat_means = self.result.groupby('cluster').mean()

        if cluster == 'simple':
            cat_means = self.result_simple.groupby('cluster').mean()

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

    def feature_corr(self,cluster,sh=12,vi=4,dpi=200,target='cluster',save=None):

        import seaborn as sns
        import matplotlib.pyplot as plt

        if cluster == 'knife':
            df  = self.result
            corr_df = df.corr()
        if cluster == 'simple':
            df  = self.result_simple
            corr_df = df.corr()

        x = corr_df.abs().sort_values(target).index[-2]
        y = corr_df.abs().sort_values(target).index[-3]

        ax=plt.figure(figsize=(sh,vi),dpi=dpi)
        sns.scatterplot(x=x, y=y, data=df, hue='cluster')
        if save =='save':
            plt.savefig('my_plot.png')
        plt.show()