import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
'''
linalg是numpy中关于计算矩阵的线性运算，比如矩阵的逆，矩阵的特征值等
'''
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import normalize
from sklearn.datasets import make_blobs
import csv
gimma = 0.2
def load_dataset():
    csvFile = open('data/twocircles.csv','r')
    reader = csv.reader(csvFile)
    original_data = []
    for item in reader:
        original_data.append(list(map(eval,item)))
    original_data = np.array(original_data)
    return original_data
def similarity_function(points):
    '''
    计算相似度矩阵，并且把对角线元素设置为0
    :param points:
    :return:
    '''
    sim_matrix = rbf_kernel(points,gamma=1/(2*gimma*gimma))
    for i in range(len(sim_matrix)):
        sim_matrix[i,i] = 0
    return sim_matrix
def spectal_clustering(points,k):
    '''
    整个谱聚类的过程是：
                    （1）生成相似度矩阵W
                    （2）生成对角矩阵D，类似于图中的度矩阵，即W矩阵的行或者列的和，
                    （3）构造拉普拉斯矩阵，L=D^(-0.5)*(D-W)*D^(-0.5)
                    （4）计算拉普拉斯矩阵L的前k个最大的特征值，和对应的特征向量，特征向量按照列排列，生成矩阵X
                    此处要注意：拉普拉斯矩阵若是采用的
                    （5）归一化矩阵X，生成Y
                    （6）对Y的每一行视作一个样本，进行k-means
    :param points:
    :param k:
    :return:
    '''
    W = similarity_function(points)
    D = np.diag(np.sum(W,axis=1))   ##axis=1是按照列的方向求和。也可以用axis=0，因为w是对称的矩阵，按照行或者列都是一样
    Dn = LA.inv(np.sqrt(D))         ##Dn就是D^(-0.5)
    #L = np.dot(np.dot(Dn,D-W),Dn)
    L = np.eye(len(points)) - np.dot(np.dot(Dn, W), Dn)
    eigvalues,eigvectors = LA.eig(L)
    indexs = np.argsort(eigvalues)[:k]     ##此处的含义是将特征值排列，取前k个最小特征值的索引
    k_smallest_eigvectors = normalize(eigvectors[:,indexs])
    return KMeans(n_clusters=k).fit_predict(k_smallest_eigvectors),D,Dn
x,y = make_blobs(n_samples=100,n_features=2,centers=4,cluster_std=[1.8,1,0.5,1,3],random_state=True)
labels,_,_ = spectal_clustering(x, 4)
kmeans = KMeans(n_clusters=4).fit_predict(x)
plt.style.use('ggplot')
# original data
fig, (ax0, ax1, ax2,ax4,ax5) = plt.subplots(ncols=5)
ax0.scatter(x[:, 0], x[:, 1], c=y)
ax0.set_title('original data')
# kmeans
ax2.scatter(x[:,0],x[:,1],c=kmeans)
ax2.set_title('kmeans data')
# the spectral clustering data
ax1.scatter(x[:, 0], x[:, 1], c=labels)
ax1.set_title('Spectral Clustering')
'''
此处是用twicircles中的数据作为样本
'''
data = load_dataset()
kmeans_csv_data = KMeans(n_clusters=2).fit_predict(data)
labels_csv_data,D,Dn = spectal_clustering(data,2)
ax4.scatter(data[:,0],data[:,1],c=kmeans_csv_data)
ax4.set_title('kmeans csvdata')
ax5.scatter(data[:,0],data[:,1],c=labels_csv_data)
ax5.set_title('spectral csvdata')
plt.show()