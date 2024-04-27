import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity


# Desabilitar warnings
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('/content/drive/MyDrive/healthcare_dataset.csv')

df.drop(columns=['Name', 'Doctor', 'Hospital', 'Room Number', 'Date of Admission', 'Discharge Date'], axis = 1, inplace = True)

df.info()

# fundo preto
plt.style.use('dark_background')

# Ajustar espaçamento vertical dos subplots
plt.subplots_adjust(hspace=0.5)

# Criar figura e eixos
fig, axs = plt.subplots(5, 2, figsize=(15, 28))
i = 0
# Loop pelas colunas do DataFrame
for coluna in df.columns:
    # Criar o histograma pra cada coluna
    sns.histplot(data = df, x = coluna, ax = axs[i//2, i%2], palette = 'cool')
    i += 1

# Exibir o plot
plt.show()

# Dividir idades por gênero
idade_masc = [df['Age'][i] for i in range(len(df['Age'])) if df['Gender'][i] == 'Male']
idade_fem = [df['Age'][i] for i in range(len(df['Age'])) if df['Gender'][i] == 'Female']

# Criar histograma
plt.style.use('dark_background')
plt.figure(figsize = (12, 6))
plt.hist([idade_masc, idade_fem], bins = 7, color = ['darkblue', 'purple'], label = ['Male', 'Female'])

# Adicionar rótulos e título
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Histogram by Gender')
plt.legend()

# Exibi
plt.show()

# Criar o boxplot
plt.figure(figsize = (10, 6))
sns.boxplot(data = df, x = 'Insurance Provider', y = 'Billing Amount', palette = 'cool')

# Adicionar título e rótulos
plt.title('Boxplot of Charge Amount by Insurance Provider')
plt.xlabel('Insurance Provider')
plt.ylabel('Billing Amount')

# Exibir o plot
plt.show()

# Definir estilo com fundo preto
plt.style.use('dark_background')

# Gráfico de Dispersão para Idade vs. Valor de Cobrança
plt.figure(figsize = (8, 6))
plt.scatter(df['Age'], df['Billing Amount'], alpha = 0.5, color = 'darkblue')
plt.title('Age vs. Billing Amount')
plt.xlabel('Age')
plt.ylabel('Billing Amount')
plt.show()

# colunas categoricas
cat_cols = df.select_dtypes(include = ['object']).columns

# encode valores
le = LabelEncoder()
for col in cat_cols:
    le.fit(df[col])
    df[col] = le.transform(df[col])

# grafico para analisar a correlação das variaveis
plt.figure(figsize = (10, 8))
sns.heatmap(df.corr(), annot = True)

# Assuming df contains your dataset

# Train and test split
X = df.drop('Test Results', axis=1)
y = df['Test Results']
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize the data (important for K-means)
scaler = StandardScaler()
xtrain_standardized = scaler.fit_transform(xtrain)
xtest_standardized = scaler.transform(xtest)

# Initialize the K-means classifier
k = 3  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=0)

# Fit the model
kmeans.fit(xtrain_standardized)

# Predict clusters for test data
test_clusters = kmeans.predict(xtest_standardized)

# Assign labels to clusters
cluster_labels = {}
for cluster in range(k):
    cluster_indices = (test_clusters == cluster)
    cluster_labels[cluster] = ytest[cluster_indices].mode()[0]

# Predictions based on cluster labels
ypred = [cluster_labels[cluster] for cluster in test_clusters]

# Evaluate the model
accuracy = accuracy_score(ytest, ypred)
silhouette = silhouette_score(xtest_standardized, test_clusters)
precision = precision_score(ytest, ypred, average='weighted')

print("Accuracy:", accuracy)
print("Silhouette Score:", silhouette)
print("Precision:", precision)

# Apply PCA to reduce dimensionality for visualization
pca = PCA(n_components=2)
xtest_pca = pca.fit_transform(xtest_standardized)

# Plot the clusters
plt.figure(figsize=(8, 6))
for cluster in range(k):
    cluster_indices = (test_clusters == cluster)
    plt.scatter(xtest_pca[cluster_indices, 0], xtest_pca[cluster_indices, 1], label=f'Cluster {cluster}')
plt.title('Clusters in 2D (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

xtrain_standardized = scaler.fit_transform(xtrain)
xtest_standardized = scaler.transform(xtest)

# Initialize the DBSCAN classifier with adjusted parameters
eps = 0.5  # Adjust this parameter
min_samples = 5  # Adjust this parameter
dbscan = DBSCAN(eps=eps, min_samples=min_samples)

# Fit the model
dbscan.fit(xtrain_standardized)

# Predict clusters for test data
test_clusters = dbscan.fit_predict(xtest_standardized)

# Evaluate the model
# You need to define ypred here, or you can remove the evaluation part since DBSCAN is an unsupervised clustering algorithm


# Evaluate the model
#accuracy = accuracy_score(ytest, ypred)
#silhouette = silhouette_score(xtest_standardized, test_clusters)
#precision = precision_score(ytest, ypred, average='weighted')

#print("Accuracy:", accuracy)
#print("Silhouette Score:", silhouette)
#print("Precision:", precision)

# Apply PCA to reduce dimensionality for visualization
pca = PCA(n_components=2)
xtest_pca = pca.fit_transform(xtest_standardized)


# Plot the clusters
plt.figure(figsize=(8, 6))
unique_labels = set(test_clusters)
for label in unique_labels:
    if label == -1:
        # Noise points
        cluster_mask = test_clusters == label
        plt.scatter(xtest_pca[cluster_mask, 0], xtest_pca[cluster_mask, 1], label=f'Noise', c='gray', alpha=0.5)
    else:
        cluster_mask = test_clusters == label
        plt.scatter(xtest_pca[cluster_mask, 0], xtest_pca[cluster_mask, 1], label=f'Cluster {label}')
plt.title('Clusters in 2D (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# Compute silhouette score
silhouette = silhouette_score(xtest_standardized, test_clusters)
print("Silhouette Score:", silhouette)

# Density plot
plt.figure(figsize=(8, 6))
plt.hist(test_clusters, bins=len(np.unique(test_clusters)), density=True, alpha=0.5)
plt.title('Density Plot of Clusters')
plt.xlabel('Cluster')
plt.ylabel('Density')
plt.show()

# Define parameters to test
num_clusters_list = [2, 3, 4, 5]  # Test different numbers of clusters
distance_metrics = ['euclidean', 'manhattan', 'canberra', 'minkowski']  # Test different distance metrics

# Create a separate plot for each distance metric
for metric in distance_metrics:
    # Initialize lists to store silhouette scores for each number of clusters
    silhouette_scores = []

    # Iterate over number of clusters
    for k in num_clusters_list:
        # Initialize the K-means classifier
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10, algorithm='auto', max_iter=300, init='k-means++')

        # Fit the model
        kmeans.fit(xtrain_standardized)

        # Predict clusters for test data
        test_clusters = kmeans.predict(xtest_standardized)

        # Calculate silhouette score
        silhouette = silhouette_score(xtest_standardized, test_clusters, metric=metric)

        # Append silhouette score to the list
        silhouette_scores.append(silhouette)

    # Plot the silhouette scores for the current distance metric
    plt.figure(figsize=(8, 6))
    plt.plot(num_clusters_list, silhouette_scores, marker='o')
    plt.title(f'Silhouette Scores for {metric.capitalize()} Distance')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.xticks(num_clusters_list)
    plt.show()
