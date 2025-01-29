# The following platforms had been utilized to source and reference codes
    # - ChatGPT
    # - W3Schools

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def main():
    # User will input to start the program and define the number of clusters as the values
    print("\nTHIS IS THE STOCK ANALYSIS PROGRAM.")
    while True:
        try:
            n_clusters = int(input("\nEnter the number of clusters (between 1 and 10): "))
            if 1 <= n_clusters <= 10:
                break
            else:
                print("Please enter a value between 1 and 10.")
        except ValueError:
            print("Invalid input. Please enter an integer between 1 and 10.")

 
    file_path = 'TESLA.csv'
    try:
        tesla_data = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: File not found. Please check the file path and try again.")
        return

   
    tesla_data['Date'] = pd.to_datetime(tesla_data['Date'])


    numerical_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    scaler = MinMaxScaler()
    tesla_data_scaled = tesla_data.copy()
    tesla_data_scaled[numerical_columns] = scaler.fit_transform(tesla_data[numerical_columns])

    # Linear Regression
    
    X = tesla_data_scaled[['Open', 'High', 'Low', 'Volume']]
    y = tesla_data_scaled['Close']

    random_seed = np.random.randint(1, 1000)  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)


    y_pred = linear_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nLinear Regression MSE: {mse}")
    print(f"Linear Regression R2 Score: {r2}")

    # Visualization of Linear Regression Results
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.7, color='b')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    plt.title('Linear Regression: Actual vs Predicted')
    plt.xlabel('Actual Closing Prices')
    plt.ylabel('Predicted Closing Prices')
    plt.show()
    


    # K-Means Clustering

    # Distinguishing optimal clusters through Elbow method
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_seed)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)

    # Plot the Elbow Method
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.show()

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed)
    clusters = kmeans.fit_predict(X)

    tesla_data_scaled['Cluster'] = clusters

    # Visualize clusters through PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 5))
    for cluster in range(n_clusters):
        plt.scatter(
            X_pca[clusters == cluster, 0], 
            X_pca[clusters == cluster, 1], 
            label=f'Cluster {cluster}'
        )
    plt.title('K-Means Clustering Results')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()