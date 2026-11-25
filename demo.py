import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from kmeans import kmeans, make_customers

X = StandardScaler().fit_transform(make_customers())
for k in (2, 3, 4, 5):
    lab, _, inertia = kmeans(X, k)
    print(f"k={k} inertia={inertia:8.1f} silhouette={silhouette_score(X, lab):.3f}")
lab, C, inertia = kmeans(X, 3)
ref = KMeans(3, n_init=10, random_state=0).fit(X)
print(f"from-scratch inertia (k=3): {inertia:.1f}  sklearn inertia: {ref.inertia_:.1f}")
print("cluster sizes:", np.bincount(lab).tolist())
