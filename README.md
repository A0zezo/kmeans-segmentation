# kmeans-segmentation

K-means with k-means++ initialisation written in NumPy, run on 300 synthetic customers (orders per year, spend, days since last order) drawn from 3 groups, and compared with scikit-learn.

## Run

```bash
pip install -r requirements.txt
python demo.py
pytest
```

## Output (reproduced)

```
$ python3 demo.py
k=2 inertia=   382.3 silhouette=0.583
k=3 inertia=    73.4 silhouette=0.760
k=4 inertia=    60.3 silhouette=0.644
k=5 inertia=    48.6 silhouette=0.513
from-scratch inertia (k=3): 73.4  sklearn inertia: 73.4
cluster sizes: [100, 100, 100]

$ pytest -q
3 passed in 0.07s
```

## Limits

- Synthetic data. A single k-means run can land in a poor local minimum, so the code keeps the best of 10 restarts. The k=3 result matches scikit-learn's inertia in the output above; compare the two inertia numbers yourself.
