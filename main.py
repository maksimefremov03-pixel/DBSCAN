import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
import pandas as pd


def region_query(X, point_i, eps):
    distances = np.linalg.norm(X - X[point_i], axis=1)
    return np.where(distances <= eps)[0]


def expand_cluster(X, labels, core_idx, neighbors, cluster_id, eps, min_samples):
    labels[core_idx] = cluster_id
    i = 0

    while i < len(neighbors):
        point_idx = neighbors[i]

        if labels[point_idx] == -1:
            labels[point_idx] = cluster_id

            point_neighbors = region_query(X, point_idx, eps)
            if len(point_neighbors) >= min_samples:
                neighbors = np.concatenate([neighbors, point_neighbors])

        i += 1


def dbscan_fit(X, eps, min_samples):
    labels = np.full(X.shape[0], -1)
    cluster_id = 0

    for i in range(len(X)):
        if labels[i] != -1:
            continue

        neighbors = region_query(X, i, eps)

        if len(neighbors) < min_samples:
            labels[i] = -1
        else:
            expand_cluster(X, labels, i, neighbors, cluster_id, eps, min_samples)
            cluster_id += 1

    return labels


datasets = [
    make_blobs(n_samples=300, centers=3, random_state=42),
    make_moons(n_samples=300, noise=0.05, random_state=42),
    make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)
]

dataset_names = ['Blobs', 'Moons', 'Circles']
eps_values = [0.3, 0.4, 0.5]
min_samples_values = [3, 5, 7]

results = []

for dataset_i, (X, y_true) in enumerate(datasets):
    X_scaled = StandardScaler().fit_transform(X)
    dataset_name = dataset_names[dataset_i]

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle(f'DBSCAN: {dataset_name} Dataset\n',
                 fontsize=16, fontweight='bold')

    plot_idx = 0

    for eps in eps_values:
        for min_samples in min_samples_values:
            y_pred = dbscan_fit(X_scaled, eps, min_samples)

            # Оценка точности
            ari = adjusted_rand_score(y_true, y_pred)

            # Подсчёт кластеров
            n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
            n_correct = np.sum(y_pred != -1)
            n_incorrect = np.sum(y_pred == -1)

            results.append({
                'Dataset': dataset_name,
                'Eps': eps,
                'Min_samples': min_samples,
                'Total_points': len(X),
                'Correct_clustered': n_correct,
                'Incorrect_clustered': n_incorrect,
                'N_Clusters': n_clusters,
                'ARI': ari
            })

            # индексы для 3x3 сетки
            row_idx = plot_idx // 3
            col_idx = plot_idx % 3
            ax = axes[row_idx, col_idx]

            unique_labels = set(y_pred)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

            for k, col in zip(unique_labels, colors):
                if k == -1:
                    col = [0, 0, 0, 1]  # черный для шума

                class_member_mask = (y_pred == k)
                xy = X_scaled[class_member_mask]
                ax.scatter(xy[:, 0], xy[:, 1], c=[col], edgecolors='k', s=30, alpha=0.7)

            ax.set_title(f'eps={eps}, min_samples={min_samples}\nClusters: {n_clusters}, ARI: {ari:.3f}',
                         fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

            plot_idx += 1

    plt.tight_layout()
    plt.show()

results = pd.DataFrame(results)
print("ПОЛНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ:")
print("=" * 80)
print(results.to_string(index=False))

print("\n" + "=" * 80)
print("АНАЛИЗ РЕЗУЛЬТАТОВ:")
print("=" * 80)

for dataset_name in dataset_names:
    dataset_results = results[results['Dataset'] == dataset_name]
    best_result = dataset_results.loc[dataset_results['ARI'].idxmax()]

    print(f"\n{dataset_name}:")
    print(f"  Лучшие параметры: eps={best_result['Eps']}, min_samples={best_result['Min_samples']}")
    print(f"  Качество (ARI): {best_result['ARI']:.3f}")
    print(f"  Найдено кластеров: {best_result['N_Clusters']}")
    print(f"  Точек в кластерах: {best_result['Correct_clustered']}")
    print(f"  Шумовых точек: {best_result['Incorrect_clustered']}")


plt.figure(figsize=(12, 10))

for i, dataset_name in enumerate(dataset_names):
    plt.subplot(1, 3, i + 1)

    dataset_results = results[results['Dataset'] == dataset_name]

    ari_matrix = np.zeros((len(eps_values), len(min_samples_values)))

    for j, eps in enumerate(eps_values):
        for k, min_samples in enumerate(min_samples_values):
            mask = (dataset_results['Eps'] == eps) & (dataset_results['Min_samples'] == min_samples)
            ari_value = dataset_results.loc[mask, 'ARI'].values[0]
            ari_matrix[j, k] = ari_value

    im = plt.imshow(ari_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    plt.xticks(np.arange(len(min_samples_values)), min_samples_values)
    plt.yticks(np.arange(len(eps_values)), eps_values)
    plt.xlabel('min_samples')
    plt.ylabel('eps')
    plt.title(f'{dataset_name}\nТочность кластеризации (ARI)')

    for j in range(len(eps_values)):
        for k in range(len(min_samples_values)):
            text = plt.text(k, j, f'{ari_matrix[j, k]:.3f}',
                            ha="center", va="center", color="black", fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.suptitle('ТОЧНОСТЬ КЛАСТЕРИЗАЦИИ DBSCAN (Adjusted Rand Index)',
             fontsize=16, fontweight='bold')
plt.colorbar(im, ax=plt.gcf().get_axes(), shrink=0.8, aspect=20)
plt.show()