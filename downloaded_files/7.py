import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import utils.constants as const
from utils.helpers import sample_2_clustered_sample
from utils.vector_generator import generate_norm_vector
from modules.clusterizers.maximin import maximin_clustering
from modules.clusterizers.kmeans import kmeans_clustering

# ==================== Параметры ==================== #
N = 50
count_of_vectors = 5
B_1 = const.B_4
M_1, M_2, M_3, M_4, M_5 = const.M_1, const.M_2, const.M_3, const.M_4, const.M_5
xs = np.linspace(-3, 3, N)
save_load_path = "data\\7"


def get_vectors(
        num_of_vectors: int, Bs: list, Ms: list,
        generate: bool, save: bool, save_load_path: str):
    if len(Bs) != num_of_vectors or len(Ms) != num_of_vectors:
        raise ValueError(
            'Количество корреляционных матриц и векторов м.о. не совпадает с числом генерируемых векторов')
    vectors = []
    if generate:
        vectors = [generate_norm_vector(N, Bs[i], Ms[i])
                   for i in range(num_of_vectors)]
        if save:
            [np.savetxt(f"{save_load_path}/X_{i + 1}.txt", vectors[i])
             for i in range(num_of_vectors)]
    else:
        vectors = [np.loadtxt(f"{save_load_path}/X_{i + 1}.txt")
                   for i in range(num_of_vectors)]
    return vectors


def painter(title: str, xs: list):
    plt.figure()
    plt.title(title)
    # "green", "red", "magenta", "pink", "black", "yellow", "cyan"
    cloud_colors = ["blue", "orange", "green", "red", "yellow"]
    for i in range(len(xs)):
        plt.plot(xs[i][0], xs[i][1], c=cloud_colors[i],
                 marker='.', linestyle='none')
    # mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')
    plt.show()


def clusters_painter(classes_vectors: dict, centers: np.ndarray, lonely: bool = True, title: str = ""):
    if lonely:
        plt.figure()
    plt.title(title)
    cloud_colors = ["blue", "orange", "green",
                    "magenta", "yellow", "cyan", "black", "pink"]
    i = 0
    for _class in classes_vectors:
        plt.plot(classes_vectors[_class][0], classes_vectors[_class][1],
                 c=cloud_colors[i], marker='.', linestyle='none', label=f"{_class} class")
        i += 1
    plt.plot(centers[0, :], centers[1, :], c="red", marker='.',
             linestyle='none', label=f"centers")
    plt.legend()
    if lonely:
        plt.show()


def dependency_graph_painter(
        title: str,
        iters_count: int,
        name_ys: dict,
        lonely: bool = True):
    if lonely:
        plt.figure()
    plt.title(title)
    for name in name_ys:
        plt.plot(np.arange(3, iters_count + 1),
                 name_ys[name], label=name)
    plt.legend()
    if lonely:
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.show()


if __name__ == "__main__":
    config = {
        "generate": False,
        "save": False,
        "checkpoints": [1, 2]
    }

    vectors = get_vectors(
        count_of_vectors, [B_1, B_1, B_1, B_1, B_1], [M_1, M_2, M_3, M_4, M_5],
        config["generate"], config["save"], save_load_path
    )
    painter("Облачка", vectors)

    sample = np.concatenate(vectors, axis=1)

    # ==================== Кластеризация sklearn - KMeans (просто проверка) ==================== #
    if 0 in config["checkpoints"]:
        count_of_clusters = 5
        kmeans = KMeans(n_clusters=count_of_clusters)
        y_predicted = kmeans.fit_predict(sample.T)
        classes_vectors = sample_2_clustered_sample(
            sample, kmeans.labels_, count_of_clusters)
        centers = kmeans.cluster_centers_.T
        clusters_painter(classes_vectors, centers)

    # ==================== Максиминное расстояние ==================== #
    """
    Разработать программу кластеризации данных с использованием максиминного
    алгоритма. В качестве типичного расстояния взять половину среднего расстояния
    между существующими кластерами. Построить отображение результатов кластеризации 
    для числа кластеров, начиная с двух. Построить график зависимости максимального 
    (из минимальных) и типичного расстояний от числа кластеров.
    """
    if 1 in config["checkpoints"]:
        clustering_result, dependency_graph = maximin_clustering(sample)
        for iter in clustering_result:
            classes_vectors = sample_2_clustered_sample(
                sample, clustering_result[iter]["labels"], clustering_result[iter]["centers"].shape[1])
            clusters_painter(
                classes_vectors, clustering_result[iter]["centers"])
        title = "Зависимость типичного и максиминного расстояния от числа кластеров"
        dependency_graph_painter(
            title,
            dependency_graph["iters_count"],
            {
                "максиминное расстояние": dependency_graph["maximin_distances"],
                "типичное расстояние": dependency_graph["typical_distances"]
            }
        )
        fig = plt.figure()
        i = 1
        for iter in clustering_result:
            fig.add_subplot(2, 3, i)
            classes_vectors = sample_2_clustered_sample(
                sample, clustering_result[iter]["labels"], clustering_result[iter]["centers"].shape[1])
            clusters_painter(
                classes_vectors, clustering_result[iter]["centers"], lonely=False)
            i += 1
        fig.add_subplot(2, 3, (5, 6))
        dependency_graph_painter(
            title,
            dependency_graph["iters_count"],
            {
                "максиминное расстояние": dependency_graph["maximin_distances"],
                "типичное расстояние": dependency_graph["typical_distances"]
            },
            lonely=False
        )
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.show()

    # ==================== K внутригрупповых средних ==================== #
    """
    Разработать программу кластеризации данных с использованием алгоритма K 
    внутригрупповых средних для числа кластеров равного 3 и 5. Для ситуации 
    5 кластеров подобрать начальные условия так, чтобы получить два результата: 
    а) чтобы кластеризация максимально соответствовала первоначальному разбиению 
    на классы («правильная» кластеризация); б) чтобы кластеризация максимально не 
    соответствовала первоначальному разбиению на классы («неправильная» кластеризация). 
    Для всех случаев построить графики зависимости числа векторов признаков, 
    сменивших номер кластера, от номера итерации алгоритма.
    """
    if 2 in config["checkpoints"]:
        title = "Зависимость числа векторов признаков, сменивших номер класса от номера итерации"
        count_of_clusters = 3
        clustering_result, dependency_graph = kmeans_clustering(
            sample, count_of_clusters)
        classes_vectors = sample_2_clustered_sample(
            sample, clustering_result["labels"], count_of_clusters)
        clusters_painter(
            classes_vectors, clustering_result["centers"], title="KMeans. Число кластеров - 3")
        dependency_graph_painter(
            title,
            dependency_graph["iters_count"],
            {"число векторов сменивших класс":
                dependency_graph["count_of_changed_class_vectors"]}
        )

        count_of_clusters = 5
        clustering_result, dependency_graph = kmeans_clustering(
            sample, count_of_clusters)
        classes_vectors = sample_2_clustered_sample(
            sample, clustering_result["labels"], count_of_clusters)
        clusters_painter(
            classes_vectors, clustering_result["centers"], title="KMeans. Число кластеров - 5.\n \"Правильная\" кластеризация")
        dependency_graph_painter(
            title,
            dependency_graph["iters_count"],
            {"число векторов сменивших класс":
                dependency_graph["count_of_changed_class_vectors"]}
        )

        count_of_clusters = 5
        clustering_result, dependency_graph = kmeans_clustering(
            sample, count_of_clusters, right_clusterization=False)
        classes_vectors = sample_2_clustered_sample(
            sample, clustering_result["labels"], count_of_clusters)
        clusters_painter(
            classes_vectors, clustering_result["centers"], title="KMeans. Число кластеров - 5.\n \"Неправильная\" кластеризация"
        )
        dependency_graph_painter(
            title,
            dependency_graph["iters_count"],
            {"число векторов сменивших класс":
                dependency_graph["count_of_changed_class_vectors"]}
        )
