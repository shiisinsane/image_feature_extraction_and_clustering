import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score, confusion_matrix
)
import warnings

warnings.filterwarnings("ignore")
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ---------------------- 聚类与原始标签一致性评估 ----------------------
def evaluate_cluster_vs_original(result_df, results_file_path, cluster_col="clustering", label_col="product_category"):
    """
    评估聚类结果与原始类别标签的一致性
    :param result_df: 包含聚类结果和原始标签的DataFrame
    :param cluster_col: 聚类结果列名
    :param label_col: 原始类别标签列名
    :return: 评估指标字典、混淆矩阵
    """
    print("\n=== 聚类结果 vs 原始类别标签 一致性评估 ===")

    # 1. 提取聚类标签和原始标签
    cluster_labels = result_df[cluster_col].values
    original_labels = result_df[label_col].values

    # 2. 基础统计
    print(f"聚类簇数量: {len(np.unique(cluster_labels))}")
    print(f"原始类别分布:")
    original_counts = pd.Series(original_labels).value_counts().sort_index()
    for label, count in original_counts.items():
        print(f"  类别{label}: {count}个样本")

    # 3. 计算核心评估指标
    metrics = {
        "调整兰德指数(ARI)": adjusted_rand_score(original_labels, cluster_labels),
        "归一化互信息(NMI)": normalized_mutual_info_score(original_labels, cluster_labels),
        "同质性(Homogeneity)": homogeneity_score(original_labels, cluster_labels),
        "完整性(Completeness)": completeness_score(original_labels, cluster_labels),
        "V-measure": v_measure_score(original_labels, cluster_labels)
    }

    # 4. 输出指标（越接近1表示一致性越高）
    print("\n核心评估指标（越接近1越好）:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    # 5. 计算混淆矩阵
    cm = confusion_matrix(original_labels, cluster_labels)
    # 标准化混淆矩阵（按原始类别归一化，便于对比）
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 6. 可视化混淆矩阵
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 原始混淆矩阵
    im1 = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.set_title('混淆矩阵（原始样本数）', fontsize=12)
    ax1.set_xlabel('聚类簇标签', fontsize=10)
    ax1.set_ylabel('原始类别标签', fontsize=10)
    # 添加数值标注
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    # 设置刻度
    ax1.set_xticks(np.arange(len(np.unique(cluster_labels))))
    ax1.set_yticks(np.arange(len(np.unique(original_labels))))
    ax1.set_xticklabels([f"簇{int(x)}" for x in np.unique(cluster_labels)])
    ax1.set_yticklabels([f"类别{int(x)}" for x in np.unique(original_labels)])

    # 归一化混淆矩阵
    im2 = ax2.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    ax2.set_title('混淆矩阵（归一化，按原始类别）', fontsize=12)
    ax2.set_xlabel('聚类簇标签', fontsize=10)
    ax2.set_ylabel('原始类别标签', fontsize=10)
    # 添加数值标注
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            ax2.text(j, i, f"{cm_normalized[i, j]:.2f}", ha="center", va="center",
                     color="white" if cm_normalized[i, j] > 0.5 else "black")
    # 设置刻度
    ax2.set_xticks(np.arange(len(np.unique(cluster_labels))))
    ax2.set_yticks(np.arange(len(np.unique(original_labels))))
    ax2.set_xticklabels([f"簇{int(x)}" for x in np.unique(cluster_labels)])
    ax2.set_yticklabels([f"类别{int(x)}" for x in np.unique(original_labels)])

    # 添加颜色条
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    plt.tight_layout()
    save_path = f'D:/杨谊瑶/大三上/大数据技术/第三次上机/img_feature_clustering/files/{results_file_path}/confusion_matrix.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # 7. 详细分析每个原始类别的聚类分布
    print("\n=== 每个原始类别的聚类分布详情 ===")
    for original_label in np.unique(original_labels):
        mask = result_df[label_col] == original_label
        cluster_dist = result_df[mask][cluster_col].value_counts().sort_index()
        print(f"原始类别{original_label}（{mask.sum()}个样本）的聚类分布:")
        for cluster_id, count in cluster_dist.items():
            ratio = count / mask.sum() * 100
            print(f"  - 聚类簇{cluster_id}: {count}个样本 ({ratio:.1f}%)")

    return metrics, cm, cm_normalized


# ---------------------- 加载特征数据 ----------------------
def load_feature_data(results_file_path, mode):
    """加载特征提取结果，分离特征和产品信息"""
    # 读取特征文件

    if results_file_path == "DL_results":
        if mode == "simple":
            # 使用神经网络进行图像特征提取
            feature_df = pd.read_csv(f"D:/杨谊瑶/大三上/大数据技术/第三次上机/img_feature_clustering/files/"
                                     f"{results_file_path}/deep_features_simple.csv",
                                     encoding="utf-8-sig")
        elif mode == "augment":
            # 使用增强的神经网络进行图像特征提取
            feature_df = pd.read_csv(f"D:/杨谊瑶/大三上/大数据技术/第三次上机/img_feature_clustering/files/"
                                     f"{results_file_path}/deep_features_augmented.csv",
                                     encoding="utf-8-sig")
        elif mode == "mix":
            # 使用混合方法进行图像特征提取
            feature_df = pd.read_csv(f"D:/杨谊瑶/大三上/大数据技术/第三次上机/img_feature_clustering/files/"
                                     f"{results_file_path}/hybrid_features.csv",
                                     encoding="utf-8-sig")

    elif results_file_path == "Tradition_results" and mode is None:
        # 传统的图片特征提取方法
        feature_df = pd.read_csv(f"D:/杨谊瑶/大三上/大数据技术/第三次上机/img_feature_clustering/files/"
                                 f"{results_file_path}/tradition_feature.csv",
                                 encoding="utf-8-sig")


    feature_df['brand'] = (feature_df['brand'].astype(int) - 100)
    # 分离产品基础信息（product_id, product_category, brand）和特征
    product_info = feature_df[["product_id", "product_category", "brand"]].copy()
    # 提取特征列
    feature_cols = [col for col in feature_df.columns if col.startswith('feature')]
    feature_matrix = feature_df[feature_cols].values

    return product_info, feature_matrix, feature_cols, feature_df


# ---------------------- 特征预处理：标准化+降维 ----------------------
def preprocess_features(feature_matrix):
    """特征标准化+ PCA降维（减少维度，提升聚类效果）"""

    # 标准化
    scaler = StandardScaler()
    feature_scaled = scaler.fit_transform(feature_matrix)
    print(f"标准化后范围: [{feature_scaled.min():.6f}, {feature_scaled.max():.6f}]")

    # PCA降维
    pca = PCA(n_components=0.97, random_state=42)
    feature_pca = pca.fit_transform(feature_scaled)

    print(f"PCA降维前维度：{feature_scaled.shape[1]}维")
    print(f"PCA降维后维度：{feature_pca.shape[1]}维")
    print(f"解释方差比例: {np.sum(pca.explained_variance_ratio_):.4f}")

    return feature_pca, scaler, pca


# ---------------------- K-Means聚类 ----------------------
def kmeans_clustering(feature_pca, optimal_k):
    """执行K-Means聚类，返回聚类标签和模型"""
    print("\n=== 执行K-Means聚类 ===")

    # 初始化K-Means
    kmeans = KMeans(
        n_clusters=optimal_k,
        random_state=42,
        n_init=30,  # 多次初始化取最优
        max_iter=1000,
        tol=1e-4
    )
    cluster_labels = kmeans.fit_predict(feature_pca)

    # 计算轮廓系数
    silhouette = silhouette_score(feature_pca, cluster_labels)

    # 输出聚类结果统计
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    print(f"最终聚类数K：{optimal_k}")
    print(f"轮廓系数：{silhouette:.4f}")
    print(f"各簇样本数量：")
    for cluster_id, count in cluster_counts.items():
        print(f"  簇{cluster_id + 1}: {count}个样本")

    return cluster_labels, kmeans, silhouette


# ---------------------- 生成聚类结果文件 ----------------------
def generate_cluster_result(product_info, feature_matrix, feature_cols, cluster_labels, results_file_path):
    """整合产品信息、特征、聚类结果，生成cluster_result.csv"""
    # 整合产品信息和聚类结果
    result_df = product_info.copy()
    # 聚类标签从1开始计数
    result_df["clustering"] = cluster_labels + 1

    # 添加所有特征列
    for i, col in enumerate(feature_cols):
        result_df[col] = feature_matrix[:, i]

    # 保存结果
    save_path = f"D:/杨谊瑶/大三上/大数据技术/第三次上机/img_feature_clustering/files/{results_file_path}/cluster_result.csv"
    result_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\n聚类结果已保存到: {save_path}")
    return result_df


# ---------------------- 聚类结果分析 ----------------------
def analyze_cluster_result(result_df, result_file_path):
    """分析K-Means聚类结果：品牌分布、类别分布"""
    print("\n=== K-Means聚类结果分析 ===")

    # 每个簇的品牌分布
    brand_cluster = pd.crosstab(result_df["clustering"], result_df["brand"])
    print("\n各簇品牌分布：")
    print(brand_cluster)

    # 每个簇的产品类别分布
    category_cluster = pd.crosstab(result_df["clustering"], result_df["product_category"])
    print("\n各簇产品类别分布：")
    print(category_cluster)

    # 保存分析结果
    brand_path = f"D:/杨谊瑶/大三上/大数据技术/第三次上机/img_feature_clustering/files/{result_file_path}/brand_cluster_distribution.csv"
    category_path = f"D:/杨谊瑶/大三上/大数据技术/第三次上机/img_feature_clustering/files/{result_file_path}/category_cluster_distribution.csv"
    brand_cluster.to_csv(brand_path, encoding="utf-8-sig")
    category_cluster.to_csv(category_path, encoding="utf-8-sig")

    return brand_cluster, category_cluster


import seaborn as sns


def analyze_cluster_feature_distribution(result_df, feature_cols, feature_groups, n_top_features=5):
    """
    分析每个簇的特征分布，识别标志性特征
    :param result_df: 包含聚类结果和特征的DataFrame
    :param feature_cols: 特征列名列表
    :param feature_groups: 特征分组字典（键：分组名，值：该组包含的特征列名）
    :param n_top_features: 每个簇需要展示的top特征数量
    :return: 各簇的特征均值字典
    """
    print("\n=== 各簇特征分布分析 ===")
    cluster_ids = sorted(result_df["clustering"].unique())
    cluster_feature_means = {}  # 存储每个簇的特征均值

    # 1. 计算每个簇的特征均值
    for cluster_id in cluster_ids:
        cluster_data = result_df[result_df["clustering"] == cluster_id]
        # 提取特征列并计算均值
        feature_means = cluster_data[feature_cols].mean().to_dict()
        cluster_feature_means[cluster_id] = feature_means
        print(f"\n--- 簇{cluster_id}的特征均值（前{n_top_features}个关键特征） ---")

        # 按均值排序，取top特征（可根据特征分组筛选）
        sorted_features = sorted(feature_means.items(), key=lambda x: x[1], reverse=True)
        for col, mean in sorted_features[:n_top_features]:
            # 找到特征所属分组
            group_name = [k for k, v in feature_groups.items() if col in v][0]
            print(f"  {group_name} - {col}: {mean:.4f}")

    # 2. 可视化关键特征组的簇间差异（以颜色、纹理、形状为例）
    for group_name, group_cols in feature_groups.items():
        if len(group_cols) == 0:
            continue
        # 计算每个簇在该组特征上的平均均值（简化高维特征）
        group_means = []
        for cluster_id in cluster_ids:
            # 取该组所有特征的均值作为组代表值
            group_mean = np.mean([cluster_feature_means[cluster_id][col] for col in group_cols])
            group_means.append(group_mean)

        # 绘制条形图
        plt.figure(figsize=(10, 6))
        sns.barplot(x=cluster_ids, y=group_means)
        plt.title(f"各簇在[{group_name}]特征上的平均表现", fontsize=12)
        plt.xlabel("聚类簇ID")
        plt.ylabel(f"{group_name}特征均值（越高越显著）")
        plt.xticks(cluster_ids)
        plt.tight_layout()
        save_path = f'D:/杨谊瑶/大三上/大数据技术/第三次上机/img_feature_clustering/files/{RESULTS_FILE_PATH}/{group_name}_cluster_comparison.png'
        plt.savefig(save_path, dpi=300)
        plt.show()

    return cluster_feature_means



if __name__ == "__main__":
    # 传统方法
    RESULTS_FILE_PATH = "Tradition_results"
    MODE = None

    # # 深度学习方法
    # RESULTS_FILE_PATH = "DL_results"
    # MODE = "augment"

    # 加载数据
    product_info, feature_matrix, feature_cols, feature_df = load_feature_data(RESULTS_FILE_PATH, MODE)

    # 特征预处理（标准化+PCA降维）
    feature_pca, scaler, pca = preprocess_features(feature_matrix)

    # K-Means聚类数
    optimal_k = 13

    # 执行K-Means聚类
    cluster_labels, kmeans_model, silhouette = kmeans_clustering(feature_pca, optimal_k)

    # 生成聚类结果
    result_df = generate_cluster_result(product_info, feature_matrix, feature_cols, cluster_labels, RESULTS_FILE_PATH)

    # 分析聚类结果（品牌/类别分布）
    brand_cluster, category_cluster = analyze_cluster_result(result_df, RESULTS_FILE_PATH)

    # 评估聚类与原始product_category标签的一致性
    eval_metrics, cm, cm_normalized = evaluate_cluster_vs_original(
        result_df,
        RESULTS_FILE_PATH,
        cluster_col="clustering",
        label_col="brand",
        #label_col="product_category", # 启用这个评价标准时，要同时修改K-Means聚类数为7
    )

    # 传统特征分组：特征描述，不同类别的产品具有怎样的特征
    if RESULTS_FILE_PATH == "Tradition_results":
        feature_groups = {
            "颜色特征（HSV直方图+颜色矩）": feature_cols[:464],  # 对应extract_color_histogram的464维
            "形状特征（HOG）": feature_cols[464:464 + 2048],  # 对应extract_hog_feature的2048维
            "纹理特征（LBP）": feature_cols[464 + 2048:464 + 2048 + 944],  # 对应extract_lbp_feature的944维
            "局部特征（SIFT）": feature_cols[464 + 2048 + 944:464 + 2048 + 944 + 2048],  # 对应extract_sift_feature的2048维
        }
        # 分析各簇的特征分布
        cluster_feature_means = analyze_cluster_feature_distribution(
            result_df,
            feature_cols,
            feature_groups,
            n_top_features=5  # 每个簇展示top5显著特征
        )


    # 聚类分析报告
    report_path = f"D:/杨谊瑶/大三上/大数据技术/第三次上机/img_feature_clustering/files/{RESULTS_FILE_PATH}/kmeans_clustering_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== K-Means 图像特征聚类分析报告 ===\n\n")
        f.write(f"1. 数据集信息:\n")
        f.write(f"   - 样本数量: {len(result_df)}\n")
        f.write(f"   - 原始特征维度: {feature_matrix.shape[1]}维\n")
        f.write(f"   - PCA降维后维度: {feature_pca.shape[1]}维\n")
        f.write(f"   - PCA解释方差比例: {np.sum(pca.explained_variance_ratio_):.4f}\n\n")
        f.write(f"2. K-Means聚类参数:\n")
        f.write(f"   - 最终聚类数K: {optimal_k}\n")
        f.write(f"   - 轮廓系数: {silhouette:.4f}\n")
        f.write(f"   - 初始化次数n_init: {kmeans_model.n_init}\n")
        f.write(f"   - 最大迭代次数max_iter: {kmeans_model.max_iter}\n\n")
        f.write(f"3. 各簇样本分布:\n")
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            f.write(f"   - 簇{cluster_id + 1}: {count}个样本 ({count / len(result_df) * 100:.1f}%)\n")
        f.write(f"\n4. 聚类与原始类别标签一致性评估:\n")
        for metric_name, value in eval_metrics.items():
            f.write(f"   - {metric_name}: {value:.4f}\n")
        f.write(f"\n5. 聚类结果文件说明:\n")
        f.write(f"   - 聚类结果: cluster_result.csv\n")
        f.write(f"   - 品牌分布: brand_cluster_distribution.csv\n")
        f.write(f"   - 类别分布: category_cluster_distribution.csv\n")
        f.write(f"   - 混淆矩阵图: confusion_matrix.png\n")