import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class FeatureEDA:
    def __init__(self, feature_path, save_dir_abs):
        """
        初始化EDA分析器
        :param feature_path: 特征CSV文件路径
        :param save_dir_abs: 可视化结果保存目录
        """
        self.df = pd.read_csv(feature_path, encoding="utf-8-sig")
        self.feature_cols = [col for col in self.df.columns if col.startswith('feature')]
        self.save_dir_abs = save_dir_abs

        # 定义各类型特征的维度范围（按顺序拼接）
        self.feature_types = {
            "颜色特征": 464,
            "HOG特征": 2048,
            "LBP特征": 944,
            "SIFT特征": 2048
        }

        # 计算各类型特征对应的列名
        self.feature_groups = self._get_feature_groups()
        print(f"加载特征数据：{len(self.df)}样本，{len(self.feature_cols)}特征")
        print(f"特征分组：{', '.join([f'{k}({v}维)' for k, v in self.feature_types.items()])}")

    def _get_feature_groups(self):
        """计算各类型特征对应的列名列表"""
        groups = {}
        start = 1  # 特征列从feature1开始

        for name, dim in self.feature_types.items():
            end = start + dim - 1
            # 生成featureX列名列表
            groups[name] = [f"feature{i}" for i in range(start, end + 1)]
            start = end + 1  # 更新下一组起始索引

        return groups

    def basic_statistics(self):
        """基本统计分析：均值、方差、极值"""
        stats = self.df[self.feature_cols].describe().T
        stats.to_csv(os.path.join(self.save_dir_abs, "feature_statistics.csv"), encoding="utf-8-sig")
        print("基本统计结果已保存至feature_statistics.csv")
        return stats

    def summarize_statistics(self):
        """对feature_statistics.csv进行进一步分析，按特征类型汇总统计量"""
        # 读取基本统计结果
        stats_path = os.path.join(self.save_dir_abs, "feature_statistics.csv")
        if not os.path.exists(stats_path):
            print("未找到feature_statistics.csv，先运行basic_statistics生成")
            return

        stats_df = pd.read_csv(stats_path, index_col=0)  # 索引为特征名（featureX）
        summary_data = []

        # 按特征类型分组计算统计量的分布
        for group_name, feature_cols in self.feature_groups.items():
            # 筛选当前特征类型的统计数据
            group_stats = stats_df.loc[feature_cols]

            # 计算该组特征统计量的描述性统计（如：所有特征的均值的均值、最大值等）
            summary = {
                "特征类型": group_name,
                "特征数量": len(group_stats),
                # 均值的统计
                "均值_平均值": group_stats['mean'].mean(),
                "均值_最大值": group_stats['mean'].max(),
                "均值_最小值": group_stats['mean'].min(),
                "均值_标准差": group_stats['mean'].std(),
                # 标准差的统计
                "标准差_平均值": group_stats['std'].mean(),
                "标准差_最大值": group_stats['std'].max(),
                "标准差_最小值": group_stats['std'].min(),
                # 四分位距的统计（反映离散程度）
                "四分位距_平均值": (group_stats['75%'] - group_stats['25%']).mean()
            }
            summary_data.append(summary)

        # 保存汇总结果
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(self.save_dir_abs, "feature_statistics_summary.csv")
        summary_df.to_csv(summary_path, encoding="utf-8-sig", index=False)
        print(f"特征统计汇总结果已保存至{summary_path}")

    def distribution_analysis(self, n_features=5):
        """为每种特征类型分别绘制分布直方图（每种类型随机选择n个特征）"""
        for group_name, cols in self.feature_groups.items():
            # 从当前特征组中随机选择n个特征
            sample_cols = np.random.choice(cols, min(n_features, len(cols)), replace=False)

            plt.figure(figsize=(15, 10))
            for i, col in enumerate(sample_cols):
                plt.subplot(2, 3, i + 1)
                sns.histplot(self.df[col], kde=True)
                plt.title(f"{group_name}：{col}分布")

            plt.tight_layout()
            save_path = os.path.join(self.save_dir_abs, f"{group_name}_distribution.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"{group_name}分布直方图已保存至{save_path}")

    def correlation_analysis(self, n_features=10):
        """为每种特征类型分别计算相关性并绘制热力图（每种类型随机选择n个特征）"""
        for group_name, cols in self.feature_groups.items():
            # 从当前特征组中随机选择n个特征（确保不超过实际数量）
            sample_size = min(n_features, len(cols))

            sample_cols = np.random.choice(cols, sample_size, replace=False)
            corr = self.df[sample_cols].corr()

            plt.figure(figsize=(12, 10))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
            plt.title(f"{group_name}相关性热力图（{sample_size}个特征）")
            plt.tight_layout()

            save_path = os.path.join(self.save_dir_abs, f"{group_name}_correlation.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"{group_name}相关性热力图已保存至{save_path}")

    def pca_visualization(self):
        """PCA降维可视化（2D）"""
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(self.df[self.feature_cols])
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=features_pca[:, 0], y=features_pca[:, 1], hue=self.df['product_category'], palette='tab10')
        plt.title(f"PCA降维可视化（解释方差：{pca.explained_variance_ratio_.sum():.2f}）")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir_abs, "pca_visualization.png"), dpi=300)
        plt.close()
        print("PCA可视化结果已保存")

    def tsne_visualization(self):
        """t-SNE降维可视化（2D）"""
        tsne = TSNE(n_components=2, random_state=42)
        # 采样200样本加速计算
        sample_df = self.df.sample(200, random_state=42)
        features_tsne = tsne.fit_transform(sample_df[self.feature_cols])

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=features_tsne[:, 0], y=features_tsne[:, 1],
                        hue=sample_df['product_category'], palette='tab10')
        plt.title("t-SNE降维可视化")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir_abs, "tsne_visualization.png"), dpi=300)
        plt.close()
        print("t-SNE可视化结果已保存")

    def run_all(self):
        """运行所有EDA分析"""
        self.basic_statistics()
        self.summarize_statistics()
        self.distribution_analysis()
        self.correlation_analysis()
        self.pca_visualization()
        self.tsne_visualization()
        print("所有EDA分析完成")


if __name__ == "__main__":
    RESULT_FILE_PATH = "Tradition_results"
    MODE = None

    prev_path = f"D:/杨谊瑶/大三上/大数据技术/第三次上机/img_feature_clustering/files/{RESULT_FILE_PATH}"

    feature_path = None
    if RESULT_FILE_PATH == "DL_results":
        if MODE == "augment":
            feature_path = prev_path + "/deep_features_augmented.csv"
        elif MODE == "simple":
            feature_path = prev_path + "/deep_features_simple.csv"
        elif MODE == "mix":
            feature_path = prev_path + "/hybrid_features.csv"
    elif RESULT_FILE_PATH == "Tradition_results" and MODE is None:
        feature_path = prev_path + "/tradition_feature.csv"

    save_dir = "eda_stat_plots"
    save_dir_abs = prev_path + "/" + save_dir
    os.makedirs(save_dir_abs, exist_ok=True)
    eda = FeatureEDA(feature_path, save_dir_abs=save_dir_abs)
    eda.run_all()