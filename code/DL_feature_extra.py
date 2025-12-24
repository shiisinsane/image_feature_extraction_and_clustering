import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import join
import warnings

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


class DeepFeatureExtractor:
    """
    深度学习特征提取
    基于ResNet50
    """

    def __init__(self):
        # 加载预训练的ResNet50模型
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # 移除最后的全连接层，使用全局平均池化后的特征
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # 移除最后的全连接层

        # 将模型设置为评估模式
        self.model.eval()
        self.model.to(device)

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet标准输入尺寸
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet均值
                std=[0.229, 0.224, 0.225]  # ImageNet标准差
            )
        ])

        print(f"ResNet50模型加载完成，特征维度: 2048维")

    def extract_features(self, image_path):
        """
        提取单张图像的深度学习特征
        :param image_path: 图像路径
        :return: 2048维特征向量
        """
        # 加载图像
        if image_path.lower().endswith(".gif"): # 处理GIF格式
            img = Image.open(image_path)
            img = img.convert("RGB")
        else:
            img = Image.open(image_path).convert("RGB")

        # 预处理
        img_tensor = self.transform(img).unsqueeze(0).to(device)

        # 提取特征，不计算梯度
        with torch.no_grad():
            features = self.model(img_tensor)

        # 展平特征向量并转换为numpy数组
        features = features.squeeze().cpu().numpy()

        # L2归一化，提高特征表示能力
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        return features


    def extract_features_with_augmentation(self, image_path, num_augmentations=3):
        """
        提取增强后的图像特征：多视图增强，提高特征鲁棒性
        :param image_path: 图像路径
        :param num_augmentations: 增强次数
        :return: 平均后的特征向量
        """
        # 数据增强转换
        augment_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # 加载原始图像
        if image_path.lower().endswith(".gif"):
            img = Image.open(image_path)
            img = img.convert("RGB")
        else:
            img = Image.open(image_path).convert("RGB")

        all_features = []

        # 提取多次增强后的特征
        with torch.no_grad():
            for i in range(num_augmentations):
                if i == 0:
                    # 第一次使用原始图像
                    img_tensor = self.transform(img).unsqueeze(0).to(device)
                else:
                    # 后续使用增强图像
                    img_aug = augment_transforms(img)
                    img_tensor = img_aug.unsqueeze(0).to(device)

                features = self.model(img_tensor)
                features = features.squeeze().cpu().numpy()

                # L2归一化
                norm = np.linalg.norm(features)
                if norm > 0:
                    features = features / norm

                all_features.append(features)

        # 计算平均特征
        avg_features = np.mean(all_features, axis=0)

        # 再次归一化
        norm = np.linalg.norm(avg_features)
        if norm > 0:
            avg_features = avg_features / norm

        return avg_features


class HybridFeatureExtractor:
    """
    混合特征提取器：深度学习特征+传统特征
    """

    def __init__(self, use_augmentation=False):
        """
        初始化混合特征提取器
        :param use_augmentation: 是否使用数据增强
        """
        self.deep_extractor = DeepFeatureExtractor()
        self.use_augmentation = use_augmentation

        # 导入传统特征提取方法
        from feature_extra import (
            preprocess_image,
            extract_color_histogram,
            extract_hog_feature,
            extract_lbp_feature,
            extract_sift_feature
        )
        self.preprocess_image = preprocess_image
        self.extract_color_histogram = extract_color_histogram
        self.extract_hog_feature = extract_hog_feature
        self.extract_lbp_feature = extract_lbp_feature
        self.extract_sift_feature = extract_sift_feature

    def extract_traditional_features(self, image_path):
        """
        提取传统特征，复用feature_extra.py
        :param image_path: 图像路径
        :return: 整合后的传统特征向量，与feature_extra.extract_all_features保持一致
        """

        # 图像预处理：统一尺寸、去噪
        img = self.preprocess_image(image_path)

        # 提取各类传统特征
        color_hist = self.extract_color_histogram(img)  # 116维
        hog = self.extract_hog_feature(img)  # 512维
        lbp = self.extract_lbp_feature(img)  # 944维
        sift = self.extract_sift_feature(img)  # 2048维

        # 权重分配，与feature_extra保持一致的
        color_w = 1.0
        hog_w = 0.7
        lbp_w = 0.6
        sift_w = 1.0

        # 整合所有传统特征
        traditional_features = np.hstack([
            color_w * color_hist,
            hog_w * hog,
            lbp_w * lbp,
            sift_w * sift
        ])

        # 归一化传统特征
        norm = np.linalg.norm(traditional_features)
        if norm > 0:
            traditional_features = traditional_features / norm

        return traditional_features



    def extract_hybrid_features(self, image_path):
        """
        提取混合特征：深度学习特征 + 传统特征
        :param image_path: 图像路径
        :return: 混合特征向量
        """
        # 提取深度学习特征
        if self.use_augmentation:
            deep_features = self.deep_extractor.extract_features_with_augmentation(image_path)
        else:
            deep_features = self.deep_extractor.extract_features(image_path)

        # 提取传统特征
        traditional_features = self.extract_traditional_features(image_path)

        # 权重分配
        deep_weight = 0.7  # 深度学习权重
        traditional_weight = 0.3  # 传统权重

        # 加权拼接
        hybrid_features = np.hstack([
            deep_weight * deep_features,
            traditional_weight * traditional_features
        ])

        # 归一化
        norm = np.linalg.norm(hybrid_features)
        if norm > 0:
            hybrid_features = hybrid_features / norm

        return hybrid_features


def batch_extract_deep_features(pic_folder, use_hybrid=False, use_augmentation=False):
    """
    批量提取深度学习特征
    :param pic_folder: 图片文件夹路径
    :param use_hybrid: 是否使用混合特征
    :param use_augmentation: 是否使用数据增强
    :return: result_df, feature_matrix
    """
    # 读取产品信息
    product_df = pd.read_excel("data_sample_400.xlsx")
    product_ids = product_df["product_id"].values
    product_brands = product_df["brand"].values
    product_categories = product_df["product_category"].values

    # 构建图片索引
    img_index = {}
    for img_name in listdir(pic_folder):
        for pid in product_ids:
            if str(pid) in img_name:
                img_index[pid] = join(pic_folder, img_name)
                break

    # 初始化特征提取器
    if use_hybrid:
        extractor = HybridFeatureExtractor(use_augmentation=use_augmentation)
        print("使用混合特征提取")
    else:
        extractor = DeepFeatureExtractor()
        print("使用纯深度学习特征提取")

    features_list = []
    result_data = []

    # 批量提取特征
    for idx, product_id in enumerate(product_ids):
        if product_id not in img_index:
            print(f"未找到产品ID{product_id}对应的图片")
            continue

        img_path = img_index[product_id]
        img_name = img_path.split("/")[-1]

        print(f"正在提取第{idx + 1}张图片特征：{img_name}")

        # 提取特征
        if use_hybrid:
            features = extractor.extract_hybrid_features(img_path)
        elif hasattr(extractor, 'extract_features_with_augmentation') and use_augmentation:
            features = extractor.extract_features_with_augmentation(img_path)
        else:
            features = extractor.extract_features(img_path)

        features_list.append(features)

        # 构建特征字典
        feature_dict = {
            "product_id": product_id,
            "product_category": product_categories[idx],
            "brand": product_brands[idx]
        }

        # 添加特征列
        for i in range(len(features)):
            feature_dict[f"feature{i + 1}"] = features[i]

        result_data.append(feature_dict)

    # 创建DataFrame和特征矩阵
    result_df = pd.DataFrame(result_data)
    feature_matrix = np.array(features_list)

    print(f"特征提取完成!")
    print(f"处理图片数量: {feature_matrix.shape[0]}张")
    print(f"特征维度: {feature_matrix.shape[1]}维")

    return result_df, feature_matrix


def save_deep_features(result_df, output_path=None):
    """
    保存深度学习特征到CSV
    :param result_df: 包含特征的DataFrame
    :param output_path: 输出路径
    """
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


if __name__ == "__main__":
    PIC_FOLDER = "pic"

    os.makedirs("D:/杨谊瑶/大三上/大数据技术/第三次上机/img_feature_clustering/files/DL_results", exist_ok=True)

    # 选项1: 仅使用深度学习特征
    print("=== 1: 仅使用深度学习特征 ===")
    result_df_deep, feature_matrix_deep = batch_extract_deep_features(
        PIC_FOLDER,
        use_hybrid=False,
        use_augmentation=False
    )

    if result_df_deep is not None:
        save_deep_features(result_df_deep, "D:/杨谊瑶/大三上/大数据技术/第三次上机/img_feature_clustering/files/DL_results/deep_features_simple.csv")

    # 选项2: 使用增强的深度学习特征
    print("\n=== 2: 使用增强的深度学习特征 ===")
    result_df_aug, feature_matrix_aug = batch_extract_deep_features(
        PIC_FOLDER,
        use_hybrid=False,
        use_augmentation=True
    )

    if result_df_aug is not None:
        save_deep_features(result_df_aug, "D:/杨谊瑶/大三上/大数据技术/第三次上机/img_feature_clustering/files/DL_results/deep_features_augmented.csv")

    # 选项3: 使用混合特征
    print("\n=== 3: 使用混合特征 ===")
    result_df_hybrid, feature_matrix_hybrid = batch_extract_deep_features(
        PIC_FOLDER,
        use_hybrid=True,
        use_augmentation=False
    )

    if result_df_hybrid is not None:
        save_deep_features(result_df_hybrid, "D:/杨谊瑶/大三上/大数据技术/第三次上机/img_feature_clustering/files/DL_results/hybrid_features.csv")