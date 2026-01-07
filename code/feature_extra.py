import cv2
import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import join
from PIL import Image  # 读取GIF图片


# ---------------------- 图片预处理----------------------
def preprocess_image(image_path):
    """图片预处理：
    读取图片（GIF/JPG/PNG）--> 转RGB--> 统一尺寸350x350--> 高斯去噪"""
    # 判断文件是否为GIF格式
    if image_path.lower().endswith(".gif"):
        # 用PIL打开GIF，取第一帧作为有效帧
        with Image.open(image_path) as img_pil:
            # 转RGB
            img_pil = img_pil.convert("RGB")
            # 转为OpenCV格式
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    else:
        # 非GIF格式，正常读取
        image_data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    # 统一尺寸
    img = cv2.resize(img, (350, 350))
    # 高斯去噪
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


# ------------------- 图片特征提取 ----------------------------
def extract_color_histogram(img, block_rows=4, block_cols=4):
    """
    分块提取HSV颜色直方图+颜色矩
    :param img: OpenCV读取的BGR图像（350x350）
    :param block_rows: 纵向分块数
    :param block_cols: 横向分块数
    :return: 分块后的颜色特征（464维）
    """
    # 转HSV空间
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img_hsv.shape[:2]
    block_h = h // block_rows
    block_w = w // block_cols

    all_block_features = []
    for i in range(block_rows):
        for j in range(block_cols):
            # 截取当前块
            y1 = i * block_h
            y2 = (i + 1) * block_h
            x1 = j * block_w
            x2 = (j + 1) * block_w
            block_img = img_hsv[y1:y2, x1:x2]

            # 1. HSV直方图（H:10bin, S:5bin, V:5bin，更适配HSV分布）
            hist_h = cv2.calcHist([block_img], [0], None, [10], [0, 180])  # H范围0-180
            hist_s = cv2.calcHist([block_img], [1], None, [5], [0, 256])
            hist_v = cv2.calcHist([block_img], [2], None, [5], [0, 256])
            # L1归一化
            hist_h = cv2.normalize(hist_h, hist_h, norm_type=cv2.NORM_L1).flatten()
            hist_s = cv2.normalize(hist_s, hist_s, norm_type=cv2.NORM_L1).flatten()
            hist_v = cv2.normalize(hist_v, hist_v, norm_type=cv2.NORM_L1).flatten()

            # 2. 颜色矩（均值、方差、偏度，补充统计特征）
            h_mean, h_std = np.mean(block_img[:, :, 0]), np.std(block_img[:, :, 0])
            s_mean, s_std = np.mean(block_img[:, :, 1]), np.std(block_img[:, :, 1])
            v_mean, v_std = np.mean(block_img[:, :, 2]), np.std(block_img[:, :, 2])
            # 偏度计算（三阶矩）
            h_skew = np.mean(((block_img[:, :, 0] - h_mean) / h_std) ** 3) if h_std != 0 else 0
            s_skew = np.mean(((block_img[:, :, 1] - s_mean) / s_std) ** 3) if s_std != 0 else 0
            v_skew = np.mean(((block_img[:, :, 2] - v_mean) / v_std) ** 3) if v_std != 0 else 0
            color_moments = np.array([h_mean, h_std, h_skew,
                                      s_mean, s_std, s_skew,
                                      v_mean, v_std, v_skew], dtype=np.float32)

            # 拼接当前块特征（20维直方图 + 9维矩 = 29维）
            block_feature = np.hstack([hist_h, hist_s, hist_v, color_moments])
            all_block_features.append(block_feature)

    # 拼接所有块（4x4块 → 16×29=464维）
    final_feature = np.hstack(all_block_features)
    return final_feature


def extract_hog_feature(img, block_rows=4, block_cols=4):
    """
    优化HOG参数 + L2归一化（适配350x350图片）
    :param img: BGR图像（350x350）
    :param block_rows/cols: 宏观分块数（4×4，每块87x87）
    :return: 分块HOG特征（4×4×128=512维）
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 增强对比度（提升边缘检测效果）
    gray = cv2.equalizeHist(gray)
    h, w = gray.shape
    block_h = h // block_rows
    block_w = w // block_cols

    # 适配87x87块的HOG参数（更紧凑，减少冗余）
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)

    all_block_hog = []
    # 遍历每个宏观分块
    for i in range(block_rows):
        for j in range(block_cols):
            # 截取当前宏观块（87x87）
            y1 = i * block_h
            y2 = (i + 1) * block_h
            x1 = j * block_w
            x2 = (j + 1) * block_w
            block_gray = gray[y1:y2, x1:x2]
            block_gray = cv2.resize(block_gray, (64, 64))

            # 计算HOG + L2归一化
            block_hog = hog.compute(block_gray)
            block_hog = cv2.normalize(block_hog, block_hog, norm_type=cv2.NORM_L2).flatten()
            # 截取核心128维（足够描述边缘，减少冗余）
            block_hog = block_hog[:128]
            all_block_hog.append(block_hog)

    # 拼接所有宏观块的HOG特征（4×4×128=2048维）
    final_hog = np.hstack(all_block_hog)
    return final_hog


def extract_lbp_feature(img, radius=2, neighbors=8, block_num=(4, 4), normalize=True):
    """
    均匀模式LBP
    :param img: 输入BGR图像
    :param radius: LBP邻域半径
    :param neighbors: LBP邻域像素数
    :param block_num: 分块数量
    :param normalize: 是否归一化
    :return: 分块LBP特征（4×4×59=944维）
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    height, width = gray.shape
    block_h, block_w = block_num
    sub_block_h = height // block_h
    sub_block_w = width // block_w

    # 均匀模式LUT
    def is_uniform(code):
        b = bin(code)[2:].zfill(8)
        return sum(b[i] != b[i + 1] for i in range(7)) <= 2

    uniform_lut = np.zeros(256, dtype=np.uint8)
    idx = 0
    for i in range(256):
        if is_uniform(i):
            uniform_lut[i] = idx
            idx += 1
        else:
            uniform_lut[i] = 58

    lbp_feature = []

    # LBP
    for i in range(block_h):
        for j in range(block_w):
            block = gray[
                i * sub_block_h:(i + 1) * sub_block_h,
                j * sub_block_w:(j + 1) * sub_block_w
            ]

            lbp = np.zeros_like(block, dtype=np.uint8)
            center = block

            for k in range(neighbors):
                dy = int(radius * np.sin(2 * np.pi * k / neighbors))
                dx = int(radius * np.cos(2 * np.pi * k / neighbors))

                neighbor = np.roll(block, shift=(-dy, -dx), axis=(0, 1))
                lbp |= ((neighbor > center).astype(np.uint8) << (7 - k))


            # 映射到均匀模式
            lbp = uniform_lut[lbp]

            hist, _ = np.histogram(
                lbp,
                bins=59,
                range=(0, 58),
                density=normalize
            )
            lbp_feature.extend(hist)

    return np.array(lbp_feature, dtype=np.float32)


def extract_sift_feature(img, block_num=(4, 4)):
    """
    启用分块SIFT（保留空间信息，替换原有全局均值版本）
    :param img: OpenCV读取的BGR图像（350x350）
    :param block_num: 分块数量 (行块数, 列块数)
    :return: 分块拼接后的SIFT特征（4×4×128=2048维）
    """
    # 灰度化 + 增强
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    h, w = gray.shape
    block_h, block_w = block_num

    sub_block_h = h // block_h
    sub_block_w = w // block_w
    # if sub_block_h == 0 or sub_block_w == 0:
    #     raise ValueError(f"分块数{block_num}过大，图像尺寸{(h, w)}无法分割！")

    sift = cv2.SIFT_create(contrastThreshold=0.02, edgeThreshold=10)  # 调整SIFT参数，提升关键点检测
    all_block_sift = []

    for i in range(block_h):
        for j in range(block_w):
            y1 = i * sub_block_h
            y2 = min((i + 1) * sub_block_h, h)
            x1 = j * sub_block_w
            x2 = min((j + 1) * sub_block_w, w)
            block_gray = gray[y1:y2, x1:x2]

            # 提取SIFT关键点和描述子
            keypoints, descriptors = sift.detectAndCompute(block_gray, None)

            # 处理无关键点情况
            if descriptors is None or len(descriptors) == 0:
                block_sift = np.zeros(128, dtype=np.float32)
            else:
                # 取均值 + 归一化
                block_sift = np.mean(descriptors, axis=0).astype(np.float32)
                block_sift /= (block_sift.sum() + 1e-7)  # L1
                block_sift = np.sqrt(block_sift)
                block_sift = cv2.normalize(block_sift, block_sift, norm_type=cv2.NORM_L2)

            all_block_sift.append(block_sift)

    # 拼接所有子块特征（4×4×128=2048维）
    final_sift = np.hstack(all_block_sift)
    return final_sift


# ---------------------- 整合所有特征 ----------------------
def extract_all_features(image_path):
    img = preprocess_image(image_path)
    color_hist = extract_color_histogram(img)
    hog = extract_hog_feature(img)
    lbp = extract_lbp_feature(img)
    sift = extract_sift_feature(img)

    # 以特征加权的方式拼接所有特征
    color_w = 1.0
    hog_w = 0.7
    lbp_w = 0.6
    sift_w = 1.0

    all_features = np.hstack([
        color_w * color_hist,
        hog_w * hog,
        lbp_w * lbp,
        sift_w * sift
    ])

    return all_features


# ---------------------- 提取400张图片特征---------------------
def batch_extract_features(pic_folder):
    product_df = pd.read_excel("data_sample_400.xlsx")
    product_ids = product_df["product_id"].values
    product_brands = product_df["brand"].values
    product_categories = product_df["product_category"].values

    # 先构建图片索引
    img_index = {}
    for img_name in listdir(pic_folder):
        for pid in product_ids:
            if str(pid) in img_name:
                img_index[pid] = join(pic_folder, img_name)
                break

    features_list = []
    result_data = []

    for idx, product_id in enumerate(product_ids):
        if product_id not in img_index:
            continue

        img_path = img_index[product_id]
        img_name = img_path.split("/")[-1]

        print(f"正在提取第{idx + 1}张图片特征：{img_name}")
        features = extract_all_features(img_path)

        features_list.append(features)
        feature_dict = {
            "product_id": product_id,
            "product_category": product_categories[idx],
            "brand": product_brands[idx]
        }
        for i in range(len(features)):
            feature_dict[f"feature{i + 1}"] = features[i]
        result_data.append(feature_dict)

    result_df = pd.DataFrame(result_data)
    feature_matrix = np.array(features_list)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)

    return result_df, feature_matrix


if __name__ == "__main__":
    PIC_FOLDER = "pic"
    result_df, feature_matrix = batch_extract_features(PIC_FOLDER)

    os.makedirs("D:/杨谊瑶/大三上/大数据技术/第三次上机/img_feature_clustering/files/Tradition_results", exist_ok=True)
    result_df.to_csv("D:/杨谊瑶/大三上/大数据技术/第三次上机/img_feature_clustering/files/Tradition_results/tradition_feature.csv",
                     index=False, encoding="utf-8-sig")
    print(f"特征维度：{feature_matrix.shape[1]}维")
    print(f"处理图片数量：{feature_matrix.shape[0]}张")