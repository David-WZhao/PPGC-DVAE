import numpy as np
import umap
import matplotlib.pyplot as plt

# 两个十维数据
properties = np.array([[0.19276148, 0.19115417, 0.57944557, 0.05882353, 0.1707089, 0.03010101,
                        0.19777205, 0.41176471, 0.14705882, 0.35294118],
                       [0.3, 0.2, 0.7, 0.1, 0.3, 0.11111111,
                        0.34103396, 0.38461538, 0.19230769, 0.23076923]], dtype=np.float32)

# 使用UMAP降维
reducer = umap.UMAP(n_components=2)
embedding = reducer.fit_transform(properties)

# 绘制二维降维结果
plt.scatter(embedding[:, 0], embedding[:, 1], c=['blue', 'green'])
plt.title('2D projection of 10D data using UMAP')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
# 保存为图片
plt.savefig('umap_projection.png')
