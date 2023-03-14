# Neural_control
该项目主要任务是进行神经信号编码，即通过神经元对自然图片反应的数据进行训练后，能够实现输入一张图片返回单个或群体神经元的反应。

需要学习三个参数：

W_s：代表该神经元的感受野，数值代表其关注的区域

W_d：对每个channel的权重，以便对特征图在感受野上的信息进行加权求和

W_b：偏置

此外还需要有一个网络对图片提取特征，以Alexnet为例，经测试V1细胞与conv1相似度最高，V4细胞与conv3相似度最高。

因此针对于V1的细胞，我们可以先把图片经过Alexnet的conv1进行特征提取，随后把特征图作为当前图片进行感受野和权重的训练。

