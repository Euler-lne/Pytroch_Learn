from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")
image_path = "data\\train\\bees_image\\16838648_415acd9e3f.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
# 需要进行类型转换
print(type(img_array))

writer.add_image("test", img_array, 2, dataformats='HWC')

# img_tensor (torch.Tensor, numpy.ndarray, or string/blobname): Image data
# global_step (int): Global step value to record 训练步骤
# 使用npmpy的话要注意三元组的顺序，来判断是否修改 dataformats='HWC'，看img_array.shape来观察

for i in range(100):
    writer.add_scalar("y=x", i, i)
writer.close()
# 打开日志文件 tensorboard --logdir=logs --port=6007
# --port指定端口号