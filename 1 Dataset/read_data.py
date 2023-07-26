from torch.utils.data import Dataset
from PIL import Image
import os


# 继承一个类
# 重写三个方法
class MyData(Dataset):

    def __init__(self, root_dir, lab_dir):
        self.root_dir = root_dir
        self.lab_dir = lab_dir
        self.path = os.path.join(root_dir, lab_dir)
        self.image_path = os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.image_path[index]
        img_item_path = os.path.join(self.root_dir, self.lab_dir, img_name)
        img = Image.open(img_item_path)
        label = self.lab_dir
        return img, label

    def __len__(self):
        return len(self.image_path)


root_dir = "dataset/train"
lab_dir = "ants"

ants_dataset = MyData(root_dir, lab_dir)

root_dir = "dataset/train"
lab_dir = "bees"

bees_dataset = MyData(root_dir, lab_dir)

train_dataset = ants_dataset + bees_dataset