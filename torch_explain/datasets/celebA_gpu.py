import pandas as pd
import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

image_folder = 'D:\CelebA_shifted\CelebA_shifted\img_align_celeba\img_align_celeba'
data_path = 'D:\CelebA_shifted\CelebA_shifted\list_attr_celeba.txt'

columns = [
    "Image", "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
    "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
    "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
    "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache",
    "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"
]

selected_concepts = [
    "Arched_Eyebrows",
    "Heavy_Makeup",
    "No_Beard",
    "Wearing_Earrings",
    "Wearing_Lipstick"
]

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CelebADataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.image_folder = image_folder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.data.iloc[idx, 0])
        with Image.open(img_name) as img:
            if self.transform:
                img = self.transform(img)
        concepts = torch.FloatTensor(self.data.iloc[idx, 1])
        isMale = torch.FloatTensor([self.data.iloc[idx, 2]])
        return img, concepts, isMale

def celebA(size, top_attributes=True):
    if size > 20699:
        size = 20699

    data = pd.read_csv(data_path, delim_whitespace=True, skiprows=1, header=None)
    data.columns = columns

    if top_attributes:
        true_counts = (data.iloc[:, 1:] == 1).sum()
        attributes_name = true_counts.nlargest(5).index.tolist()
    else:
        attributes_name = selected_concepts
    boolean_data = data[attributes_name] == 1
    data['Concepts'] = boolean_data.values.tolist()
    boolean_data2 = data["Male"] == 1
    data['isMale'] = boolean_data2.values.tolist()

    new_data = data[['Image', 'Concepts', 'isMale']].head(size)

    dataset = CelebADataset(new_data, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    x, c, y = [], [], []

    for images, concepts, labels in dataloader:
        x.append(images)
        c.append(concepts)
        y.append(labels)

    x = torch.cat(x)
    c = torch.cat(c)
    y = torch.cat(y)

    return x, c, y, attributes_name

def main():
    x, c, y, concepts_names = celebA(10, False)
    print(x.shape)
    print(c.shape)
    print(y.shape)
    #print("\nX:\n")
    #print(x.shape)
    #print("\nC:\n")
    #print(c.shape)
    #print("\nY:\n")
    #print(y.shape)
    #print("\nConcepts:\n")
    print(concepts_names)

if __name__ == '__main__':
    main()