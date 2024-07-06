import pandas as pd
import numpy as np
import os
from PIL import Image
import torch
from torchvision import transforms
import time

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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def image_to_tensor(image_name):
    image_path = os.path.join(image_folder, image_name)
    with Image.open(image_path) as img:
        tensor = transform(img)
    return tensor



def celebA(size):

    if (size > 20699):
        size = 20699

    data = pd.read_csv(data_path, delim_whitespace=True, skiprows=1, header=None)
    data.columns = columns

    true_counts = (data.iloc[:, 1:] == 1).sum()
    top_attributes = true_counts.nlargest(5).index.tolist()
    boolean_data = data[top_attributes] == 1
    data['Concepts'] = boolean_data.values.tolist()
    boolean_data2 = data["Male"] == 1
    data['isMale'] = boolean_data2.values.tolist()

    new_data = data[['Image', "Concepts", "isMale"]].head(size)

    tensors = [image_to_tensor(img_name) for img_name in new_data['Image']]

    x = torch.stack(tensors)
    c = torch.FloatTensor(new_data['Concepts'].values.tolist())
    y = torch.FloatTensor(new_data['isMale'].values.tolist())

    return x, c, y.unsqueeze(-1), top_attributes



def main():
    x, c, y, concepts_names = celebA(10)
    print("\nX:\n")
    print(x)
    print("\nC:\n")
    print(c)
    print("\nY:\n")
    print(y)
    print("\nConcepts:\n")
    print(concepts_names)

if __name__ == '__main__':
    main()