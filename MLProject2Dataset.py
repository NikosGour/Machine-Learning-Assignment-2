import glob
import os

import torch
import pandas as pd
import torchvision
from torch.utils.data import Dataset


class MLProject2Dataset(Dataset):
    def __init__(self, data_dir, metadata_fname='metadata.csv', transform: torchvision.transforms = None):
        self.data_dir = data_dir
        self.transform = transform

        self.df = pd.DataFrame(columns=['image_id', 'path'])

        # Get all files in data_dir
        files = glob.glob(data_dir + '/part_1/*.jpg')
        # If on windows, replace backslash with forward slash
        # on linux, this does nothing
        files = [file.replace(os.sep, '/') for file in files]
        files = files[:1000]

        # Add all files to dataframe
        for file in files:
            image_id = file.split('/')[-1].split('.')[0]
            path = file

            # Add to dataframe
            self.df.loc[len(self.df)] = [image_id, path]

        # Add metadata
        self.metadata = pd.read_csv(os.path.join(data_dir, metadata_fname))
        dx_categorical = pd.Categorical(self.metadata['dx'])
        self.number_to_label_array = [dx for dx in dx_categorical.categories]

        self.metadata['dx'] = dx_categorical.codes

        # Merge metadata with dataframe
        self.df = self.df.merge(self.metadata, on='image_id')

        self.df = self.df.drop(columns=["lesion_id", "dx_type", "age", "sex", "localization"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get image path
        image_path = self.df.iloc[idx]['path']

        # Load image
        image = torchvision.io.read_image(image_path)

        # Change pixel values to float
        image = image.to(torch.float32)

        # Normalize image
        image = image / 255.0

        # Get label
        label = self.df.iloc[idx]['dx']

        if self.transform:
            image = self.transform(image)

        return image, label



x = MLProject2Dataset('data')
print()
