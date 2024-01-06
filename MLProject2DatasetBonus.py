import glob
import os
import PIL.Image
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset


class MLProject2DatasetBonus(Dataset):
    def __init__(self, data_dir:str, metadata_fname='metadata.csv', transform: torchvision.transforms = None):
        self.data_dir = data_dir
        self.transform = transform

        self.df = pd.DataFrame(columns=['image_id', 'path'])

        # Get all files in data_dir
        files = glob.glob(data_dir + '/*/*.jpg')
        # If on windows, replace backslash with forward slash
        # on linux, this does nothing
        files = [file.replace(os.sep, '/') for file in files]
        # The line commented bellow was used for testing on a smaller dataset
        # files = files[:1000]

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

        self.df = self.df.drop(columns=["lesion_id", "dx_type"])

        self.df['age'] = self.df['age'].map(lambda x: x/100)
        self.df = pd.get_dummies(self.df, columns=["localization",'sex'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get image path
        image_path = self.df.iloc[idx]['path']

        # Load image
        image = PIL.Image.open(image_path)

        # Change pixel values to float
        image = image.convert('RGB')

        # Normalize image
        image = torchvision.transforms.ToTensor()(image)

        demographic_columns = ['localization_abdomen',
       'localization_acral', 'localization_back', 'localization_chest',
       'localization_ear', 'localization_face', 'localization_foot',
       'localization_genital', 'localization_hand',
       'localization_lower extremity', 'localization_neck',
       'localization_scalp', 'localization_trunk', 'localization_unknown',
       'localization_upper extremity', 'sex_female', 'sex_male',
       'sex_unknown']

        demographic_vector = self.df.iloc[idx][demographic_columns].values
        demographic_vector = demographic_vector.astype('float32')
        demographic_vector = torch.tensor(demographic_vector)
        # Get label
        label = self.df.iloc[idx]['dx']

        if self.transform:
            image = self.transform(image)

        return image,demographic_vector, label

