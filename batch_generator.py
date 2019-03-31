import os
import numpy as np

from skimage import io, img_as_float
from skimage import transform
from torch.utils.data import Dataset


class BatchGenerator(Dataset):
    def __init__(self, dataset_path, output_shape=[20, 16], recognizable_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"):
        self.dataset_path = dataset_path
        self.output_shape = output_shape
        self.filenames = os.listdir(dataset_path)
        self.num_files = len(self.filenames)
        self.recognizable_characters = recognizable_characters
        self.recognizable_characters_dict = {char : num for num, char in enumerate(recognizable_characters)}
        self.num_labels = len(self.recognizable_characters)

    def __len__(self):
        return self.num_files
    
    def crop(self, binarized_image):
        non_empty_i, non_empty_j = np.where(binarized_image == 0)
        if len(non_empty_i) == 0 or len(non_empty_j) == 0:
            return binarized_image
        return binarized_image[non_empty_i.min():non_empty_i.max()+1, non_empty_j.min():non_empty_j.max()+1]

    def __getitem__(self, indx):
        img_name = self.filenames[indx]
        img_path = os.path.join(self.dataset_path, img_name)
        
        label = img_name[0].upper()
        label_index = self.recognizable_characters_dict[label]
        one_hot = np.zeros(self.num_labels)
        one_hot[label_index] = 1
        
        img = io.imread(img_path).mean(-1)
        img /= img.max()
        image_shape = img.shape
        #img = transform.resize(img, self.output_shape, mode='constant')
        binarized = img > .5
        binarized = self.crop(binarized)
        image_shape = binarized.shape
        binarized = transform.resize(binarized, self.output_shape, mode='constant') > .5
        
        binarized = binarized.astype(float).flatten()
        aspect_ratio = image_shape[1]/image_shape[0]
        output = np.zeros(self.output_shape[0] * self.output_shape[1] + 1)
        output[:-1] = binarized
        output[-1] = aspect_ratio
        return output, one_hot
        