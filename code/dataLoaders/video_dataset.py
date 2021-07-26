import os
import sys
sys.path.append('./code')

from sklearn.model_selection import train_test_split

import torch
import cv2
import numpy as np
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from mypath import Path


class VideoDataset(Dataset):
    """
    A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.
    Args:
        dataset (str): Name of dataset. Defaults to 'DMD'.
        split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
        clip_len (int): Determines how many frames are there in each clip. Defaults to 70.
        preprocess (bool): Determines whether to preprocess dataset. Default is False.
        transform(object): torchvison's transform
    """

    def __init__(self, dataset='DMD-lite-70', split='train', clip_len=70, preprocess=False, transform=None):

        self.root_dir, self.output_dir = Path.db_dir(dataset)
        folder = os.path.join(self.output_dir, split)

        self.clip_len = clip_len
        self.split = split

        self.transform = transform
        self.resize_height = 224
        self.resize_width = 224
        self.crop_size = 112

        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from official website.')

        if preprocess:
            print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
            self.preprocess()

        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                if len(os.listdir(os.path.join(folder,label,fname))) != 70:
                    continue
                else:
                    self.fnames.append(os.path.join(folder, label, fname))

                labels.append(label)


        assert len(labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        if dataset == "DMD-lite-70":
           # if not os.path.exists('./DMD-labels.json'):
            with open('./DMD-labels.json', 'w') as f:
                f.write(json.dumps(self.label2index, indent=4))

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer = self.load_frames(self.fnames[index])
        labels = np.array(self.label_array[index])

        # Abandon self-written processor
        # buffer = self.crop(buffer, self.clip_len, self.crop_size)
        # if self.split == 'test':
        #     # Perform data augmentation
        #     buffer = self.randomflip(buffer)

        buffer = self.normalize(buffer)
        # buffer = self.to_tensor(buffer)

        if self.transform is not None:
            buffer = self.transform(buffer)

        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def check_integrity(self):
        if not os.path.exists(self.root_dir):
            return False
        else:
            return True


    def preprocess(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, 'train'))
            os.mkdir(os.path.join(self.output_dir, 'val'))
            # os.mkdir(os.path.join(self.output_dir, 'test'))

        # Split train/val/test sets
        for file in os.listdir(self.root_dir):
            file_path = os.path.join(self.root_dir, file)
            video_files = [name for name in os.listdir(file_path)]

            # train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
            train, val = train_test_split(video_files, test_size=0.2, random_state=42)

            train_dir = os.path.join(self.output_dir, 'train', file)
            val_dir = os.path.join(self.output_dir, 'val', file)
            # test_dir = os.path.join(self.output_dir, 'test', file)

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(val_dir):
                os.mkdir(val_dir)
            # if not os.path.exists(test_dir):
            #     os.mkdir(test_dir)

            for video in train:
                self.process_video(video, file, train_dir)

            for video in val:
                self.process_video(video, file, val_dir)

            # for video in test:
            #     self.process_video(video, file, test_dir)

        print('Preprocessing finished.')


    def process_video(self, video, action_name, save_dir):
        # Initialize a VideoCapture object to read video data into a numpy array
        video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))

        capture = cv2.VideoCapture(os.path.join(self.root_dir, action_name, video))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Make sure splited video has at least 16 frames
        EXTRACT_FREQUENCY = 1
        if frame_count // EXTRACT_FREQUENCY < 70:
            print('Error. The clip only has %d frames which is less than 70 frames.' % frame_count)
            return None

        count = 0
        i = 0
        retaining = True

        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % EXTRACT_FREQUENCY == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
                i += 1
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()

    def load_frames(self, file_dir):
        
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame

        return buffer

    # Abandon self-written processor
    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    # Abandon self-written processor
    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    # Abandon self-written processor
    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    # Abandon self-written processor
    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        time_index = np.random.randint(buffer.shape[0] - clip_len)

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images_new = []
        labels_new = []
#         print(images[0].size())
        for i in range(len(images)):
            # print(images[i].size())
            images_new.append(images[i])
            labels_new.append(labels[i])

        images = torch.stack(images_new, dim=0)
        labels = torch.as_tensor(labels_new)
        return images, labels



# if __name__ == "__main__":

#     train_data = VideoDataset(dataset='DMD-lite-70', split='train', clip_len=70, preprocess=False)
#     train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=8)

#     for i, sample in enumerate(train_loader):
#         inputs = sample[0]
#         labels = sample[1]
#         print(inputs.size())
#         print(labels)
    
#         if i == 1:
#             break
