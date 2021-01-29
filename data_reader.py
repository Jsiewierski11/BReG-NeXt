
from dataclasses import dataclass
from collections import OrderedDict

class FER2013Dataset_Alternative(Dataset):
    """
    Face Expression Recognition Dataset -https://medium.com/analytics-vidhya/read-fer2013-face-expression-recognition-dataset-using-pytorch-torchvision-9ff64f55018e
    """
    
    def __init__(self, file_path):
        """
        Args:
            file_path (string): Path to the csv file with emotion, pixel & usage.
        """
        self.file_path = file_path
        self.classes = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral') # Define the name of classes / expression
        
        with open(self.file_path) as f: # read all the csv using readlines
            self.data = f.readlines()
            
        self.total_images = len(self.data) - 1 #reduce 1 for row of column

    def __len__(self):  
        return self.total_images
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        with open(fer_path) as f:
            emotion, img, usage = self.data[idx + 1].split(",") #plus 1 to skip first row (column name)
            
        emotion = int(emotion) # just make sure it is int not str
        img = img.split(" ") # because the pixels are seperated by space
        img = np.array(img, 'int') # just make sure it is int not str
        img = img.reshape(48,48) # change shape from 2304 to 48 * 48

        sample = {'image': img, 'emotion': emotion}
        
        return sample