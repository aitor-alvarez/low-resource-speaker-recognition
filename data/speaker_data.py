import os
import torchaudio
from torch.utils.data import Dataset
import pandas as pd


class VoxCeleb2Dataset(Dataset):

	def __init__(self, transform=None):

		path = './voxceleb2_veri_dev.txt'
		split = pd.read_table(path, sep=' ', header=None, names=['label', 'path'])
		self.dataset = split['path']
		self.labels = split['label']


	def __getitem__(self, item):

		label = self.labels[item]

		track_path = self.dataset[item]

		audio_path = os.path.join(track_path, 'voxceleb1_wav', track_path)
		print(audio_path)
		waveform, sample_rate = torchaudio.load(audio_path)

		return waveform, label


	def __len__(self):
		return len(self.dataset)




