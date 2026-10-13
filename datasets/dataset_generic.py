from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
from scipy import stats
from torch.utils.data import Dataset
import h5py

from utils.utils import generate_split, nth

def save_splits(split_datasets, column_keys, filename, boolean_style=False):
	# 处理 split_datasets 中可能存在 None 的情况
	splits = []
	for i in range(len(split_datasets)):
		if split_datasets[i] is not None:
			splits.append(split_datasets[i].slide_data['slide_id'])
		else:
			# 如果 split 是 None，添加空的 Series
			splits.append(pd.Series(dtype=str))
	if not boolean_style:
		df = pd.concat(splits, ignore_index=True, axis=1)
		df.columns = column_keys
	else:
		df = pd.concat(splits, ignore_index = True, axis=0)
		index = df.values.tolist()
		one_hot = np.eye(len(split_datasets)).astype(bool)
		bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
		df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])

	df.to_csv(filename)
	print()

class Generic_WSI_Classification_Dataset(Dataset):
	def __init__(self,
				 csv_path = 'dataset_csv/ccrcc_clean.csv',
				 mode = 'clam',
				 shuffle = False,
				 seed = 7,
				 print_info = True,
				 label_dict = {},
				 filter_dict = {},
				 ignore=[],
				 patient_strat=False,
				 label_col = None,
				 patient_voting = 'max',
				 ):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			shuffle (boolean): Whether to shuffle
			seed (int): random seed for shuffling the data
			print_info (boolean): Whether to print a summary of the dataset
			label_dict (dict): Dictionary with key, value pairs for converting str labels to int
			ignore (list): List containing class labels to ignore
		"""
		self.label_dict = label_dict
		self.num_classes = len(set(self.label_dict.values()))
		self.seed = seed
		self.print_info = print_info
		self.patient_strat = patient_strat
		self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
		self.data_dir_s = None
		self.data_dir_l = None
		if not label_col:
			label_col = 'label'
		self.label_col = label_col

		slide_data = pd.read_csv(csv_path, encoding='utf-8-sig')
		slide_data = self.filter_df(slide_data, filter_dict)
		slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col)

		if shuffle:
			np.random.seed(seed)
			np.random.shuffle(slide_data)

		self.slide_data = slide_data

		self.patient_data_prep(patient_voting)
		self.mode = mode
		self.cls_ids_prep()

		if print_info:
			self.summarize()

	def cls_ids_prep(self):
		self.patient_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def patient_data_prep(self, patient_voting='max'):
		patients = np.unique(np.array(self.slide_data['case_id'])) 
		patient_labels = []

		for p in patients:
			locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
			assert len(locations) > 0
			label = self.slide_data['label'][locations].values
			if patient_voting == 'max':
				label = label.max() 
			elif patient_voting == 'maj':
				# label = stats.mode(label)[0]
				values, counts = np.unique(label, return_counts=True)
				label = values[np.argmax(counts)]
			else:
				raise NotImplementedError
			patient_labels.append(label)

		self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}

	@staticmethod
	def df_prep(data, label_dict, ignore, label_col):
		if label_col != 'label':
			data['label'] = data[label_col].copy()

		mask = data['label'].isin(ignore)
		data = data[~mask]
		data.reset_index(drop=True, inplace=True)
		for i in data.index:
			key = data.loc[i, 'label']
			data.at[i, 'label'] = label_dict[key]

		return data

	def filter_df(self, df, filter_dict={}):
		if len(filter_dict) > 0:
			filter_mask = np.full(len(df), True, bool)
			for key, val in filter_dict.items():
				mask = df[key].isin(val)
				filter_mask = np.logical_and(filter_mask, mask)
			df = df[filter_mask]
		return df

	def __len__(self):
		if self.patient_strat:
			return len(self.patient_data['case_id'])

		else:
			return len(self.slide_data)

	def summarize(self):
		print("label column: {}".format(self.label_col))
		print("label dictionary: {}".format(self.label_dict))
		print("number of classes: {}".format(self.num_classes))
		print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
		for i in range(self.num_classes):
			print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
			print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

	def create_splits(self, k = 3, val_num = (25, 25), test_num = (40, 40), label_frac = 1.0, custom_test_ids = None):
		settings = {
			'n_splits' : k,
			'val_num' : val_num,
			'test_num': test_num,
			'label_frac': label_frac,
			'seed': self.seed,
			'custom_test_ids': custom_test_ids
		}

		if self.patient_strat:
			settings.update({'cls_ids' : self.patient_cls_ids, 'samples': len(self.patient_data['case_id'])})
		else:
			settings.update({'cls_ids' : self.slide_cls_ids, 'samples': len(self.slide_data)})

		self.split_gen = generate_split(**settings)

	def set_splits(self,start_from=None):
		if start_from:
			ids = nth(self.split_gen, start_from)

		else:
			ids = next(self.split_gen)

		if self.patient_strat:
			slide_ids = [[] for i in range(len(ids))]

			for split in range(len(ids)):
				for idx in ids[split]:
					case_id = self.patient_data['case_id'][idx]
					slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
					slide_ids[split].extend(slide_indices)

			self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]

		else:
			self.train_ids, self.val_ids, self.test_ids = ids

	def get_split_from_df(self, all_splits, split_key='train'):
		split = all_splits[split_key]
		split = split.dropna().reset_index(drop=True)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(split.tolist())
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(df_slice, data_dir_s=self.data_dir_s, data_dir_l=self.data_dir_l, mode=self.mode, num_classes=self.num_classes)
		else:
			split = None

		return split

	def get_merged_split_from_df(self, all_splits, split_keys=['train']):
		merged_split = []
		for split_key in split_keys:
			split = all_splits[split_key]
			split = split.dropna().reset_index(drop=True).tolist()
			merged_split.extend(split)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(merged_split)
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(df_slice, data_dir_s=self.data_dir_s, data_dir_l=self.data_dir_l, mode=self.mode, num_classes=self.num_classes)
		else:
			split = None

		return split

	def return_splits(self, from_id=True, csv_path=None):

		if from_id:
			if len(self.train_ids) > 0:
				train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
				train_split = Generic_Split(train_data, data_dir_s=self.data_dir_s, data_dir_l=self.data_dir_l, mode=self.mode,  num_classes=self.num_classes)

			else:
				train_split = None

			if len(self.val_ids) > 0:
				val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
				val_split = Generic_Split(val_data, data_dir_s=self.data_dir_s, data_dir_l=self.data_dir_l, mode=self.mode,  num_classes=self.num_classes)

			else:
				val_split = None

			if len(self.test_ids) > 0:
				test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
				test_split = Generic_Split(test_data, data_dir_s=self.data_dir_s, data_dir_l=self.data_dir_l, num_classes=self.num_classes)

			else:
				test_split = None


		else:
			assert csv_path
			all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)  
			train_split = self.get_split_from_df(all_splits, 'train')  
			
			# 支持只有 train 和 test 的划分（没有 val）
			if 'val' in all_splits.columns:
				val_split = self.get_split_from_df(all_splits, 'val')
			else:
				print("Warning: 'val' column not found in splits file. Skipping validation set.")
				val_split = None
				
			test_split = self.get_split_from_df(all_splits, 'test') 

		return train_split, val_split, test_split

	def get_list(self, ids):
		return self.slide_data['slide_id'][ids]

	def getlabel(self, ids):
		return self.slide_data['label'][ids]

	def __getitem__(self, idx):
		return None

	def test_split_gen(self, return_descriptor=False):

		if return_descriptor:
			index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in range(self.num_classes)]
			columns = ['train', 'val', 'test']
			df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index= index,
							  columns= columns)

		count = len(self.train_ids)
		print('\nnumber of training samples: {}'.format(count))
		labels = self.getlabel(self.train_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'train'] = counts[u]

		count = len(self.val_ids)
		print('\nnumber of val samples: {}'.format(count))
		labels = self.getlabel(self.val_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'val'] = counts[u]

		count = len(self.test_ids)
		print('\nnumber of test samples: {}'.format(count))
		labels = self.getlabel(self.test_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'test'] = counts[u]

		assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
		assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
		# assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

		if return_descriptor:
			return df

	def save_split(self, filename):
		train_split = self.get_list(self.train_ids)
		val_split = self.get_list(self.val_ids)
		test_split = self.get_list(self.test_ids)
		df_tr = pd.DataFrame({'train': train_split})
		df_v = pd.DataFrame({'val': val_split})
		df_t = pd.DataFrame({'test': test_split})
		df = pd.concat([df_tr, df_v, df_t], axis=1)
		df.to_csv(filename, index = False)


class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
	def __init__(self,
				data_dir_s,
				data_dir_l,
				mode,
				**kwargs):

		super(Generic_MIL_Dataset, self).__init__(**kwargs)
		self.data_dir_s = data_dir_s
		self.data_dir_l = data_dir_l
		self.mode = mode
		self.use_h5 = False

	def load_from_h5(self, toggle):
		self.use_h5 = toggle

	def __getitem__(self, idx):
		slide_id = self.slide_data['slide_id'][idx]
		slide_id = slide_id.replace('.svs', '')
		slide_id = slide_id.replace('.mrxs', '')
		label = self.slide_data['label'][idx]
		if type(self.data_dir_s) == dict and type(self.data_dir_l) == dict:
			source = self.slide_data['source'][idx]
			data_dir_s = self.data_dir_s['source']
			data_dir_l = self.data_dir_l['source']
		else:
			data_dir_s = self.data_dir_s
			data_dir_l = self.data_dir_l

		if not self.use_h5:
			if self.data_dir_s and self.data_dir_l:
				if(self.mode == 'transformer'):
					scale_s = h5py.File(os.path.join(data_dir_s, '{}.h5'.format(slide_id)), 'r')
					scale_l = h5py.File(os.path.join(data_dir_l, '{}.h5'.format(slide_id)), 'r')
					features_s = torch.from_numpy(np.array(scale_s['features']))
					coords_s = torch.from_numpy(np.array(scale_s['coords']))
					features_l = torch.from_numpy(np.array(scale_l['features']))
					coords_l = torch.from_numpy(np.array(scale_l['coords']))
					return features_s, coords_s, features_l, coords_l, label

			else:
				return slide_id, label

		else:
			full_path = os.path.join(data_dir_s, 'h5_files', '{}.h5'.format(slide_id))
			with h5py.File(full_path,'r') as hdf5_file:
				features = hdf5_file['features'][:]
				coords = hdf5_file['coords'][:]

			features = torch.from_numpy(features)
			return features, label, coords


class Generic_MTL_Dataset(Generic_WSI_Classification_Dataset):
	"""Extended dataset class for multi-task learning"""
	def __init__(self,
			csv_path,
			mode,
			data_dir_s,
			data_dir_l,
			aux_label_cols=None,
			**kwargs):

		self.aux_label_cols = aux_label_cols if aux_label_cols is not None else []
		# For MTL, we don't need a single label column
		# Temporarily set label_col to a dummy value to avoid KeyError
		if 'label_col' not in kwargs:
			kwargs['label_col'] = 'slide_id'  # Use slide_id as dummy label column
			
		# Create a dummy label_dict if not provided
		if 'label_dict' not in kwargs:
			kwargs['label_dict'] = {}

		# Remove aux_label_cols from kwargs to avoid passing it to parent class
		if 'aux_label_cols' in kwargs:
			kwargs.pop('aux_label_cols')

		super(Generic_MTL_Dataset, self).__init__(csv_path=csv_path, mode=mode, **kwargs)
		self.data_dir_s = data_dir_s
		self.data_dir_l = data_dir_l
		self.mode = mode
		self.use_h5 = False
		

	def load_from_h5(self, toggle):
		self.use_h5 = toggle

	def df_prep(self, data, label_dict, ignore, label_col):
		"""Override df_prep to handle MTL case without needing 'label' column"""
		# For MTL, we don't need to process a single label column
		# Just return the data as-is, ignoring the label processing
		return data

	def cls_ids_prep(self):
		"""Override cls_ids_prep since we don't have a single label column"""
		# For MTL, we don't need class IDs for a single task
		pass

	def patient_data_prep(self, patient_voting='max'):
		"""Override patient_data_prep since we don't have a single label column"""
		# For MTL, we don't need patient-level data for a single task
		pass

	def summarize(self):
		"""Override summarize to show MTL-specific info"""
		print("Multi-Task Learning Dataset")
		print("Auxiliary label columns:", self.aux_label_cols)
		print("Number of samples:", len(self.slide_data))
	
	def get_task_class_indices(self, task_idx, class_idx):
		"""
		Get indices of samples belonging to a specific class for a specific auxiliary task.
		
		Args:
			task_idx: Index of the auxiliary task
			class_idx: Class label to filter by
			
		Returns:
			np.array of indices
		"""
		if task_idx >= len(self.aux_label_cols):
			return np.array([])
		
		col = self.aux_label_cols[task_idx]
		if col not in self.slide_data.columns:
			return np.array([])
		
		mask = self.slide_data[col] == class_idx
		indices = np.where(mask)[0]
		return indices

	def __getitem__(self, idx):
		slide_id = self.slide_data['slide_id'].iloc[idx]
		slide_id = slide_id.replace('.svs', '')                   #删去后缀svs
		slide_id = slide_id.replace('.mrxs', '') 
		if type(self.data_dir_s) == dict and type(self.data_dir_l) == dict:
			source = self.slide_data['source'].iloc[idx]
			data_dir_s = self.data_dir_s['source']
			data_dir_l = self.data_dir_l['source']
		else:
			data_dir_s = self.data_dir_s
			data_dir_l = self.data_dir_l

		if not self.use_h5:
			if self.data_dir_s and self.data_dir_l:
				if(self.mode == 'transformer'):
					scale_s = h5py.File(os.path.join(data_dir_s, '{}.h5'.format(slide_id)), 'r')
					scale_l = h5py.File(os.path.join(data_dir_l, '{}.h5'.format(slide_id)), 'r')
					features_s = torch.from_numpy(np.array(scale_s['features']))
					coords_s = torch.from_numpy(np.array(scale_s['coords']))
					features_l = torch.from_numpy(np.array(scale_l['features']))
					coords_l = torch.from_numpy(np.array(scale_l['coords']))
					return features_s, coords_s, features_l, coords_l

			else:
				return slide_id

		else:
			full_path = os.path.join(data_dir_s, 'h5_files', '{}.h5'.format(slide_id))
			with h5py.File(full_path,'r') as hdf5_file:
				features = hdf5_file['features'][:]
				coords = hdf5_file['coords'][:]

			features = torch.from_numpy(features)
			return features, coords


class Generic_Split(Generic_MIL_Dataset):
	def __init__(self, slide_data, data_dir_s=None, data_dir_l=None, mode='clam', num_classes=2):
		self.use_h5 = False
		self.slide_data = slide_data
		self.data_dir_s = data_dir_s
		self.data_dir_l = data_dir_l
		self.mode = mode
		self.num_classes = num_classes
		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def __len__(self):
		return len(self.slide_data)


class Generic_MTL_Split(Generic_MIL_Dataset):
	"""Extended dataset class for multi-task learning with Cox regression and auxiliary tasks"""
	def __init__(self, slide_data, data_dir_s=None, data_dir_l=None, mode='transformer', 
	             cox_duration_col='duration', cox_event_col='event', 
	             aux_label_cols=None, main_task_type='cox', main_task_label_col='final_label'):
		self.use_h5 = False
		self.slide_data = slide_data
		self.data_dir_s = data_dir_s
		self.data_dir_l = data_dir_l
		self.mode = mode
		self.cox_duration_col = cox_duration_col
		self.cox_event_col = cox_event_col
		self.aux_label_cols = aux_label_cols if aux_label_cols is not None else []
		self.main_task_type = main_task_type
		self.main_task_label_col = main_task_label_col
		
	def __len__(self):
		return len(self.slide_data)
	
	def __getitem__(self, idx):
		slide_id = self.slide_data['slide_id'].iloc[idx]
		slide_id = slide_id.replace('.svs', '')                   #删去后缀svs
		slide_id = slide_id.replace('.mrxs', '') 
		
		# Load image features
		if self.data_dir_s and self.data_dir_l:
			if self.mode == 'transformer':
				scale_s = h5py.File(os.path.join(self.data_dir_s, '{}.h5'.format(slide_id)), 'r')
				scale_l = h5py.File(os.path.join(self.data_dir_l, '{}.h5'.format(slide_id)), 'r')
				features_s = torch.from_numpy(np.array(scale_s['features']))
				coords_s = torch.from_numpy(np.array(scale_s['coords']))
				features_l = torch.from_numpy(np.array(scale_l['features']))
				coords_l = torch.from_numpy(np.array(scale_l['coords']))
			else:
				features_s = None
				coords_s = None
				features_l = None
				coords_l = None
		else:
			features_s = None
			coords_s = None
			features_l = None
			coords_l = None
		
		# Load main task labels
		cox_duration = None
		cox_event = None
		main_task_label = None
		
		if self.main_task_type == 'cox' or self.main_task_type == 'mdn':
			# Load Cox regression labels (also used for MDN)
			if self.cox_duration_col in self.slide_data.columns:
				dur_val = self.slide_data[self.cox_duration_col].iloc[idx]
				if pd.notna(dur_val):
					cox_duration = torch.tensor(float(dur_val), dtype=torch.float32)
			if self.cox_event_col in self.slide_data.columns:
				evt_val = self.slide_data[self.cox_event_col].iloc[idx]
				if pd.notna(evt_val):
					cox_event = torch.tensor(int(evt_val), dtype=torch.float32)
		elif self.main_task_type == 'binary':
			# Load binary main task label
			if 'final_label' in self.slide_data.columns:
				label_val = self.slide_data['final_label'].iloc[idx]
				if pd.notna(label_val):
					main_task_label = torch.tensor(int(label_val), dtype=torch.long)
		# else: main_task_type == 'none', no main task labels to load
		
		# Load auxiliary task labels
		aux_labels = []
		for col in self.aux_label_cols:
			if col in self.slide_data.columns:
				label_val = self.slide_data[col].iloc[idx]
				if pd.notna(label_val):
					aux_labels.append(torch.tensor(int(label_val), dtype=torch.long))
				else:
					aux_labels.append(None)
			else:
				aux_labels.append(None)
		
		return features_s, coords_s, features_l, coords_l, cox_duration, cox_event, main_task_label, aux_labels