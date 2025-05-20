import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from .datasets import IuxrayMultiImageDataset, MimiccxrSingleImageDataset

try:
    from transformers import AutoTokenizer
except ImportError:  # transformers may not be available during static checks
    AutoTokenizer = None


class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle, note_tokenizer=None):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.note_tokenizer = note_tokenizer
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset(
                self.args, self.tokenizer, self.split, transform=self.transform, note_tokenizer=self.note_tokenizer
            )
        else:
            self.dataset = MimiccxrSingleImageDataset(
                self.args, self.tokenizer, self.split, transform=self.transform, note_tokenizer=self.note_tokenizer
            )

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        # support optional note and label information
        if len(data[0]) == 5:
            images_id, images, reports_ids, reports_masks, seq_lengths = zip(*data)
            note_ids = note_masks = labels = None
        else:
            images_id, images, reports_ids, reports_masks, seq_lengths, note_ids, note_masks, labels = zip(*data)
        images = torch.stack(images, 0)
        max_seq_length = max(seq_lengths)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks

        batch = [images_id, images, torch.LongTensor(targets), torch.FloatTensor(targets_masks)]

        if note_ids is not None and note_masks is not None:
            max_note_len = max(len(n) for n in note_ids)
            notes = np.zeros((len(note_ids), max_note_len), dtype=int)
            note_mask_mat = np.zeros((len(note_ids), max_note_len), dtype=int)
            for i, n in enumerate(note_ids):
                notes[i, :len(n)] = n
            for i, m in enumerate(note_masks):
                note_mask_mat[i, :len(m)] = m
            batch.extend([torch.LongTensor(notes), torch.LongTensor(note_mask_mat)])
            if labels is not None:
                batch.append(torch.FloatTensor(labels))
        return tuple(batch)

