import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Base dataset used for both IU X-Ray and MIMIC-CXR."""

    def __init__(self, args, tokenizer, split, transform=None, note_tokenizer=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.note_tokenizer = note_tokenizer
        self.transform = transform
        self.use_note = getattr(args, 'use_note', False)
        self.use_labels = getattr(args, 'use_labels', False)

        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])
            if self.use_note and 'impression' in self.examples[i] and self.note_tokenizer is not None:
                note_encoded = self.note_tokenizer(
                    self.examples[i]['impression'],
                    truncation=True,
                    max_length=getattr(args, 'max_text_len', 128),
                    padding='max_length'
                )
                self.examples[i]['note_ids'] = note_encoded['input_ids']
                self.examples[i]['note_mask'] = note_encoded['attention_mask']
            if self.use_labels and 'labels' in self.examples[i]:
                self.examples[i]['labels'] = self.examples[i]['labels']

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        note_ids = example.get('note_ids', None)
        note_mask = example.get('note_mask', None)
        labels = example.get('labels', None)
        sample = (image_id, image, report_ids, report_masks, seq_length, note_ids, note_mask, labels)
        return sample
