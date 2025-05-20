import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder
try:
    from transformers import AutoModel
except ImportError:
    AutoModel = None


class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer, note_tokenizer=None):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.note_tokenizer = note_tokenizer

        self.use_note = getattr(args, 'use_note', False)
        self.use_labels = getattr(args, 'use_labels', False)
        self.use_adapter = getattr(args, 'use_adapter', False)
        self.adapter_dim = getattr(args, 'adapter_dim', 64)
        self.use_prompt = getattr(args, 'use_prompt', False)
        self.prompt_len = getattr(args, 'prompt_len', 5)

        self.visual_extractor = VisualExtractor(args)
        self.img_proj = nn.Linear(args.d_vf, args.d_model)

        if self.use_note and AutoModel is not None:
            self.text_encoder = AutoModel.from_pretrained(args.text_encoder)
            self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, args.d_model)
            self.modal_embed = nn.Parameter(torch.zeros(2, args.d_model))
        else:
            self.text_encoder = None

        if self.use_labels:
            cls_dim = args.d_vf if args.dataset_name != 'iu_xray' else args.d_vf * 2
            self.classifier = nn.Linear(cls_dim, 14)
        else:
            self.classifier = None

        self.encoder_decoder = EncoderDecoder(args, tokenizer)

        if self.use_prompt:
            self.prompt_embeddings = nn.Parameter(torch.randn(self.prompt_len, args.d_model))
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train', note_ids=None, note_mask=None):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        att_feats = self.img_proj(att_feats)

        if self.use_note and note_ids is not None and self.text_encoder is not None:
            txt = self.text_encoder(input_ids=note_ids, attention_mask=note_mask).last_hidden_state
            txt = self.text_proj(txt)
            img = att_feats + self.modal_embed[0]
            txt = txt + self.modal_embed[1]
            att_feats = torch.cat([img, txt], dim=1)

        if self.use_prompt:
            bs = att_feats.size(0)
            prompts = self.prompt_embeddings.unsqueeze(0).expand(bs, -1, -1)
            att_feats = torch.cat([prompts, att_feats], dim=1)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError

        cls_logits = self.classifier(fc_feats) if self.classifier is not None else None
        return output, cls_logits

    def forward_mimic_cxr(self, images, targets=None, mode='train', note_ids=None, note_mask=None):
        att_feats, fc_feats = self.visual_extractor(images)
        att_feats = self.img_proj(att_feats)

        if self.use_note and note_ids is not None and self.text_encoder is not None:
            txt = self.text_encoder(input_ids=note_ids, attention_mask=note_mask).last_hidden_state
            txt = self.text_proj(txt)
            img = att_feats + self.modal_embed[0]
            txt = txt + self.modal_embed[1]
            att_feats = torch.cat([img, txt], dim=1)

        if self.use_prompt:
            bs = att_feats.size(0)
            prompts = self.prompt_embeddings.unsqueeze(0).expand(bs, -1, -1)
            att_feats = torch.cat([prompts, att_feats], dim=1)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError

        cls_logits = self.classifier(fc_feats) if self.classifier is not None else None
        return output, cls_logits

