# modified from https://github.com/openai/CLIP

from typing import Optional
from .clip_utils.tokenize import tokenize
from .clip_utils.model import TextTransformer, GatherLayer
from .mvit_model import MViT, TransformerBasicHead

import torch.nn.functional as F
import torch
from torch import nn

# from PIL import Image
# try:
#     from torchvision.transforms import InterpolationMode
#     BICUBIC = InterpolationMode.BICUBIC
# except ImportError:
#     BICUBIC = Image.BICUBIC


from .build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class CLIPmvit_clsloss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # visual_config=config.visual
        # text_config=config.text
        # embed_dim=config.dim

        ############################# visual encoder ###########################
            
        self.visual_net = MViT(cfg)
        self.head_cls = TransformerBasicHead(
            cfg.MODEL.NUM_CLASSES,
            cfg.MODEL.CLS_CLASSES,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
            cfg=cfg,
        )

        ############################# text encoder #############################
        self.text_net = TextTransformer(
            embed_dim=cfg.TEXT_NET.DIM,
            vocab_size=cfg.TEXT_NET.VOCAB_SIZE,
            context_length=cfg.TEXT_NET.CONTEXT_LENGTH,
            width=cfg.TEXT_NET.WIDTH,
            heads=cfg.TEXT_NET.HEADS,
            layers=cfg.TEXT_NET.LAYERS
        )

        self.logit_scale = nn.parameter.Parameter(
            torch.log(torch.tensor(1.0/cfg.TEMPERATURE))  )
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction="sum")

        self.loss_type = cfg.LOSS_TYPE
        assert self.loss_type in ["local", "global"]


    def __encode_image(self, image) -> torch.Tensor:
        return self.visual_net(image)

    def __encode_text(self, text_token):
        return self.text_net(text_token)

    def __encode_normed_image(self, image):
        return self.__normalize(self.__encode_image(image))

    def __encode_normed_text(self, text_token):
        return self.__normalize(self.__encode_text(text_token))

    def __normalize(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings / embeddings.norm(dim=-1, keepdim=True)

    def forward(self, image=None, text=None):

        if image is None and text is not None:
            # text(token) -> normed text_feature
            return self.__encode_normed_text(text)
        
        elif self.training:
            # image, token -> logit
            image_features = self.__encode_normed_image(image)
            text_features = self.__encode_normed_text(text)

            assert image_features.size(0) == text_features.size(0)

            if self.loss_type == "global":
                image_features = torch.cat(GatherLayer.apply(image_features), 0)
                text_features  = torch.cat(GatherLayer.apply(text_features), 0)

            return self.__scaled_product(image_features, text_features), self.head_cls(image_features)


        else:
            # image,  text_feature -> logit
            
            image_feat = self.__encode_normed_image(image)
            return self.__scaled_product(image_feat, text), self.head_cls(image_feat)



    def __scaled_product(self, image_features, text_features):
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        # logits_per_text = logits_per_image.t()
        # shape = [global_batch_size, global_batch_size]

        return logits_per_image #, logits_per_text


    # def info_nce_loss(self, logits_image, logits_text):
    #     """return loss_image, loss_text"""

    #     batchsize = logits_image.size(0)

    #     ground_truth = torch.arange(batchsize, 
    #         dtype=torch.long, device=logits_image.device)
    #     # size = [bz, bz]

    #     loss_image = self.ce_loss(logits_image, ground_truth)
    #     loss_text  = self.ce_loss(logits_text, ground_truth)

    #     return loss_image, loss_text


    


# def initialize_clip_weight(clip_model, config, strict=True):
#     """load pretrain weight"""
#     if config.initial.weight:
#         state_dict = torch.load(config.initial.weight)
#         if isinstance(state_dict, dict) and 'state_dict' in state_dict:
#             state_dict = state_dict['state_dict']
        
#         clip_pref = "clip_model."
#         state_dict = {
#             k[len(clip_pref):] if k.startswith(clip_pref) else k  :  v
#             for k,v in state_dict.items()
#         }


#         # for key in ["input_resolution", "context_length", "vocab_size"]:
#         #     if key in state_dict:
#         #         del state_dict[key]
        
#         if config.initial.extra_token == 'copy':
#             state_dict['visual_net.class_embedding'] = \
#                 state_dict['visual_net.class_embedding'].expand_as(clip_model.visual_net.class_embedding)
#             state_dict['visual_net.positional_embedding'] = torch.cat([
#                 state_dict['visual_net.positional_embedding'][0].unsqueeze(0).repeat(config.visual.extra_token, 1),
#                 state_dict['visual_net.positional_embedding'][1:],
#             ], dim=0)
#         elif config.initial.extra_token == 'rand':
#             state_dict['visual_net.class_embedding'] = torch.randn(*clip_model.visual_net.class_embedding.shape)
#             state_dict['visual_net.positional_embedding'] = torch.cat([
#                 torch.randn(config.visual.extra_token, state_dict['visual_net.positional_embedding'].shape[1]),
#                 state_dict['visual_net.positional_embedding'][1:],
#             ], dim=0)
#         missing_keys, unexpected_keys = clip_model.load_state_dict(state_dict, strict=False)

#         if strict:
#             assert len(unexpected_keys) == 0, str(unexpected_keys)
#             assert len(missing_keys) == 0, str(missing_keys)
#         else:
#             if len(unexpected_keys) > 0:
#                 print("unexpected_keys", str(unexpected_keys))
#             if len(missing_keys) > 0:
#                 print("missing_keys", str(missing_keys))

#     # freeze backbone
#     for p1 in config.freeze_params:
#         used = False
#         for name, param in clip_model.named_parameters():
#             if p1 in name:
#                 param.requires_grad_(False)
#                 used = True
#         assert used, 'Unrecognized parameter name: %s' % p1

#     # convert_weights(clip_model)



# def convert_weights(model: nn.Module):
#     """Convert applicable model parameters to fp16"""

#     def _convert_weights_to_fp16(l):
#         if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
#             l.weight.data = l.weight.data.half()
#             if l.bias is not None:
#                 l.bias.data = l.bias.data.half()

#         if isinstance(l, nn.MultiheadAttention):
#             for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
#                 tensor = getattr(l, attr)
#                 if tensor is not None:
#                     tensor.data = tensor.data.half()

#         for name in ["text_projection", "proj"]:
#             if hasattr(l, name):
#                 attr = getattr(l, name)
#                 if attr is not None:
#                     attr.data = attr.data.half()

#     model.apply(_convert_weights_to_fp16)
