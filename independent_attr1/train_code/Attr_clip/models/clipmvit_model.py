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
class CLIPmvit(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # visual_config=config.visual
        # text_config=config.text
        # embed_dim=config.dim

        ############################# visual encoder ###########################
            
        self.visual_net = MViT(cfg)

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

            return self.__scaled_product(image_features, text_features)


        else:
            # image,  text_feature -> logit
            
            image_feat = self.__encode_normed_image(image)
            return self.__scaled_product(image_feat, text)



    def __scaled_product(self, image_features, text_features):
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        # logits_per_text = logits_per_image.t()
        # shape = [global_batch_size, global_batch_size]

        return logits_per_image #, logits_per_text
