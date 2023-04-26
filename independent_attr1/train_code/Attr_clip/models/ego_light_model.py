# modified from https://github.com/openai/CLIP

from models.clip_utils.visual_prompy import Aggregator, AtemporalProbe
from models.clipimg_model import CLIPimg
from .clip_utils.tokenize import tokenize
from .clip_utils.model import TextTransformer, GatherLayer, VisionTransformer

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
class EgoLightNet(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.atp_frames = cfg.VISUAL_NET.atp_frames
        
        visual_config=cfg.VISUAL_NET
        # text_config=config.text

        ############################# visual encoder ###########################
        
        if visual_config.type == "ViT":
            self.visual_net = VisionTransformer(
                input_resolution=visual_config.resolution,
                patch_size=visual_config.patch_size,
                width=visual_config.width,
                layers=visual_config.layers,
                heads=visual_config.heads,
                output_dim=visual_config.dim,
                extra_token=visual_config.extra_token
            )

        self.aggregator = AtemporalProbe(cfg).eval()

        if visual_config.aggregation is not None and visual_config.aggregation != "none":
            if visual_config.aggregation == "atp":
                self.second_aggregator = AtemporalProbe(cfg)
            else:
                self.second_aggregator = Aggregator(cfg)
        else:
            self.second_aggregator = None

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
        batchsize, total_frame, ch, h, w = image.shape
        selected_frame = total_frame // self.atp_frames
        image = image.view(-1, ch, h, w)
        # (bz, frame, ch, w, w) -> (bz*frame, ch, w, w)

        features = self.visual_net(image) # (bz*frame, dim)

        n_dim = features.shape[-1]
        features = features.view(batchsize*selected_frame, self.atp_frames, n_dim)
        features = self.aggregator(features)   # (bz*selected_frame, dim)

        features = features.view(batchsize, selected_frame, n_dim)
        # (bz, selected_frame, dim)

        if self.second_aggregator is None:
            # bz = 1, no aggregation
            assert features.size(1) == 1
            features = features.squeeze(1)
        else:
            features = self.second_aggregator(features)

        return features

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

            if text is None:
                return image_feat
            else:
                return self.__scaled_product(image_feat, text)



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
















    # def forward(self, image=None, text=None):

    #     # (bz, frame*atp_frames, ch, w, w)

    #     if image is None and text is not None:
    #         # text(token) -> normed text_feature
    #         return self.__encode_normed_text(text)
        
    #     else:
    #         # atp frame selection
    #         image_features = self.__encode_normed_image(image)
    #         self.atp_frames = 


    #         if self.training:
    #             # image, token -> logit
    #             text_features = self.__encode_normed_text(text)

    #             assert image_features.size(0) == text_features.size(0)

    #             if self.loss_type == "global":
    #                 image_features = torch.cat(GatherLayer.apply(image_features), 0)
    #                 text_features  = torch.cat(GatherLayer.apply(text_features), 0)

    #             return self.__scaled_product(image_features, text_features)


    #         else:
    #             # image,  text_feature -> logit
                

    #             if text is None:
    #                 return image_features

    #             else:
    #                 return self.__scaled_product(image_features, text)


