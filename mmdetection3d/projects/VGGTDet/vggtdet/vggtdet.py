from typing import List, Tuple, Union

import torch
import torch.nn as nn
from contextlib import nullcontext

from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType, OptConfigType
from projects.VGGTDet.detr3_models.transformer import (TransformerDecoder, TransformerDecoder_Multilevel,
                                                       TransformerDecoderLayer)
from vggt.models.vggt import VGGT


try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


def npu_is_available():
    return hasattr(torch, 'npu') and torch.npu.is_available()


def get_vggt_device():
    if npu_is_available():
        return torch.device('npu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def get_vggt_dtype(device):
    if device.type == 'cuda':
        return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    if device.type == 'npu':
        return torch.float16
    return torch.float32


def vggt_autocast(device, dtype, enabled=True):
    if not enabled or device.type == 'cpu':
        return nullcontext()
    if device.type == 'npu' and hasattr(torch, 'npu') and hasattr(torch.npu, 'amp'):
        return torch.npu.amp.autocast(dtype=dtype)
    if device.type == 'cuda':
        return torch.cuda.amp.autocast(dtype=dtype)
    if hasattr(torch, 'autocast'):
        return torch.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()


class ChannelProjecter(nn.Module):
    def __init__(self, in_channels=2048, out_channels=256):
        super().__init__()
        
        self.proj = nn.Sequential(

            nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels//2,
                    kernel_size=1,
                    stride=1,
                    padding=0
                            ),
            nn.GroupNorm(num_groups=1, num_channels=in_channels//2),
            nn.GELU(),

            nn.Conv2d(
                    in_channels=in_channels//2,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                            )
        )
        
        self.res = nn.Sequential(
            nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                            )
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.proj(x) + self.res(x)
        del x
        return res   # [B, D, N, T]
    
@MODELS.register_module()
class VGGTDet(Base3DDetector):
    def __init__(
            self,
            backbone: ConfigType,
            neck: ConfigType,
            neck_3d: ConfigType,
            bbox_head: ConfigType,
            prior_generator: ConfigType,
            n_voxels: List,
            voxel_size: List,
            head_2d: ConfigType = None,
            train_cfg: OptConfigType = None,
            test_cfg: OptConfigType = None,
            data_preprocessor: OptConfigType = None,
            init_cfg: OptConfigType = None,
            #  pretrained,
            aabb: Tuple = None,
            near_far_range: List = None,
            N_samples: int = 64,
            N_rand: int = 2048,
            depth_supervise: bool = False,
            use_nerf_mask: bool = True,
            nerf_sample_view: int = 3,
            nerf_mode: str = 'volume',
            squeeze_scale: int = 4,
            rgb_supervision: bool = True,
            nerf_density: bool = False,
            render_testing: bool = False,
            gs_cfg=None,
            vis_dir=None,
            visualize_bbox=False,
            topk=3,
            alpha_thres=None,
            sigma_w=0.5,
            vggt_pretrained="/home/Newdisk1/lvxueqiang/DuDet/mmdetection3d/pretrain/VGGT-1B",
            decoder_cfg: OptConfigType = None,
            if_learnable_query=True,
            num_queries=128,
            token_dim=1024,
            test_only_last_layer=True,
            if_use_gt_query=False,
            position_embedding="fourier",
            if_mix_precision=False,
            if_save_vggt_feature=False,
            use_multi_layers=False,
            if_simpler_project=False,
            if_use_pred_pc_query=False,
            if_use_atten_sample=False,
            atten_sample_ratio=10,
            depth_thres=1000,
            if_use_atten_fps=False,
            lambda_dist=1.0,
            if_task_query=False,
            if_add_noises=False,
            noise_level=None
    ):

        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)

        self.vggt_device = get_vggt_device()
        self.vggt_dtype = get_vggt_dtype(self.vggt_device)
        self.vggt_encoder = VGGT.from_pretrained(vggt_pretrained).to(self.vggt_device)

        for param in self.vggt_encoder.parameters():
            param.requires_grad = False

        self.vggt_encoder.eval()

        self.decoder = build_decoder(decoder_cfg, if_multilevel=use_multi_layers)

        # self.proj_feat_dim = nn.Conv2d(
        #             in_channels=2048,
        #             out_channels=token_dim,
        #             kernel_size=1,
        #             stride=1,
        #             padding=0
        #         )
        if if_simpler_project:
            if use_multi_layers:
                self.proj_feat_dim0 = nn.Conv2d(
                    in_channels=2048,
                    out_channels=token_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
                self.proj_feat_dim1 = nn.Conv2d(
                    in_channels=2048,
                    out_channels=token_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
                self.proj_feat_dim2 = nn.Conv2d(
                    in_channels=2048,
                    out_channels=token_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
                self.proj_feat_dim3 = nn.Conv2d(
                    in_channels=2048,
                    out_channels=token_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
                # self.proj_feat_dim4 = nn.Conv2d(
                #     in_channels=2048,
                #     out_channels=token_dim,
                #     kernel_size=1,
                #     stride=1,
                #     padding=0
                # )
            else:
                self.proj_feat_dim = nn.Conv2d(
                    in_channels=2048,
                    out_channels=token_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
        else:
            if use_multi_layers:
                self.proj_feat_dim0 = ChannelProjecter(in_channels=2048, out_channels=token_dim)  # for _ in range(4)]
                self.proj_feat_dim1 = ChannelProjecter(in_channels=2048, out_channels=token_dim)
                self.proj_feat_dim2 = ChannelProjecter(in_channels=2048, out_channels=token_dim)
                self.proj_feat_dim3 = ChannelProjecter(in_channels=2048, out_channels=token_dim)
                # self.proj_feat_dim4 = ChannelProjecter(in_channels=2048, out_channels=token_dim)
            else:
                self.proj_feat_dim = ChannelProjecter(in_channels=2048, out_channels=token_dim)

        self.prior_generator = TASK_UTILS.build(prior_generator)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # self.proj_norm = nn.LayerNorm(token_dim)
        self.num_queries = num_queries
        self.if_learnable_query = if_learnable_query

        if if_learnable_query:
            self.queries = nn.Parameter(torch.Tensor(num_queries, token_dim))
            nn.init.xavier_normal_(self.queries)
        ######### idea 2 ############
        self.if_task_query = if_task_query
        if if_task_query:
            self.task_query = nn.Parameter(torch.Tensor(1, token_dim))
            nn.init.xavier_normal_(self.task_query)
        ######### idea 2 ############
        self.test_only_last_layer = test_only_last_layer

        self.if_use_gt_query = if_use_gt_query
        # assert if_learnable_query is not self.if_use_gt_query

        self.if_use_pred_pc_query = if_use_pred_pc_query
        # assert
        assert (self.if_use_pred_pc_query + self.if_use_gt_query + self.if_learnable_query) == 1, \
            "Only one of 'if_use_pred_pc_query', 'if_use_gt_query', or 'if_learnable_query' must be True."

        self.if_mix_precision = if_mix_precision
        self.if_save_vggt_feature = if_save_vggt_feature

        self.use_multi_layers = use_multi_layers
        self.if_use_atten_sample = if_use_atten_sample
        self.atten_sample_ratio = atten_sample_ratio
        self.depth_thres = depth_thres
        self.if_use_atten_fps = if_use_atten_fps
        self.lambda_dist = lambda_dist
        self.if_add_noises = if_add_noises
        self.noise_level = noise_level

    def maybe_autocast(self, enabled=True):
        return vggt_autocast(self.vggt_device, self.vggt_dtype, enabled=enabled)

    @torch.no_grad()
    def extract_feat(self, batch_inputs_dict: dict,
                     batch_data_samples: SampleList, mode):

        if self.vggt_encoder.training:
            for param in self.vggt_encoder.parameters():
                param.requires_grad = False

            self.vggt_encoder.eval()

        with torch.no_grad():
            with self.maybe_autocast():
                img = batch_inputs_dict['imgs'] # (bs, 40, 3, 392, 518)
                img = img.to(self.vggt_device).float()
                if self.if_use_atten_sample or self.if_use_atten_fps:
                    aggregated_tokens_list, ps_idx, images_patch_attn = self.vggt_encoder.aggregator(img, if_norm=False, if_detach=True, 
                                                                                                     if_only_last_layer=(not self.use_multi_layers), 
                                                                                                     if_use_atten_sample=True, 
                                                                                                     if_task_query=self.if_task_query) # if_norm=False because we have norm it in the data layer
                    return aggregated_tokens_list, ps_idx, img, images_patch_attn
                else:
                    aggregated_tokens_list, ps_idx = self.vggt_encoder.aggregator(img, if_norm=False, 
                                                                                  if_detach=True, 
                                                                                  if_only_last_layer=(not self.use_multi_layers), 
                                                                                  if_use_atten_sample=False,
                                                                                  if_task_query=self.if_task_query) 
                    return aggregated_tokens_list, ps_idx, img, None


    @torch.no_grad()
    def batch_random_sample(self, points, k=100000, depth_mask=None, weights=None):
        B, N, _ = points.shape
        device = points.device
        
        rand_values = torch.rand(B, N, device=device)
        if depth_mask is not None:
            rand_values[depth_mask] = 0

        perm = torch.argsort(rand_values, dim=-1, descending=True)
        
        indices = perm[:, :k]
        
        batch_indices = torch.arange(B, device=device)[:, None]

        if weights is not None:
            return points[batch_indices, indices], weights[batch_indices, indices] 
        else:
            return points[batch_indices, indices]

    def get_box_features(self, vggt_token_list, ps_idx, batch_inputs_dict, images, images_patch_attn):

        if self.use_multi_layers:
            x = []
            for idx_layer, tokens in enumerate(vggt_token_list):
                tokens_permute = tokens.permute(0, 3, 1, 2).contiguous()  
                patch_tokens = tokens_permute[:, :, :, ps_idx:]
                # patch_tokens_list.append(patch_tokens)
                if idx_layer == 0:
                    patch_tokens_projected = self.proj_feat_dim0(patch_tokens)
                elif idx_layer == 1:
                    patch_tokens_projected = self.proj_feat_dim1(patch_tokens)
                elif idx_layer == 2:
                    patch_tokens_projected = self.proj_feat_dim2(patch_tokens)
                elif idx_layer == 3:
                    patch_tokens_projected = self.proj_feat_dim3(patch_tokens)
                elif idx_layer == 4:
                    patch_tokens_projected = self.proj_feat_dim4(patch_tokens)
                # if not self.if_use_pred_pc_query:
                del patch_tokens

                batch_size, feat_dim, im_num, token_num = patch_tokens_projected.shape
                patch_tokens_projected = patch_tokens_projected.reshape(batch_size, feat_dim, -1)
                patch_tokens_projected = patch_tokens_projected.permute(2, 0, 1).contiguous() 
                x.append(patch_tokens_projected)

            if not self.if_use_pred_pc_query:
                del vggt_token_list

        else:
            tokens_last_layer = vggt_token_list[-1]
            patch_tokens_last_layer = tokens_last_layer[:, :, ps_idx:, :]  
            x = patch_tokens_last_layer.permute(0, 3, 1, 2).contiguous()
            x = self.proj_feat_dim(x)
            batch_size, feat_dim, im_num, token_num = x.shape
            x = x.reshape(batch_size, feat_dim, -1)
            x = x.permute(2, 0, 1).contiguous()

        tgt = self.queries.unsqueeze(1).expand(-1, batch_size, -1)  # [num_queries, batch_size, token_dim]

        box_features = self.decoder(tgt, x, query_pos=None, pos=None)[0]

        return box_features

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> Union[dict, list]:


        vggt_token_list, ps_idx, img, images_patch_attn = self.extract_feat(batch_inputs_dict, batch_data_samples, 'train')

        if self.if_mix_precision:
            with self.maybe_autocast():
                box_features = self.get_box_features(vggt_token_list, ps_idx, batch_inputs_dict, img, images_patch_attn)
        else: 
            box_features = self.get_box_features(vggt_token_list, ps_idx, batch_inputs_dict, img, images_patch_attn)

        losses = self.bbox_head.loss(box_features, batch_data_samples, batch_inputs_dict, **kwargs) 
        return losses


    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:

        vggt_token_list, ps_idx, img, images_patch_attn = self.extract_feat(batch_inputs_dict, batch_data_samples, 'train')

        if self.if_mix_precision:
            with self.maybe_autocast():
                box_features = self.get_box_features(vggt_token_list, ps_idx, batch_inputs_dict, img, images_patch_attn)
        else:
            box_features = self.get_box_features(vggt_token_list, ps_idx, batch_inputs_dict, img, images_patch_attn)

        if self.test_only_last_layer:
            box_features = [box_features[-1]]

        results_list = self.bbox_head.predict(box_features, batch_data_samples, batch_inputs_dict, **kwargs)
        # results_list[0]['labels_3d'] = torch.ones_like(results_list[0]['labels_3d']) * 2
        predictions = self.add_pred_to_datasample(batch_data_samples,
                                                  results_list)
        return predictions


    def _forward(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                 *args, **kwargs) -> Tuple[List[torch.Tensor]]:
        vggt_token_list, ps_idx, img, images_patch_attn = self.extract_feat(batch_inputs_dict, batch_data_samples, 'train')

        if self.if_mix_precision:
            with self.maybe_autocast():
                box_features = self.get_box_features(vggt_token_list, ps_idx, batch_inputs_dict, img, images_patch_attn)
        else:
            box_features = self.get_box_features(vggt_token_list, ps_idx, batch_inputs_dict, img, images_patch_attn)

        if self.test_only_last_layer:
            box_features = [box_features[-1]]

        results = self.bbox_head.forward(box_features, batch_inputs_dict)
        return results

def build_decoder(args, if_multilevel=False):
    decoder_layer = TransformerDecoderLayer(
        d_model=args.dec_dim,
        nhead=args.dec_nhead,
        dim_feedforward=args.dec_ffn_dim,
        dropout=args.dec_dropout,
    )

    if if_multilevel:
         decoder = TransformerDecoder_Multilevel(
            decoder_layer, num_layers=args.dec_nlayers, return_intermediate=True
        )       
    else:
        decoder = TransformerDecoder(
            decoder_layer, num_layers=args.dec_nlayers, return_intermediate=True
        )
    return decoder
