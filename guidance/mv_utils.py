import sys

from dataclasses import dataclass, field

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, CLIPTextModel
from mvdream.camera_utils import convert_opengl_to_blender, normalize_camera
from mvdream.model_zoo import build_model
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    PNDMScheduler,
    DDIMScheduler,
    StableDiffusionPipeline,
)
# import threestudio
# from threestudio.models.prompt_processors.base import PromptProcessorOutput
# from threestudio.utils.base import BaseModule
# from threestudio.utils.misc import C, cleanup, parse_version
# from threestudio.utils.typing import *


class MultiviewDiffusionGuidance(nn.Module):
    def __init__(
        self,
        device,
        model_name: str = "sd-v2.1-base-4view", # check mvdream.model_zoo.PRETRAINED_MODELS
        ckpt_path = None, # path to local checkpoint (None for loading from url)
        guidance_scale = 50.0,
        grad_clip = None,  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights = True,

    ):

        super().__init__()

        self.device = device
        self.model_name = model_name
        self.ckpt_path = ckpt_path
        self.guidance_scale = 50.0
        self.grad_clip = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        self.half_precision_weights = True

        self.camera_condition_type = "rotation"
        self.view_dependent_prompting = False

        self.n_view = 4
        self.image_size = 256
        self.recon_loss = True
        self.recon_std_rescale = 0.5
        print(f"[INFO] Loading Multiview Diffusion ...")

        self.model = build_model(self.model_name, ckpt_path=self.ckpt_path)
        for p in self.model.parameters():
            p.requires_grad_(False)

        min_step_percent = 0.02
        max_step_percent = 0.98
        self.num_train_timesteps = 1000

        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
        self.grad_clip_val: Optional[float] = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", subfolder="text_encoder"
        )
        self.scheduler = DDIMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", subfolder="scheduler",
        )
        self.to(self.device)


    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        self.text_embeddings = self.encode_text(prompts)  # [1, 77, 768]
        self.uncond_text_embeddings =  self.encode_text(negative_prompts)
    def encode_text(self, prompt):
        # prompt: [str]
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings

    @torch.no_grad()
    def get_text_embeddings(
        self,
        elevation, #: Float[Tensor, "B"],
        azimuth,#: Float[Tensor, "B"],
        camera_distances,#: Float[Tensor, "B"],
        view_dependent_prompting: bool = False,
    ): #-> Float[Tensor, "BB N Nf"]:
        batch_size = elevation.shape[0]
        # if view_dependent_prompting:
        #     pass
            # # Get direction
            # direction_idx = torch.zeros_like(elevation, dtype=torch.long)
            # for d in self.directions:
            #     direction_idx[
            #         d.condition(elevation, azimuth, camera_distances)
            #     ] = self.direction2idx[d.name]

            # # Get text embeddings
            # text_embeddings = self.text_embeddings_vd[direction_idx]  # type: ignore
            # uncond_text_embeddings = self.uncond_text_embeddings_vd[direction_idx]  # type: ignore
        # else:
        text_embeddings = self.text_embeddings.expand(batch_size, -1, -1)  # type: ignore
        uncond_text_embeddings = self.uncond_text_embeddings.expand(  # type: ignore
            batch_size, -1, -1
        )

        # IMPORTANT: we return (cond, uncond), which is in different order than other implementations!
        return torch.cat([text_embeddings, uncond_text_embeddings], dim=0)

    # @torch.no_grad()
    # def get_text_embeds(self, prompts):
    #     if text_embeddings is None:
    #         self.text_embeddings = prompt_utils.get_text_embeddings(
    #             elevation, azimuth, camera_distances, self.view_dependent_prompting
    #         )

    @torch.no_grad()
    def refine(self, 
        pred_rgb,
        elevation, 
        azimuth,
        camera_distances,
        c2w,
        fovy = None,
        text_embeddings = None,
        timestep = None,
        guidance_scale=50, 
        strength=0.6,
        steps=50, 
        **kwargs,
        ):

        # t = int(steps * strength)
        
        self.scheduler.set_timesteps(steps)
        init_step = int(steps * strength)
        batch_size = pred_rgb.shape[0]
        camera = c2w

        if text_embeddings is None:
            text_embeddings = self.get_text_embeddings(
                elevation, azimuth, camera_distances, self.view_dependent_prompting
            )


        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        latents = self.encode_images(pred_rgb_512)
        if timestep is None:
            t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=latents.device)
        else:
            assert timestep >= 0 and timestep < self.num_train_timesteps
            t = torch.full([1], timestep, dtype=torch.long, device=latents.device)
        t_expand = t.repeat(text_embeddings.shape[0])
        # latents = torch.randn((1, 4, 64, 64), device=self.device, dtype=self.dtype)

        # self.scheduler.set_timesteps(steps)
        # init_step = int(steps * strength)
        # latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[init_step])

        noise = torch.randn_like(latents)
        # print(self.scheduler.timesteps[init_step])
        # print(t)
        # print(t_expand)
        # latents_noisy = self.model.q_sample(latents, self.scheduler.timesteps[init_step], noise)
        latents_noisy = self.model.q_sample(latents, t, noise)


        # for i, t in enumerate(self.scheduler.timesteps[init_step:]):
        # for i, t in enumerate(self.scheduler.timesteps[init_step:]):
        latent_model_input = torch.cat([latents_noisy] * 2)


        if camera is not None:
            camera = self.get_camera_cond(camera, fovy)
            camera = camera.repeat(2,1).to(text_embeddings)
            context = {"context": text_embeddings, "camera": camera, "num_frames": self.n_view}
        else:
            context = {"context": text_embeddings}
        noise_pred = self.model.apply_model(latent_model_input, t_expand, context)

        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2) # Note: flipped compared to stable-dreamfusion
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

        # reconstruct x0
        latents_recon = self.model.predict_start_from_noise(latents_noisy, t, noise_pred)
        # latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            # # clip or rescale x0
            # if self.recon_std_rescale > 0:
            #     latents_recon_nocfg = self.model.predict_start_from_noise(latents_noisy, t, noise_pred_text)
            #     latents_recon_nocfg_reshape = latents_recon_nocfg.view(-1,self.n_view, *latents_recon_nocfg.shape[1:])
            #     latents_recon_reshape = latents_recon.view(-1,self.n_view, *latents_recon.shape[1:])
            #     factor = (latents_recon_nocfg_reshape.std([1,2,3,4],keepdim=True) + 1e-8) / (latents_recon_reshape.std([1,2,3,4],keepdim=True) + 1e-8)
                
            #     latents_recon_adjust = latents_recon.clone() * factor.squeeze(1).repeat_interleave(self.n_view, dim=0)
            #     latents_recon = self.recon_std_rescale * latents_recon_adjust + (1-self.recon_std_rescale) * latents_recon

            # x0-reconstruction loss from Sec 3.2 and Appendix
            # noise_pred = self.unet(
            #     latent_model_input, t, encoder_hidden_states=self.embeddings,
            # ).sample

            # noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            

        imgs = self.model.decode_first_stage(latents_recon) # [1, 3, 512, 512]
        return imgs

    def get_camera_cond(self, 
            camera, #: Float[Tensor, "B 4 4"],
            fovy = None,
    ):
        # Note: the input of threestudio is already blender coordinate system
        # camera = convert_opengl_to_blender(camera)
        if self.camera_condition_type == "rotation": # normalized camera
            camera = normalize_camera(camera)
            camera = camera.flatten(start_dim=1)
        else:
            raise NotImplementedError(f"Unknown camera_condition_type={self.camera_condition_type}")
        return camera

    def encode_images(
        self, imgs, #: Float[Tensor, "B 3 256 256"]
    ):# -> Float[Tensor, "B 4 32 32"]:
        imgs = imgs * 2.0 - 1.0
        latents = self.model.get_first_stage_encoding(self.model.encode_first_stage(imgs))
        return latents  # [B, 4, 32, 32] Latent space image

    def train_step(
        self,
        images, #: Float[Tensor, "B H W C"],
        #prompt_utils, #: PromptProcessorOutput,
        elevation, #: Float[Tensor, "B"],
        azimuth, #: Float[Tensor, "B"],
        camera_distances, #: Float[Tensor, "B"],
        c2w, #: Float[Tensor, "B 4 4"],
        images_as_latents: bool = False,
        fovy = None,
        timestep=None,
        text_embeddings=None,
        input_is_latent=False,
        **kwargs,
    ):
        batch_size = images.shape[0]
        camera = c2w

        # images_BCHW = images.permute(0, 2, 3, 1)


        if text_embeddings is None:
            text_embeddings = self.get_text_embeddings(
                elevation, azimuth, camera_distances, self.view_dependent_prompting
            )
        
        if input_is_latent:
            latents = images
        else:
            latents: Float[Tensor, "B 4 64 64"]
            if images_as_latents:
                latents = F.interpolate(input=images, size=(64, 64), mode='bilinear', align_corners=False) * 2 - 1
            else:
                # interp to 512x512 to be fed into vae.
                pred_images = F.interpolate(input=images, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
                # encode image into latents with vae, requires grad!
                latents = self.encode_images(pred_images)

        # sample timestep
        if timestep is None:
            t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=latents.device)
        else:
            assert timestep >= 0 and timestep < self.num_train_timesteps
            t = torch.full([1], timestep, dtype=torch.long, device=latents.device)
        t_expand = t.repeat(text_embeddings.shape[0])

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.model.q_sample(latents, t, noise)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            # save input tensors for UNet
            if camera is not None:
                camera = self.get_camera_cond(camera, fovy)
                camera = camera.repeat(2,1).to(text_embeddings)
                context = {"context": text_embeddings, "camera": camera, "num_frames": self.n_view}
            else:
                context = {"context": text_embeddings}
            noise_pred = self.model.apply_model(latent_model_input, t_expand, context)

        # perform guidance
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2) # Note: flipped compared to stable-dreamfusion
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

        if self.recon_loss:
            # reconstruct x0
            latents_recon = self.model.predict_start_from_noise(latents_noisy, t, noise_pred)

            # clip or rescale x0
            if self.recon_std_rescale > 0:
                latents_recon_nocfg = self.model.predict_start_from_noise(latents_noisy, t, noise_pred_text)
                latents_recon_nocfg_reshape = latents_recon_nocfg.view(-1,self.n_view, *latents_recon_nocfg.shape[1:])
                latents_recon_reshape = latents_recon.view(-1,self.n_view, *latents_recon.shape[1:])
                factor = (latents_recon_nocfg_reshape.std([1,2,3,4],keepdim=True) + 1e-8) / (latents_recon_reshape.std([1,2,3,4],keepdim=True) + 1e-8)
                
                latents_recon_adjust = latents_recon.clone() * factor.squeeze(1).repeat_interleave(self.n_view, dim=0)
                latents_recon = self.recon_std_rescale * latents_recon_adjust + (1-self.recon_std_rescale) * latents_recon

            # x0-reconstruction loss from Sec 3.2 and Appendix
            loss = 0.5 * F.mse_loss(latents, latents_recon.detach(), reduction="sum") / latents.shape[0]
            grad = torch.autograd.grad(loss, latents, retain_graph=True)[0]

        else:
            # Original SDS
            # w(t), sigma_t^2
            w = (1 - self.alphas_cumprod[t])
            grad = w * (noise_pred - noise)

            # clip grad for stable training?
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
            grad = torch.nan_to_num(grad)

            target = (latents - grad).detach()
            # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
            loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / latents.shape[0]

        return loss
        # return {
        #     "loss_sds": loss,
        #     "grad_norm": grad.norm(),
        # }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        min_step_percent = C(self.min_step_percent, epoch, global_step)
        max_step_percent = C(self.max_step_percent, epoch, global_step)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)