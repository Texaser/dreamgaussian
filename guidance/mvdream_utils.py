import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mvdream.camera_utils import get_camera, convert_opengl_to_blender, normalize_camera
from mvdream.model_zoo import build_model
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from .perpneg_utils import weighted_perpendicular_aggregator
from diffusers import DDIMScheduler

class MVDream(nn.Module):
    def __init__(
        self,
        device,
        model_name='sd-v2.1-base-4view',
        ckpt_path=None,
        t_range=[0.02, 0.98],
    ):
        super().__init__()

        self.device = device
        self.model_name = model_name
        self.ckpt_path = ckpt_path

        self.model = build_model(self.model_name, ckpt_path=self.ckpt_path).eval().to(self.device)
        self.model.device = device
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.dtype = torch.float32

        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])

        self.time_prior = [800, 500, 300, 100]
        m1, m2, s1, s2 = self.time_prior
        weights = torch.cat(
            (
                torch.exp(
                    -(torch.arange(self.num_train_timesteps, m1, -1) - m1)
                        / (2 * s1)
                    ),
                torch.ones(m1 - m2 + 1),
                torch.exp(
                        -(torch.arange(m2 - 1, 0, -1) - m2) / (2 * s2)
                    ),
            )
        )
        weights = weights / torch.sum(weights)
        self.time_prior_acc_weights = torch.cumsum(weights, dim=0)
        self.iters = 1000
        self.t_choice = self.t_choice_nonlinear(self.iters)

        self.embeddings = None
        self.recon_loss = False
        self.recon_std_rescale = 0.5
        self.n_view = 4
        self.scheduler = DDIMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", subfolder="scheduler", torch_dtype=self.dtype
        )

    @torch.no_grad()
    def get_text_embed(self, prompts, negative_prompts):
        pos_embeds = self.encode_text(prompts).repeat(4,1,1)  # [1, 77, 768]
        neg_embeds = self.encode_text(negative_prompts).repeat(4,1,1)
        self.embeddings = torch.cat([neg_embeds, pos_embeds], dim=0)  # [2, 77, 768]
        return self.embeddings
    def encode_text(self, prompt):
        # prompt: [str]
        embeddings = self.model.get_learned_conditioning(prompt).to(self.device)
        return embeddings

    @torch.no_grad()
    def get_text_embeds(self, prompts):
        embeddings = self.encode_text(prompts)
        return embeddings
    
    @torch.no_grad()
    def refine(self, pred_rgb, text_embeddings, weights, camera,
               guidance_scale=100, steps=50, strength=0.8,
        ):

        batch_size = pred_rgb.shape[0]
        pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)
        latents = self.encode_imgs(pred_rgb_256.to(self.dtype))
        # latents = torch.randn((1, 4, 64, 64), device=self.device, dtype=self.dtype)

        self.scheduler.set_timesteps(steps)
        init_step = int(steps * strength)
        latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[init_step])

        camera = camera[:, [0, 2, 1, 3]] # to blender convention (flip y & z axis)
        camera[:, 1] *= -1
        camera = normalize_camera(camera).view(batch_size, 16)
        camera = camera.repeat(2, 1)
        context = {"context": self.embeddings, "camera": camera, "num_frames": 4}

        for i, t in enumerate(self.scheduler.timesteps[init_step:]):
    
            latent_model_input = torch.cat([latents] * 2)
            
            tt = torch.cat([t.unsqueeze(0).repeat(batch_size)] * 2).to(self.device)

            noise_pred = self.model.apply_model(latent_model_input, tt, context)

            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        imgs = self.decode_latents(latents) # [1, 3, 512, 512]
        return imgs

    def train_step(
        self,
        pred_rgb, # [B, C, H, W], B is multiples of 4
        camera, # [B, 4, 4]
        step=None,
        step_ratio=None,
        guidance_scale=50,
        as_latent=False,
    ):
        
        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)

        if as_latent:
            latents = F.interpolate(pred_rgb, (32, 32), mode="bilinear", align_corners=False) * 2 - 1
        else:
            # interp to 256x256 to be fed into vae.
            pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode="bilinear", align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_256)

        if step_ratio is not None:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
            # t = self.t_choice[step - 1]
            # print(t)
            t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)

        # camera = convert_opengl_to_blender(camera)
        # flip_yz = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).unsqueeze(0)
        # camera = torch.matmul(flip_yz.to(camera), camera)
        camera = camera[:, [0, 2, 1, 3]] # to blender convention (flip y & z axis)
        camera[:, 1] *= -1
        camera = normalize_camera(camera).view(batch_size, 16)

        ###############
        # sampler = DDIMSampler(self.model)
        # shape = [4, 32, 32]
        # c_ = {"context": self.embeddings[4:]}
        # uc_ = {"context": self.embeddings[:4]}

        # # print(camera)

        # # camera = get_camera(4, elevation=0, azimuth_start=0)
        # # camera = camera.repeat(batch_size // 4, 1).to(self.device)

        # # print(camera)

        # c_["camera"] = uc_["camera"] = camera
        # c_["num_frames"] = uc_["num_frames"] = 4

        # latents_, _ = sampler.sample(S=30, conditioning=c_,
        #                                 batch_size=batch_size, shape=shape,
        #                                 verbose=False, 
        #                                 unconditional_guidance_scale=guidance_scale,
        #                                 unconditional_conditioning=uc_,
        #                                 eta=0, x_T=None)

        # # Img latents -> imgs
        # imgs = self.decode_latents(latents_)  # [4, 3, 256, 256]
        # import kiui
        # kiui.vis.plot_image(imgs)
        ###############

        camera = camera.repeat(2, 1)
        context = {"context": self.embeddings, "camera": camera, "num_frames": 4}

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.model.q_sample(latents, t, noise)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)

            # import kiui
            # kiui.lo(latent_model_input, t, context['context'], context['camera'])
            
            noise_pred = self.model.apply_model(latent_model_input, tt, context)
            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)

            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
            # noise_pred_text, noise_pred_uncond = noise_pred.chunk(2) # Note: flipped compared to stable-dreamfusion
            # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        if self.recon_loss:
            # reconstruct x0
            latents_recon = self.model.predict_start_from_noise(latents_noisy, t, noise_pred)

            # clip or rescale x0
            if self.recon_std_rescale > 0:
                latents_recon_nocfg = self.model.predict_start_from_noise(latents_noisy, t, noise_pred_pos)
                latents_recon_nocfg_reshape = latents_recon_nocfg.view(-1,self.n_view, *latents_recon_nocfg.shape[1:])
                latents_recon_reshape = latents_recon.view(-1,self.n_view, *latents_recon.shape[1:])
                factor = (latents_recon_nocfg_reshape.std([1,2,3,4],keepdim=True) + 1e-8) / (latents_recon_reshape.std([1,2,3,4],keepdim=True) + 1e-8)
                
                latents_recon_adjust = latents_recon.clone() * factor.squeeze(1).repeat_interleave(self.n_view, dim=0)
                latents_recon = self.recon_std_rescale * latents_recon_adjust + (1-self.recon_std_rescale) * latents_recon

            # x0-reconstruction loss from Sec 3.2 and Appendix
            loss = 0.5 * F.mse_loss(latents, latents_recon.detach(), reduction="sum") / latents.shape[0]
            # grad = torch.autograd.grad(loss, latents, retain_graph=True)[0]
        
        # grad = (noise_pred - noise)
        # grad = torch.nan_to_num(grad)

        # # seems important to avoid NaN...
        # # grad = grad.clamp(-1, 1)

        # target = (latents - grad).detach()
        # loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[0]

        return loss

    def train_step_perpneg(
        self,
        pred_rgb, # [B, C, H, W], B is multiples of 4
        text_embeddings,
        weights,
        camera, # [B, 4, 4]
        step=None,
        step_ratio=None,
        guidance_scale=50,
        as_latent=False,
    ):
        
        batch_size = pred_rgb.shape[0]
        K = (text_embeddings.shape[0] // batch_size) - 1
        pred_rgb = pred_rgb.to(self.dtype)

        if as_latent:
            latents = F.interpolate(pred_rgb, (32, 32), mode="bilinear", align_corners=False) * 2 - 1
        else:
            # interp to 256x256 to be fed into vae.
            pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode="bilinear", align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_256)

        if step_ratio is not None:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
            # t = self.t_choice[step - 1]
            # print(t)
            t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)

        # camera = convert_opengl_to_blender(camera)
        # flip_yz = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).unsqueeze(0)
        # camera = torch.matmul(flip_yz.to(camera), camera)
        camera = camera[:, [0, 2, 1, 3]] # to blender convention (flip y & z axis)
        camera[:, 1] *= -1
        camera = normalize_camera(camera).view(batch_size, 16)

        camera = camera.repeat((1 + K), 1)
        context = {"context": text_embeddings, "camera": camera, "num_frames": 4}

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.model.q_sample(latents, t, noise)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * (1 + K))
            tt = torch.cat([t] * (1 + K))

            # import kiui
            # kiui.lo(latent_model_input, t, context['context'], context['camera'])
            # print(latent_model_input.shape)
            noise_pred = self.model.apply_model(latent_model_input, tt, context)
            # print(noise_pred.shape)
            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_pos = noise_pred[:batch_size], noise_pred[batch_size:]
            # print(noise_pred_uncond.shape)
            # print(noise_pred_pos.shape)

            delta_noise_preds = noise_pred_pos - noise_pred_uncond.repeat(K, 1, 1, 1)
            # print(delta_noise_preds.shape)
            noise_pred = noise_pred_uncond + guidance_scale * weighted_perpendicular_aggregator(delta_noise_preds, weights, batch_size)
            # print(noise_pred.shape)
            # noise_pred_text, noise_pred_uncond = noise_pred.chunk(2) # Note: flipped compared to stable-dreamfusion
            # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        if self.recon_loss:
            # reconstruct x0
            latents_recon = self.model.predict_start_from_noise(latents_noisy, t, noise_pred)
            # print(latents_recon.shape)
            # clip or rescale x0
            if self.recon_std_rescale > 0:
                latents_recon_nocfg = self.model.predict_start_from_noise(latents_noisy, t, noise_pred_pos)
                latents_recon_nocfg_reshape = latents_recon_nocfg.view(-1,self.n_view, *latents_recon_nocfg.shape[1:])
                latents_recon_reshape = latents_recon.view(-1,self.n_view, *latents_recon.shape[1:])
                factor = (latents_recon_nocfg_reshape.std([1,2,3,4],keepdim=True) + 1e-8) / (latents_recon_reshape.std([1,2,3,4],keepdim=True) + 1e-8)
                
                latents_recon_adjust = latents_recon.clone() * factor.squeeze(1).repeat_interleave(self.n_view, dim=0)
                latents_recon = self.recon_std_rescale * latents_recon_adjust + (1-self.recon_std_rescale) * latents_recon

            # x0-reconstruction loss from Sec 3.2 and Appendix
            loss = 0.5 * F.mse_loss(latents, latents_recon.detach(), reduction="sum") / latents.shape[0]
            # grad = torch.autograd.grad(loss, latents, retain_graph=True)[0]
        
        grad = (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        # seems important to avoid NaN...
        # grad = grad.clamp(-1, 1)

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[0]

        return loss

    def decode_latents(self, latents):
        imgs = self.model.decode_first_stage(latents)
        imgs = ((imgs + 1) / 2).clamp(0, 1)
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, 256, 256]
        imgs = 2 * imgs - 1
        latents = self.model.get_first_stage_encoding(self.model.encode_first_stage(imgs))
        return latents # [B, 4, 32, 32]

    @torch.no_grad()
    def prompt_to_img(
        self,
        prompts,
        negative_prompts="",
        height=256,
        width=256,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
        elevation=0,
        azimuth_start=0,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]
        
        batch_size = len(prompts) * 4

        # Text embeds -> img latents
        sampler = DDIMSampler(self.model)
        shape = [4, height // 8, width // 8]
        c_ = {"context": self.encode_text(prompts).repeat(4,1,1)}
        uc_ = {"context": self.encode_text(negative_prompts).repeat(4,1,1)}

        camera = get_camera(4, elevation=elevation, azimuth_start=azimuth_start)
        camera = camera.repeat(batch_size // 4, 1).to(self.device)

        c_["camera"] = uc_["camera"] = camera
        c_["num_frames"] = uc_["num_frames"] = 4

        latents, _ = sampler.sample(S=num_inference_steps, conditioning=c_,
                                        batch_size=batch_size, shape=shape,
                                        verbose=False, 
                                        unconditional_guidance_scale=guidance_scale,
                                        unconditional_conditioning=uc_,
                                        eta=0, x_T=None)

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [4, 3, 256, 256]
        
        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")

        return imgs

    def t_choice_nonlinear(self, max_step: int):
        t_choice = []
        for i in range(0, max_step):
            current_step_ratio = i / max_step
            time_index = torch.where(
                        (self.time_prior_acc_weights - current_step_ratio) > 0
                    )[0][0]
            if time_index == 0 or torch.abs(
                self.time_prior_acc_weights[time_index] - current_step_ratio
            ) < torch.abs(
                self.time_prior_acc_weights[time_index - 1] - current_step_ratio
            ):
                t = self.num_train_timesteps - time_index
            else:
                t = self.num_train_timesteps - time_index + 1
            t = torch.clip(t, self.min_step, self.max_step + 1)
            # t = torch.full((1,), t, dtype=torch.long, device=self.device)
            t_choice.append(t)
        return t_choice

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--negative", default="", type=str)
    parser.add_argument("--steps", type=int, default=30)
    opt = parser.parse_args()

    device = torch.device("cuda")

    sd = MVDream(device)

    while True:
        imgs = sd.prompt_to_img(opt.prompt, opt.negative, num_inference_steps=opt.steps)

        grid = np.concatenate([
            np.concatenate([imgs[0], imgs[1]], axis=1),
            np.concatenate([imgs[2], imgs[3]], axis=1),
        ], axis=0)

        # visualize image
        plt.imshow(grid)
        plt.show()
