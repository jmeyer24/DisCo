
import os
import sys
import math
import json
import time
import random
import shutil
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from packaging import version
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision import transforms
import numpy as np

import transformers
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, PNDMScheduler

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder


from model.fusion import ObjectFusionTokenizer
from model.cond_vae import SceneVAEModel
from model.attention import register_attention_control
from data import build_train_dataloader
from loss import VaeGaussCriterion, BoxL1Criterion

#accelerate launch train_disco.py --use_ema --resolution=512 --batch_size=1 --gradient_accumulation_steps=4 --gradient_checkpointing --max_train_steps=50000 --learning_rate=1e-05  --lr_scheduler="linear" --checkpointing_steps 5000
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_diffusion_model_path", type=str, default='./StableDiffusion/', help="Path to pretrained model or model identifier from huggingface.co/models.",)
    parser.add_argument('--data_dir', type=str, default='./VisualGenome/', help='path to training dataset')
    parser.add_argument('--output_dir', type=str, default="./results", help='path to save checkpoint')
    parser.add_argument("--logging_dir", type=str, default="logs", help="TensorBoard log directory.")

    parser.add_argument('--dataloader_num_workers', type=int, default=8, help='num_workers')
    parser.add_argument('--dataloader_shuffle', type=bool, default=True, help='shuffle')
    parser.add_argument("--tracker_project_name", type=str, default="text2image", help="The `project_name` passed to Accelerator",)
    parser.add_argument('--resolution', type=int, default=512, help='resolution')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument("--num_train_epochs", type=int, default=200)
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",)
    parser.add_argument("--checkpointing_steps", type=int, default=5000, help="Save a checkpoint of the training state every X updates.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--allow_tf32", action="store_true", help="Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16)",)
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")

    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]')
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    
    parser.add_argument("--vae_loss_weight", type=float, default=0.1, help="")
    parser.add_argument("--box_loss_weight", type=float, default=1, help="")
    parser.add_argument("--diff_loss_weight", type=float, default=1.0, help="")
    parser.add_argument('--embedding_dim', type=int, default=64, help='')
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    timestamp = time.strftime("%Y%m%d-%Hh%Mm%Ss", time.localtime())
    args.output_dir = os.path.join(args.output_dir, 'train', f'{args.tracker_project_name}-{timestamp}') 
    return args

## accelerate launch train_image_diffusion.py  --mixed_precision="fp16"   --use_ema --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=4 --gradient_checkpointing --max_train_steps=5000 --learning_rate=1e-05  --lr_scheduler="constant"
class Trainer:
    def __init__(self, args):
        # Init settings
        self.args = args
        self.logger = get_logger(__name__, log_level="INFO")
        logging_dir = os.path.join(args.output_dir, args.logging_dir)
        accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with="tensorboard",
            project_config=accelerator_project_config,
        )

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        self.logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if args.seed is not None:
            set_seed(args.seed)

        # Enable TF32 for faster training on Ampere GPUs,
        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        # Handle the repository creation
        if self.accelerator.is_main_process:
            if args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)
                with open(f'{args.output_dir}/config.json', 'wt') as f:
                    json.dump(vars(args), f, indent=4)

        # Load scheduler, tokenizer and models.
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_diffusion_model_path, 
            subfolder="scheduler"
        )
        self.scheduler = PNDMScheduler.from_pretrained(
            args.pretrained_diffusion_model_path, 
            subfolder="scheduler"
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_diffusion_model_path, 
            subfolder="tokenizer"
        )

        def deepspeed_zero_init_disabled_context_manager():
            """
            returns either a context list that includes one that will disable zero.Init or an empty context list
            """
            deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
            if deepspeed_plugin is None:
                return []
            return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            self.text_encoder = CLIPTextModel.from_pretrained(
                args.pretrained_diffusion_model_path, 
                subfolder="text_encoder", 
            )
            self.vae = AutoencoderKL.from_pretrained(
                args.pretrained_diffusion_model_path, 
                subfolder="vae",
            )
        
        self.unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_diffusion_model_path, 
            subfolder="unet"
        )
        register_attention_control(self.unet)

        # Data
        self.train_dataloader, self.val_dataloader, self.vocab = build_train_dataloader(args, tokenizer=self.tokenizer)

        num_objs = len(self.vocab['object_idx_to_name'])
        num_rels = len(self.vocab['pred_idx_to_name'])
        self.sl_vae = SceneVAEModel(self.args, num_objs, num_rels)
        self.object_fusion_tokenizer = ObjectFusionTokenizer()

        # Freeze vae and text_encoder and set unet to trainable
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.train()
        self.sl_vae.train()
        self.object_fusion_tokenizer.train()

        # Criterion
        self.vae_criterion = VaeGaussCriterion()
        self.box_criterion = BoxL1Criterion()

        # Create EMA for the unet.
        if args.use_ema:
            self.ema_unet = UNet2DConditionModel.from_pretrained(
                args.pretrained_diffusion_model_path, 
                subfolder="unet"
            )
            register_attention_control(self.ema_unet)
            self.ema_unet = EMAModel(self.ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=self.ema_unet.config)
            self.ema_temp = UNet2DConditionModel.from_pretrained(
                args.pretrained_diffusion_model_path, 
                subfolder="unet"
            )

        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
            def save_model_hook(models, weights, output_dir):
                if self.accelerator.is_main_process:
                    if args.use_ema:
                        self.ema_unet.store(self.unet.parameters())
                        self.ema_unet.copy_to(self.unet.parameters())
                        self.unet.save_pretrained(os.path.join(output_dir, "unet_ema"))
                        self.ema_unet.restore(self.unet.parameters())

                    for i, model in enumerate(models):
                        try:
                            model.save_pretrained(os.path.join(output_dir, "unet"))
                        except:
                            if model.__class__.__name__ == "SceneVAEModel":
                                torch.save(model.state_dict(), os.path.join(output_dir, "sl_vae.pt"))
                            elif model.__class__.__name__ == "ObjectFusionTokenizer":
                                torch.save(model.state_dict(), os.path.join(output_dir, "fusion.pt"))

                        # make sure to pop weight so that corresponding model is not saved again
                        weights.pop()

            def load_model_hook(models, input_dir):
                if args.use_ema:
                    load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                    self.ema_unet.load_state_dict(load_model.state_dict())
                    self.ema_unet.to(self.accelerator.device)
                    del load_model

                for _ in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()

                    # load diffusers style into model
                    load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model
            self.accelerator.register_save_state_pre_hook(save_model_hook)
            self.accelerator.register_load_state_pre_hook(load_model_hook)

        if args.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        # Enable TF32 for faster training on Ampere GPUs,
        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True


        params=[]
        params_multi=[]
        for name, module in self.unet.named_modules():
            if any(child for child in module.children()):
                continue
            split_name = name.split('.')
            if 'processor' in split_name:
                params_multi.append(dict(params=module.parameters(), lr=args.learning_rate * 10))
            else:
                params.append(dict(params=module.parameters(), lr=args.learning_rate))
        params_multi.append(dict(params=self.object_fusion_tokenizer.parameters(), lr=args.learning_rate * 10))
        params_multi.append(dict(params=self.sl_vae.parameters(), lr=args.learning_rate * 10))

        # Initialize the optimizer
        self.optimizer = torch.optim.AdamW(
            params,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        self.optimizer_multi = torch.optim.AdamW(
            params_multi,
            lr=args.learning_rate*10,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        self.lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=args.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=args.max_train_steps * self.accelerator.num_processes,
        )

        self.lr_scheduler_multi = get_scheduler(
            args.lr_scheduler,
            optimizer=self.optimizer_multi,
            num_warmup_steps=args.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=args.max_train_steps * self.accelerator.num_processes,
        )

        # Prepare everything with our `accelerator`.
        self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler, self.lr_scheduler_multi, self.sl_vae, self.object_fusion_tokenizer = self.accelerator.prepare(
            self.unet,
            self.optimizer, 
            self.train_dataloader, 
            self.lr_scheduler,
            self.lr_scheduler_multi,
            self.sl_vae,
            self.object_fusion_tokenizer
        )

        if args.use_ema:
            self.ema_unet.to(self.accelerator.device)

        # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
            args.mixed_precision = self.accelerator.mixed_precision
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16
            args.mixed_precision = self.accelerator.mixed_precision
        
        # Move text_encode and vae to gpu and cast to weight_dtype
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        self.progress_bar = tqdm(
            range(0, self.args.max_train_steps),
            initial=0,
            desc="Steps",
            disable=not self.accelerator.is_local_main_process,
        )
        
        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.accelerator.is_main_process:
            tracker_config = dict(vars(args))
            self.accelerator.init_trackers(args.tracker_project_name, tracker_config)

    def start(self):
        # Print information
        self.logger.info('  Global configuration as follows:')
        for key, val in vars(self.args).items():
            self.logger.info("  {:28} {}".format(key, val))
        
        # Start to training
        total_batch_size = self.args.batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps
        self.logger.info("\n")
        self.logger.info(f"  Running training:")
        self.logger.info(f"  Num Iterations = {len(self.train_dataloader)}")
        self.logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.args.batch_size}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
        
        self.train()

        # Create the pipeline using the trained modules and save it.
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.unet = self.accelerator.unwrap_model(self.unet)
            if self.args.use_ema:
                self.ema_unet.copy_to(self.unet.parameters())
            
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.args.pretrained_diffusion_model_path,
                text_encoder=self.text_encoder,
                vae=self.vae,
                unet=self.unet
            )

            pipeline.save_pretrained(self.args.output_dir)
            self.sl_vae = self.accelerator.unwrap_model(self.sl_vae)
            self.object_fusion_tokenizer = self.accelerator.unwrap_model(self.object_fusion_tokenizer)
            torch.save(self.sl_vae.state_dict(), os.path.join(self.args.output_dir, "sl_vae.pt"))
            torch.save(self.object_fusion_tokenizer.state_dict(), os.path.join(self.args.output_dir, "fusion.pt"))

        self.accelerator.end_training()
    
    def train(self):
        self.global_step = 0
        for epoch in range(0, self.args.num_train_epochs):
            self.train_one_epoch(epoch)

    def train_one_epoch(self, epoch):
        log_loss = 0.0
        log_box_loss = 0.0
        log_vae_loss = 0.0
        log_diff_loss = 0.0
        for step, batch in enumerate(self.train_dataloader):
            with self.accelerator.accumulate(self.unet):
                imgs, objs, obj_clip_embs, layout, triples, rel_clip_embs, obj_to_img, triple_to_img, img_paths, caption = batch

                # Convert images to latent space
                latents = self.vae.encode(imgs.to(self.weight_dtype)).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                bsz = latents.shape[0]

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)

                # Sample a random timestep for each image
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                mu, logvar, layout_pred, semantics_embs = self.sl_vae(objs, obj_clip_embs, layout, triples, rel_clip_embs)
                object_embeddings, meta_data = self.object_fusion_tokenizer(layout, semantics_embs.squeeze(0))

                cross_attention_kwargs={}
                cross_attention_kwargs['object_embeddings']= object_embeddings
                cross_attention_kwargs['object_attention_masks']= meta_data['object_attention_masks']

                # Forward diffusion process: Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                # # Get the text embedding for conditioning
                encoder_hidden_states = self.text_encoder(caption, return_dict=False)[0]

                # Predict the noise residual 
                model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, cross_attention_kwargs=cross_attention_kwargs, return_dict=False)[0]

                # Compute loss
                vae_loss = self.vae_criterion(mu, logvar)
                box_loss = self.box_criterion(layout_pred, layout)
                diff_loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                loss = box_loss * self.args.box_loss_weight + vae_loss * self.args.vae_loss_weight + diff_loss * self.args.diff_loss_weight

                # Gather the losses across all processes for logging (if we use distributed training).
                log_loss += self.gather_loss(loss)
                log_box_loss += self.gather_loss(box_loss)
                log_vae_loss += self.gather_loss(vae_loss)
                log_diff_loss += self.gather_loss(diff_loss)
                
                # Backpropagate
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.unet.parameters(), self.args.max_grad_norm)
                
                self.lr_scheduler.step()
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.lr_scheduler_multi.step()
                self.optimizer_multi.step()
                self.optimizer_multi.zero_grad()
 
            # Checks if the accelerator has performed an optimization step behind the scenes
            if self.accelerator.sync_gradients:
                if self.args.use_ema:
                    self.ema_unet.step(self.unet.parameters())
                self.progress_bar.update(1)
                self.global_step += 1
                self.accelerator.log({"train_loss": log_loss}, step=self.global_step)
                self.accelerator.log({"box_loss": log_box_loss}, step=self.global_step)
                self.accelerator.log({"vae_loss": log_vae_loss}, step=self.global_step)
                self.accelerator.log({"diff_loss": log_diff_loss}, step=self.global_step)
                self.accelerator.log({"lr": self.lr_scheduler.get_last_lr()[0]}, step=self.global_step)
                log_loss = 0.0
                log_box_loss = 0.0
                log_vae_loss = 0.0
                log_diff_loss = 0.0

                logs = {"step_loss": '%.4f' % loss.detach().item(), "lr": '%.2e' % self.lr_scheduler.get_last_lr()[0]}
                self.progress_bar.set_postfix(**logs)

                if self.global_step % args.checkpointing_steps == 0:
                    if self.accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{self.global_step}")
                        self.accelerator.save_state(save_path)
                        self.logger.info(f"Saved state to {save_path}")

                        # Validation
                        if self.args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            self.ema_unet.store(self.unet.parameters())
                            self.ema_unet.copy_to(self.unet.parameters())

                        with torch.no_grad():
                            # Validation
                            self.log_validation(self.global_step)

                        if self.args.use_ema:
                            # Switch back to the original UNet parameters.
                            self.ema_unet.restore(self.unet.parameters())

            if self.global_step >= args.max_train_steps:
                break
    
    def gather_loss(self, loss):
        avg_loss = self.accelerator.gather(loss.repeat(args.batch_size)).mean()
        loss = avg_loss.item() / self.args.gradient_accumulation_steps
        return loss
    
    @torch.no_grad()
    def log_validation(self, epoch):
        box_mean_est, box_cov_est = self.sl_vae.collect_data_statistics(self.train_dataloader, self.accelerator.device)
        pbar = tqdm(self.val_dataloader, file=sys.stdout)

        pil_images = []
        for idx, batch in enumerate(pbar):

            imgs, objs, obj_clip_embs, boxes, triples, rel_clip_embs, obj_to_img, triple_to_img, img_paths, caption = batch
            objs, triples = objs.to(self.accelerator.device), triples.to(self.accelerator.device)
            obj_clip_embs, rel_clip_embs = obj_clip_embs.to(self.accelerator.device), rel_clip_embs.to(self.accelerator.device)
            caption = caption.to(self.accelerator.device)

            layout_preds, semantics_embs = self.sl_vae.sample(box_mean_est, box_cov_est, objs, obj_clip_embs, triples, rel_clip_embs, self.accelerator.device)
            layout_image = self.layout_visualization(objs, layout_preds)
            object_embeddings, meta_data = self.object_fusion_tokenizer(layout_preds, semantics_embs.squeeze(0))
        
            cross_attention_kwargs={}
            cross_attention_kwargs['object_embeddings']= torch.cat([object_embeddings, object_embeddings])
            cross_attention_kwargs['object_attention_masks'] = torch.cat([meta_data['object_attention_masks'], meta_data['object_attention_masks']])

            # Tokenize the text and generate the embeddings from the prompt
            cond_embeddings = self.text_encoder(caption)[0]
	
            # Process unconditional input "" for classifier-free guidance.
            max_length = caption.shape[-1]
            uncond_input = self.tokenizer(
                [""],
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.accelerator.device))[0]

            # Concatenate the conditional and unconditional embeddings
            text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

            # Set the scheduler’s timesteps to use during denoising.
            self.scheduler.set_timesteps(self.args.num_inference_steps)

            # Sampling the initial latents.
            latent_size = (1, self.unet.config.in_channels, self.args.resolution // 8, self.args.resolution // 8)
            latent = torch.randn(latent_size, generator=torch.manual_seed(self.args.seed))
            latent = latent.to(self.accelerator.device)
            latent = latent * self.scheduler.init_noise_sigma

            image = None
            for t in tqdm(self.scheduler.timesteps):
                latent_model_input = torch.cat([latent] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)
                
                # Predict the noise residual
                noise_pred = self.unet(latent_model_input, t, text_embeddings, cross_attention_kwargs=cross_attention_kwargs).sample
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.args.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # Compute the previous noisy sample x_t -> x_t-1
                latent = self.scheduler.step(noise_pred, t, latent).prev_sample
		
                # Convert Latents to PIL Images
                scaled_latents = 1.0 / 0.18215 * latent.clone()
                image = self.vae.decode(scaled_latents.to(self.weight_dtype)).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
                image = (image * 255).round().astype("uint8")

            grid = Image.new('RGB', size=(self.args.resolution * 2, self.args.resolution))
            grid.paste(Image.fromarray(image.squeeze(0)), box=(0, 0))
            grid.paste(layout_image, box=(self.args.resolution, 0))
            pil_images.append(grid)

        for index,  pil_image in enumerate(pil_images):
            pil_image.save(f'{index}.png')   

        for tracker in self.accelerator.trackers:
            np_images = np.stack([np.asarray(img) for img in pil_images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")

    def layout_visualization(self, objs, boxes, images=None):
        color = list(np.random.choice(range(256), size=(len(boxes), 3)))
        if images == None:
            layout = Image.new('RGB', size=(self.args.resolution, self.args.resolution)) # Shape: W, H
        draw_layout = ImageDraw.Draw(layout)

        for i, (obj, box) in enumerate(zip(objs, boxes)):
            obj_text = self.vocab['object_idx_to_name'][obj]
            box = box * self.args.resolution
            x0, y0, x1, y1 = box
            if x1 < x0 or y1 < y0:
                break
            draw_layout.rectangle([x0, y0, x1, y1], outline=tuple(color[i]))
            draw_layout.text(xy=(x0, y0), text=obj_text, fill=tuple(color[i]))
        
        return layout

if __name__ == '__main__':
    args = parse_args()

    # start training
    trainer = Trainer(args)
    trainer.start()
