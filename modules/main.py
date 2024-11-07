# main.py

import torch
from pathlib import Path
import pytorch_lightning as pl
import wandb
import time
import curses
import logging

# Pytorch Lightning Imports
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from datasets import load_dataset

# Local Imports
from config import TrainingConfig
from dataloader import TextClassificationDataModule
from lightning_trainer import LightningClassifier # check in lightning_trainer.py
from lightning_model_utils import setup_lightning_model
from lightning_inference import LightningSentimentPredictor



def run_training():
    # Add logging setup at the start
    logging.basicConfig(
        filename='training.log',
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Initialize config (keep existing config setup)
    config = TrainingConfig(
        model_name="facebook/opt-125m",
        train_size=45000, # increased from 1000 to 50000
        eval_size=10000, # increased from 1000 to 10000
        batch_size=64, # increased from 32 to 64
        num_epochs=10, # increased from 6 to 10
        model_max_length=512,
        learning_rate=1e-5, # consider reducing this 
        weight_decay=0.01,
        warmup_ratio=0.1,
        output_dir="./tmp_output",
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_steps=10000,
        save_steps=1000, # loonsider increasing thisg every 1000 steps
        early_stopping_patience=5,
        num_train_samples=45000,  # Added
        num_eval_samples=10000,     # Added
        save_top_k=3
    )

    # Initialize wandb logger
    wandb_logger = WandbLogger(project="sentiment-analysis",
                               name=f"run-{config.model_name}",
                               config=config.__dict__)
    
    # Initialize callbacks
    callbacks = [ModelCheckpoint(dirpath=config.output_dir,
                                  filename="{epoch}-{val_f1:.2f}",
                                  monitor="val_f1",
                                  mode="max",
                                  save_top_k=config.save_top_k,
                                  save_last=True),
                 EarlyStopping(monitor="val_loss",
                               patience=config.early_stopping_patience,
                               mode="min")]


    # Initialize trainer
    trainer = pl.Trainer(
        accelerator='gpu' if config.device == 'cuda' else 'cpu',
        callbacks=callbacks,
        logger=wandb_logger,
        max_epochs=config.num_epochs,
        max_steps=config.max_steps,
        precision='16-mixed' if config.device == 'cuda' else '32',
        gradient_clip_val=config.gradient_clip_val,
        enable_checkpointing=True,
        devices=1
    )

    # Initialize model and data module
    model, tokenizer = setup_lightning_model(config)
    data_module = TextClassificationDataModule(config, tokenizer)

    # Train model
    try:
        trainer.fit(model, datamodule=data_module)
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        # Save model checkpoint to W&B
        artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}", 
            type="model",
            description="Trained sentiment analysis model"
        )
        artifact.add_dir(config.output_dir)
        wandb.log_artifact(artifact)
        wandb.finish()

if __name__ == "__main__":
    run_training()