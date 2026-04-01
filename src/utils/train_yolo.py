'''
Train YOLO model with command line arguments.
'''


import os
import sys
# Add parent directory to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import logging
from ultralytics import YOLO
import torch
from src.logging_utils import setup_logger

# Initialize logger
logger = setup_logger(name="train_yolo")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Windows-specific fix for a common OpenMP error with some libraries
torch.set_num_threads(8)

def train_yolo(data_path: str = None, resume_path: str = None, total_epochs: int = 100, output_engine_path: str = "weights/best.engine", batch: int = 16, imgsz: int = 640):
    if not torch.cuda.is_available():
        logger.error("PyTorch cannot see the GPU.")
        return

    logger.info(f"Success: Using GPU - {torch.cuda.get_device_name(0)}")

    # 1. Point directly to the last.pt file of your interrupted run
    # I have adjusted the path you provided to target last.pt

    if resume_path is not None:
        model_path = resume_path
        model = YOLO(model_path)
        model.train(resume=True, epochs=total_epochs)

    if data_path is not None and resume_path is None:
        logger.info("Training a new model from scratch...")
        # Train a new model
        # Using the official YOLOv8s.pt as a starting point for transfer learning.
        model = YOLO('yolov8s.pt')
        model.train(
            data=data_path,      
            epochs=total_epochs,           
            imgsz=imgsz,             
            batch=batch,           # Optimized for 8GB VRAM
            device=0,              
            project='runs',        
            name='crowdhuman_run'  
        )
    
    engine_path = model.export(
        format="engine",
        half=True,
        device=0,
        path=output_engine_path
    )

    logger.info(f"Engine saved at: {engine_path}")
    logger.info("Training completely finished!")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train YOLO model with command line arguments.")
    parser.add_argument('--data_path', type=str, default='data/data.yaml', help='Path to data.yaml file')
    parser.add_argument('--resume_path', type=str, help='Path to the last.pt file to resume training')
    parser.add_argument('--epochs', type=int, default=100, help='Total number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--output_engine', type=str, default='weights/best.engine', help='Output path for the engine file')
    
    args = parser.parse_args()
    
    train_yolo(
        data_path=args.data_path,
        resume_path=args.resume_path,
        total_epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        output_engine_path=args.output_engine
    )