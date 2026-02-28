#!/usr/bin/env python3
"""Create sample training data for testing."""
import os
import random
from pathlib import Path

def create_sample_data(output_dir: str, num_samples: int = 1000):
    """Create sample Russian and English text data."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sample Russian texts
    ru_samples = [
        "Привет, мир! Это тестовый текст для обучения модели.",
        "Искусственный интеллект - это будущее технологий.",
        "Машинное обучение позволяет компьютерам учиться на данных.",
        "Нейронные сети используются для распознавания образов.",
        "Python - популярный язык программирования.",
        "Диффузионные модели генерируют высококачественные изображения.",
        "Трансформеры революционизировали обработку естественного языка.",
        "Обучение с подкреплением используется в играх и робототехнике.",
    ]
    
    # Sample English texts
    en_samples = [
        "Hello world! This is test text for training the model.",
        "Artificial intelligence is the future of technology.",
        "Machine learning allows computers to learn from data.",
        "Neural networks are used for pattern recognition.",
        "Python is a popular programming language.",
        "Diffusion models generate high-quality images.",
        "Transformers revolutionized natural language processing.",
        "Reinforcement learning is used in games and robotics.",
    ]
    
    # Generate mixed dataset
    with open(output_path / "sample_text.txt", "w", encoding="utf-8") as f:
        for i in range(num_samples):
            if random.random() < 0.6:
                # Russian (60%)
                text = random.choice(ru_samples)
            else:
                # English (40%)
                text = random.choice(en_samples)
            
            # Add some variation
            text = text + " " + str(i)
            f.write(text + "\n\n")
    
    print(f"Created {num_samples} samples in {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--num-samples", type=int, default=1000)
    args = parser.parse_args()
    
    create_sample_data(args.output_dir, args.num_samples)
