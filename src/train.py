import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modeling import train_and_export_pipeline

if __name__ == "__main__":
    train_and_export_pipeline()
