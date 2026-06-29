import argparse
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Waste Classification System")
    parser.add_argument("--mode", choices=["train", "evaluate"], default="train")
    args = parser.parse_args()

    if args.mode == "train":
        train_model()
    else:
        evaluate_model()

if __name__ == "__main__":
    main()
