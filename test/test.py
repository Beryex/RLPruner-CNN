import torch

def main():
    print(f"PyTorch Version: {torch.__version__}")

    tensor = torch.tensor([1, 2, 3])
    print(f"Created Tensor: {tensor}")

if __name__ == "__main__":
    main()