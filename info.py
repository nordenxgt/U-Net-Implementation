from torchinfo import summary
from model import UNet

def main(): summary(UNet(), input_size=[1, 3, 572, 572])

if __name__ == "__main__":
    main()