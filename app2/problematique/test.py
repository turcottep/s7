import torch
#53x53
#gris de 0 a 155
#1 instance par image, 3 max
#bg-background: 000

def main():
    print("yo")
    x = torch.rand(5,3,2)
    print(x)

if __name__ == "__main__":
    main()