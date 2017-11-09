from PIL import Image
filepath = '/home/ubuntu/work/github/rajdeepd/neuralnetwork-programming/ch02/data/train-250-reduced/Type_2/1.jpg'
with Image.open(filepath) as img:
    width, height = img.size
    print width
    print height