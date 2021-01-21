from PIL import Image
import sys

im = Image.open(sys.argv[1])
fil = open('val.csv', 'w')
pixel = im.load()
row, column = im.size
for y in range(column):
    for x in range(row):
        pix = pixel[x, y]
        pix = pix / 255.0
        fil.write(str(pix) + '\n')
fil.close()