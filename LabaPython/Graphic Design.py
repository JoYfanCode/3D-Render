import numpy as np
from PIL import Image
import math
from random import randint


# Squares
# region

# IMAGE - 1
n = 200
img = np.zeros((n, n), dtype=np.uint8)
image1 = Image.fromarray(img, mode='L')
image1.save('Task1Image1.png')

# IMAGE - 2
n = 300
m = 400
img = np.zeros((n, m), dtype=np.uint8)

for i in range(n):
    for j in range(m):
        img[i][j] = 255

image2 = Image.fromarray(img, mode='L')
image2.save('Task1Image2.png')

# IMAGE - 3
n = 300
m = 400
img = np.full((n, m, 3), (255, 0, 0), dtype=np.uint8)
image3 = Image.fromarray(img, mode='RGB')
image3.save('Task1Image3.png')

# IMAGE - 4
n = 128
img = np.full((n, n, 3), (0, 0, 0), dtype=np.uint8)

for i in range(n):
    for j in range(n):
        img[i][j][0] = i
        img[i][j][1] = j
        img[i][j][2] = 0

image4 = Image.fromarray(img, mode='RGB')
image4.save('task1Image4.png')

print("1 task ended up successfully")

# endregion

# Stars
# region

def dotted_line(_x0, _y0, _x1, _y1):

    count = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    step = 1.0 / count

    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * _x0 + t * _x1)
        y = round((1.0 - t) * _y0 + t * _y1)
        img[y, x] = 0


def x_loop_line(x0, y0, x1, y1):

    step = 0

    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        step = 1

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    dx = x1 - x0
    dy = y1 - y0

    q = 0
    if (dx != 0):
        q = abs(dy / dx)

    p = 0.0
    y = int(y0)

    for x in range(int(x0), int(x1)):
        if (step == 1):
            img[y, x] = 0
        else:
            img[x, y] = 0

        p += q

        if (p > 0.5):
            if (y1 > y0):
                y += 1
            else:
                y -= 1

            p -= 1


# Dotted line Star
n = 200
img = np.full((n, n), 255, dtype=np.uint8)

x0 = n/2
y0 = n/2
x1 = 0
y1 = 0
phi = 0.0

while phi < 2 * np.pi:
    x1 = x0 + (n/2-5) * np.cos(phi)
    y1 = y0 + (n/2-5) * np.sin(phi)
    i = 0.0
    dotted_line(x0, y0, x1, y1)
    phi += np.pi / 13

task2Img1 = Image.fromarray(img, 'L')
task2Img1.save('task2-image1.png')

# x loop line Star
n = 200
img = np.full((n, n), 255, dtype=np.uint8)

x0 = n/2
y0 = n/2
x1 = 0
y1 = 0
phi = 0.0
while phi < 2 * np.pi:
    x1 = x0 + (n/2-5) * np.cos(phi)
    y1 = y0 + (n/2-5) * np.sin(phi)
    i = 0.0
    x_loop_line(x0, y0, x1, y1)
    phi += np.pi / 13

task2Img1 = Image.fromarray(img, 'L')
task2Img1.save('task2-image2.png')

print("2 task ended up successfully")

# endregion

# Models
# region

# Vertices
with open("model_1.obj",'r') as f:

    n = 1000
    m = 1000
    img = np.full((n, m), 255, dtype=np.uint8)
    Vectors = []
    Data = f.readlines()

    print("3 task ended up successfully")
    
    for line in Data:
        if(line[0]=='v'):
            Vectors.append(line[2:-1])

    for line in Vectors:
        xyz = line.split()
        x = int(5*n*float(xyz[1]) + n/2)
        y = int(5*m*float(xyz[0]) + m/2)

        if (x >= 0 and x < n and y >= 0 and y < m):
            img[x,y] = 0

    image = Image.fromarray(img,'L')
    image.save("Task4Image.png")

print("4 task ended up successfully")

def DrawLine(x0, y0, x1, y1):
    x_loop_line(x0, y0, x1, y1)

def DrawPolygon(points):

    line1 = lines[int(points[0])-1]
    line2 = lines[int(points[1])-1]
    line3 = lines[int(points[2])-1]

    DrawLine(line1[0], line1[1], line2[0], line2[1])
    DrawLine(line2[0], line2[1], line3[0], line3[1])
    DrawLine(line3[0], line3[1], line1[0], line1[1])

# Polygons
with open("model_1.obj",'r') as f:

    n = 1000
    m = 1000
    img = np.full((n, m), 255, dtype=np.uint8)
    Data = f.readlines()
    
    VectorsF = []
    Polygons = []

    for line in Data:
        if(line[0]=='f'):
            VectorsF.append(line[2:-1])

    for line in VectorsF:
        xyz = line.split()
        numbers = []

        for i in xyz:
            numbers.append(i.split('/')[0])

        Polygons.append(numbers)

    print("5 task ended up successfully")

    VectorsV = []
    lines = []

    for line in Data:
        if(line[0]=='v'):
            VectorsV.append(line[2:-1])

    for line in VectorsV:
        xyz = line.split()
        x = int(5*n*float(xyz[1]) + n/2)
        y = int(5*m*float(xyz[0]) + m/2)
        lines.append([x,y])
        
    for i in Polygons:
        DrawPolygon(i)
        
    image = Image.fromarray(img,'L')
    image.save("Task6Image.png")

print("6 task ended up successfully")

#endregion


def Barycentering(x, y, x0, y0, x1, y1, x2, y2):

    Error = (-1, -1, -1)

    if((x1 - x2)*(y0 - y2) - (y1 - y2)*(x0 - x2)) == 0:
        return Error
    if((x2 - x0)*(y1 - y0) - (y2 - y0)*(x1 - x0)) == 0:
        return Error
    if((x0 - x1)*(y2 - y1) - (y0 - y1)*(x2 - x1)) == 0:
        return Error

    lambda0 = ((x1 - x2)*(y - y2) - (y1 - y2)*(x - x2)) / ((x1 - x2)*(y0 - y2) - (y1 - y2)*(x0 - x2))
    lambda1 = ((x2 - x0)*(y - y0) - (y2 - y0)*(x - x0)) / ((x2 - x0)*(y1 - y0) - (y2 - y0)*(x1 - x0))
    lambda2 = ((x0 - x1)*(y - y1) - (y0 - y1)*(x - x1)) / ((x0 - x1)*(y2 - y1) - (y0 - y1)*(x2 - x1))

    return (lambda0, lambda1, lambda2)


def Rectangle(x0, y0, x1, y1, x2, y2):

    xmin = min(x0, x1, x2)
    if (xmin < 0):
        xmin = 0

    ymin = min(y0, y1, y2)
    if (ymin < 0):
        ymin = 0

    xmax = max(x0, x1, x2)
    if (xmax > (n - 1)):
        xmax = n - 1

    ymax = max(y0, y1, y2)
    if (ymax > (m - 1)):
        ymax = m - 1

    return(xmin, xmax, ymin, ymax)

def DrawPixel(x, y, x0, y0, x1, y1, x2, y2, color):

    VectorBar = Barycentering(x, y, x0, y0, x1, y1, x2, y2)

    if (VectorBar[0] >= 0 and VectorBar[1] >= 0 and VectorBar[2] >= 0):
        img[x, y] = color

def ColorPolygon(i):

    line1 = lines[int(i[0])-1]
    line2 = lines[int(i[1])-1]
    line3 = lines[int(i[2])-1]

    x0 = int(line1[0])
    y0 = int(line1[1])
    x1 = int(line2[0])
    y1 = int(line2[1])
    x2 = int(line3[0])
    y2 = int(line3[1])

    space = Rectangle(x0, y0, x1, y1, x2, y2)
    color = (randint(100, 200), randint(100, 200), randint(100, 200))

    for i in range(space[0], space[1]):
        for j in range(space[2], space[3]):
            DrawPixel(i, j, x0, y0, x1, y1, x2, y2, color)

with open("model_1.obj",'r') as f:

    n = 1000
    m = 1000
    img = np.full((n, m, 3), (255, 255, 255), dtype=np.uint8)
    Data = f.readlines()
    
    VectorsF = []
    Polygons = []

    for line in Data:
        if(line[0]=='f'):
            VectorsF.append(line[2:-1])

    for line in VectorsF:
        xyz = line.split()
        numbers = []

        for i in xyz:
            numbers.append(i.split('/')[0])

        Polygons.append(numbers)

    VectorsV = []
    lines = []

    for line in Data:
        if(line[0]=='v'):
            VectorsV.append(line[2:-1])

    for line in VectorsV:
        xyz = line.split()
        x = int(5*n*float(xyz[1]) + n/2)
        y = int(5*m*float(xyz[0]) + m/2)
        lines.append([x,y])
        
    for i in Polygons:
        ColorPolygon(i)
        
    image = Image.fromarray(img,'RGB')
    image.save("Task10Image.png")

print("7-10 tasks ended up successfully")

    

