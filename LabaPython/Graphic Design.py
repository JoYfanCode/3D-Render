import numpy as np
from PIL import Image, ImageOps
import math
from random import randint

# Squares
# region

## IMAGE - 1
#n = 200
#img = np.zeros((n, n), dtype=np.uint8)
#image1 = Image.fromarray(img, mode='L')
#image1.save('Task1Image1.png')

## IMAGE - 2
#n = 300
#m = 400
#img = np.zeros((n, m), dtype=np.uint8)

#for i in range(n):
#    for j in range(m):
#        img[i][j] = 255

#image2 = Image.fromarray(img, mode='L')
#image2.save('Task1Image2.png')

## IMAGE - 3
#n = 300
#m = 400
#img = np.full((n, m, 3), (255, 0, 0), dtype=np.uint8)
#image3 = Image.fromarray(img, mode='RGB')
#image3.save('Task1Image3.png')

## IMAGE - 4
#n = 128
#img = np.full((n, n, 3), (0, 0, 0), dtype=np.uint8)

#for i in range(n):
#    for j in range(n):
#        img[i][j][0] = i
#        img[i][j][1] = j
#        img[i][j][2] = 0

#image4 = Image.fromarray(img, mode='RGB')
#image4.save('task1Image4.png')

#print("1 task ended up successfully")

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


## Dotted line Star
#n = 200
#img = np.full((n, n), 255, dtype=np.uint8)

#x0 = n/2
#y0 = n/2
#x1 = 0
#y1 = 0
#phi = 0.0

#while phi < 2 * np.pi:
#    x1 = x0 + (n/2-5) * np.cos(phi)
#    y1 = y0 + (n/2-5) * np.sin(phi)
#    i = 0.0
#    dotted_line(x0, y0, x1, y1)
#    phi += np.pi / 13

#task2Img1 = Image.fromarray(img, 'L')
#task2Img1.save('task2-image1.png')

## x loop line Star
#n = 200
#img = np.full((n, n), 255, dtype=np.uint8)

#x0 = n/2
#y0 = n/2
#x1 = 0
#y1 = 0
#phi = 0.0
#while phi < 2 * np.pi:
#    x1 = x0 + (n/2-5) * np.cos(phi)
#    y1 = y0 + (n/2-5) * np.sin(phi)
#    i = 0.0
#    x_loop_line(x0, y0, x1, y1)
#    phi += np.pi / 13

#task2Img1 = Image.fromarray(img, 'L')
#task2Img1.save('task2-image2.png')

#print("2 task ended up successfully")

# endregion

# Models
# region

# VerticesCoordinates
#with open("model_1.obj",'r') as f:

#    n = 1000
#    m = 1000
#    img = np.full((n, m), 255, dtype=np.uint8)
#    Vectors = []
#    Data = f.readlines()

#    print("3 task ended up successfully")
    
#    for line in Data:
#        if(line[0]=='v' and line[1]==' '):
#            Vectors.append(line[2:-1])

#    for line in Vectors:
#        VerticeCoordinates = line.split()
#        x = int(5*n*float(VerticeCoordinates[1]) + n/2)
#        y = int(5*m*float(VerticeCoordinates[0]) + m/2)

#        if (x >= 0 and x < n and y >= 0 and y < m):
#            img[x,y] = 0

#    image = Image.fromarray(img,'L')
#    image.save("Task4Image.png")

#print("4 task ended up successfully")

def DrawLine(x0, y0, x1, y1):
    x_loop_line(x0, y0, x1, y1)

def DrawPolygon(polygon):

    line1 = VerticesCoordinates[int(polygon[0])-1]
    line2 = VerticesCoordinates[int(polygon[1])-1]
    line3 = VerticesCoordinates[int(polygon[2])-1]

    DrawLine(line1[0], line1[1], line2[0], line2[1])
    DrawLine(line2[0], line2[1], line3[0], line3[1])
    DrawLine(line3[0], line3[1], line1[0], line1[1])

# PolygonsNumbers
#with open("model_1.obj",'r') as f:

#    n = 1000
#    m = 1000
#    img = np.full((n, m), 255, dtype=np.uint8)
#    Data = f.readlines()
    
#    LinesF = []
#    PolygonsNumbers = []

#    for line in Data:
#        if(line[0]=='f'):
#            LinesF.append(line[2:-1])

#    for line in LinesF:
#        VerticeCoordinates = line.split()
#        numbers = []

#        for i in VerticeCoordinates:
#            numbers.append(i.split('/')[0])

#        PolygonsNumbers.append(numbers)

#    print("5 task ended up successfully")

#    LinesV = []
#    VerticesCoordinates = []

#    for line in Data:
#        if(line[0]=='v' and line[1]==' '):
#            LinesV.append(line[2:-1])

#    for line in LinesV:
#        VerticeCoordinates = line.split()
#        x = int(5*n*float(VerticeCoordinates[1]) + n/2)
#        y = int(5*m*float(VerticeCoordinates[0]) + m/2)
#        VerticesCoordinates.append([x,y])
        
#    for i in PolygonsNumbers:
#        DrawPolygon(i)
        
#    image = Image.fromarray(img,'L')
#    image.save("Task6Image.png")

#print("6 task ended up successfully")

#endregion

# Colorful Models
#region

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
    lambda2 = 1.0 - lambda0 - lambda1

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

    return(int(xmin), int(xmax), int(ymin), int(ymax))

def DrawPixel(x, y, x0, y0, x1, y1, x2, y2, color):

    VectorBar = Barycentering(x, y, x0, y0, x1, y1, x2, y2)

    if (VectorBar[0] >= 0 and VectorBar[1] >= 0 and VectorBar[2] >= 0):
        img[x, y] = color

def ColorPolygon(polygon):

    line1 = VerticesCoordinates[int(polygon[0])-1]
    line2 = VerticesCoordinates[int(polygon[1])-1]
    line3 = VerticesCoordinates[int(polygon[2])-1]

    x0 = line1[0]
    y0 = line1[1]
    x1 = line2[0]
    y1 = line2[1]
    x2 = line3[0]
    y2 = line3[1]

    rect = Rectangle(x0, y0, x1, y1, x2, y2)
    color = (randint(100, 200), randint(100, 200), randint(100, 200))

    for i in range(rect[0], rect[1]):
        for j in range(rect[2], rect[3]):
            DrawPixel(i, j, x0, y0, x1, y1, x2, y2, color)

#with open("model_1.obj",'r') as f:

#    n = 1000
#    m = 1000
#    img = np.full((n, m, 3), (255, 255, 255), dtype=np.uint8)
#    Data = f.readlines()
    
#    LinesF = []
#    PolygonsNumbers = []

#    for line in Data:
#        if(line[0]=='f'):
#            LinesF.append(line[2:-1])

#    for line in LinesF:
#        VerticeCoordinates = line.split()
#        numbers = []

#        for i in VerticeCoordinates:
#            numbers.append(i.split('/')[0])

#        PolygonsNumbers.append(numbers)

#    LinesV = []
#    VerticesCoordinates = []

#    for line in Data:
#        if(line[0]=='v' and line[1]==' '):
#            LinesV.append(line[2:-1])

#    for line in LinesV:
#        VerticeCoordinates = line.split()
#        x = int(5*n*float(VerticeCoordinates[1]) + n/2)
#        y = int(5*m*float(VerticeCoordinates[0]) + m/2)
#        VerticesCoordinates.append([x,y])
        
#    for i in PolygonsNumbers:
#        ColorPolygon(i)
        
#    image = Image.fromarray(img,'RGB')
#    image.save("Task10Image.png")

#print("7-10 tasks ended up successfully")

#endregion

# VerticesLightGuro Models
#region

def CalculateNormalPolygon(polygonNumbers):

    line1 = VerticesCoordinatesN[int(polygonNumbers[0])-1]
    line2 = VerticesCoordinatesN[int(polygonNumbers[1])-1]
    line3 = VerticesCoordinatesN[int(polygonNumbers[2])-1]

    x0 = line1[0]
    y0 = line1[1]
    z0 = line1[2]
    x1 = line2[0]
    y1 = line2[1]
    z1 = line2[2]
    x2 = line3[0]
    y2 = line3[1]
    z2 = line3[2]

    xn = (y1 - y0)*(z1 - z2) - (y1 - y2)*(z1 - z0)
    yn = (x1 - x2)*(z1 - z0) - (x1 - x0)*(z1 - z2)
    zn = (x1 - x0)*(y1 - y2) - (x1 - x2)*(y1 - y0)

    NormalLength = math.sqrt(xn**2 + yn**2 + zn**2)

    if (NormalLength == 0):
        return [0, 0, 0]

    xn /= NormalLength
    yn /= NormalLength
    zn /= NormalLength

    return [xn, yn, zn]

def CosPolygon(polygonNumbers):
    
    VerticesLightGuro = [0, 0, 1]

    NormalPolygon = CalculateNormalPolygon(polygonNumbers)

    return VerticesLightGuro[0]*NormalPolygon[0] + VerticesLightGuro[1]*NormalPolygon[1] + VerticesLightGuro[2]*NormalPolygon[2]

def NeighboringVerticesNormalsPolygon(polygonNumbers):

    numberVertice1 = int(polygonNumbers[0])-1
    numberVertice2 = int(polygonNumbers[1])-1
    numberVertice3 = int(polygonNumbers[2])-1

    NormalPolygon = CalculateNormalPolygon(polygonNumbers)

    VerticesNormalsOwn[numberVertice1][0] += NormalPolygon[0]
    VerticesNormalsOwn[numberVertice1][1] += NormalPolygon[1]
    VerticesNormalsOwn[numberVertice1][2] += NormalPolygon[2]

    VerticesNormalsOwn[numberVertice2][0] += NormalPolygon[0]
    VerticesNormalsOwn[numberVertice2][1] += NormalPolygon[1]
    VerticesNormalsOwn[numberVertice2][2] += NormalPolygon[2]

    VerticesNormalsOwn[numberVertice3][0] += NormalPolygon[0]
    VerticesNormalsOwn[numberVertice3][1] += NormalPolygon[1]
    VerticesNormalsOwn[numberVertice3][2] += NormalPolygon[2]

def DrawPixelWithBuffer(x, y, x0, y0, z0, x1, y1, z1, x2, y2, z2, color):

    VectorBar = Barycentering(x, y, x0, y0, x1, y1, x2, y2)

    if (VectorBar[0] >= 0 and VectorBar[1] >= 0 and VectorBar[2] >= 0):
        ZBarKoef = VectorBar[0]*z0 + VectorBar[1]*z1 + VectorBar[2]*z2

        if ZBarKoef <= ZBuffer[x][y]:
            ZBuffer[x][y] = ZBarKoef
            img[x, y] = color

def LightPolygon(polygon, color):

    line1 = VerticesCoordinates[int(polygon[0])-1]
    line2 = VerticesCoordinates[int(polygon[1])-1]
    line3 = VerticesCoordinates[int(polygon[2])-1]

    CosLight = CosPolygon(polygon)

    x0 = line1[0]
    y0 = line1[1]
    z0 = line1[2]
    x1 = line2[0]
    y1 = line2[1]
    z1 = line2[2]
    x2 = line3[0]
    y2 = line3[1]
    z2 = line3[2]

    rect = Rectangle(x0, y0, x1, y1, x2, y2)
    color = (-color[0]*CosLight, -color[1]*CosLight, -color[2]*CosLight)

    for i in range(rect[0], rect[1]):
        for j in range(rect[2], rect[3]):
            DrawPixelWithBuffer(i, j, x0, y0, z0, x1, y1, z1, x2, y2, z2, color)

#with open("model_1.obj",'r') as f:

#    n = 1000
#    m = 1000
#    img = np.full((n, m, 3), (255, 255, 255), dtype=np.uint8)
#    Data = f.readlines()
#    ZBuffer = [[np.inf for x in range(n)] for x in range(n)]
    
#    LinesF = []
#    PolygonsNumbers = []

#    VerticesNormalsOwn = []

#    for line in Data:
#        if(line[0]=='f'):
#            LinesF.append(line[2:-1])

#    for line in LinesF:
#        VerticeCoordinates = line.split()
#        numbers = []

#        for i in VerticeCoordinates:
#            numbers.append(i.split('/')[0])

#        PolygonsNumbers.append(numbers)

#    LinesV = []
#    VerticesCoordinates = []

#    for line in Data:
#        if(line[0]=='v' and line[1]==' '):
#            LinesV.append(line[2:-1])

#    for line in LinesV:
#        VerticeCoordinates = line.split()
#        x = int(5*n*float(VerticeCoordinates[1]) + n/2)
#        y = int(5*m*float(VerticeCoordinates[0]) + m/2)
#        z = int(5*n*float(VerticeCoordinates[2]))
#        VerticesCoordinates.append([x, y, z])
        
#    for i in PolygonsNumbers:
#        if (CosPolygon(i) < 0):
#            LightPolygon(i, (150, 150, 150))
        
#    image = Image.fromarray(img,'RGB')
#    image.save("Task14Image.png")

#print("11-14 tasks ended up successfully")

#endregion

# Movement
#region

def Rotate(x, y, z):

    alfa = 2*math.pi * 0 / 360
    betta = 2*math.pi * 180 / 360
    gamma = 2*math.pi * 180 / 360

    X = np.array([x, y, z])

    R1 = np.array([[1, 0, 0], [0, np.cos(alfa), np.sin(alfa)], [0, -np.sin(alfa), np.cos(alfa)]])
    R2 = np.array([[np.cos(betta), 0, np.sin(betta)], [0, 1, 0], [-np.sin(betta), 0, np.cos(betta)]])
    R3 = np.array([[np.cos(gamma), np.sin(gamma), 0], [-np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])

    R = np.dot(np.dot(R1, R2), R3)
    return np.dot(R, X)

def Move(x, y, z):

    u0 = n/2.0
    v0 = m/2.0
    MultX = 40*n
    MultY = 40*m
    MultZ = 40*n

    x += 0
    y += 0.045
    z += 5.0

    u = MultX * x / z + u0
    v = MultY * y / z + v0

    return np.array([u, v, MultZ])


#with open("model_1.obj",'r') as f:

#    n = 1000
#    m = 1000
#    img = np.full((n, m, 3), (255, 255, 255), dtype=np.uint8)
#    Data = f.readlines()
#    ZBuffer = [[np.inf for x in range(n)] for x in range(n)]
    
#    LinesF = []
#    PolygonsNumbers = []

#    for line in Data:
#        if(line[0]=='f'):
#            LinesF.append(line[2:-1])

#    for line in LinesF:
#        VerticeCoordinates = line.split()
#        numbers = []

#        for i in VerticeCoordinates:
#            numbers.append(i.split('/')[0])

#        PolygonsNumbers.append(numbers)

#    LinesV = []
#    VerticesCoordinates = []

#    for line in Data:
#        if(line[0]=='v' and line[1]==' '):
#            LinesV.append(line[2:-1])

#    for line in LinesV:
#        VerticeCoordinates = line.split()

#        VerticeCoordinates = Rotate(float(VerticeCoordinates[0]), float(VerticeCoordinates[1]), float(VerticeCoordinates[2]))
#        VerticeCoordinates = Move(float(VerticeCoordinates[0]), float(VerticeCoordinates[1]), float(VerticeCoordinates[2]))
#        VerticeCoordinates = [float(VerticeCoordinates[1]), float(VerticeCoordinates[0]), float(VerticeCoordinates[2])]

#        VerticesCoordinates.append([int(VerticeCoordinates[0]), int(VerticeCoordinates[1]), int(5*n*VerticeCoordinates[2])])
        
#    for i in PolygonsNumbers:
#        if (CosPolygon(i) < 0):
#            LightPolygon(i, (150, 150, 150))
        
#    image = Image.fromarray(img,'RGB')
#    image.save("Task16Image.png")

#print("15-16 tasks ended up successfully")

#endregion

# Guro VerticesLight
#region

def GuroVertice(normalsPolygon):
    
    l = [0, 0, 1]
    
    nx = normalsPolygon[0]
    ny = normalsPolygon[1]
    nz = normalsPolygon[2]

    lenNormal = math.sqrt(pow(nx, 2) + pow(ny, 2) + pow(nz, 2))
    lenLight = math.sqrt(pow(l[0], 2) + pow(l[1], 2) + pow(l[2], 2))
    
    if (lenNormal != 0 and lenLight != 0):
        return (l[0]*nx + l[1]*ny + l[2]*nz)/(lenNormal*lenLight)
    else:
        return 1

def DrawPixelGuro(x, y, x0, y0, z0, x1, y1, z1, x2, y2, z2, polygonNormalsNumbers):

    l1 = VerticesLightGuro[int(polygonNormalsNumbers[0]) - 1]
    l2 = VerticesLightGuro[int(polygonNormalsNumbers[1]) - 1]
    l3 = VerticesLightGuro[int(polygonNormalsNumbers[2]) - 1]

    VectorBar = Barycentering(x, y, x0, y0, x1, y1, x2, y2)

    if (VectorBar[0] >= 0 and VectorBar[1] >= 0 and VectorBar[2] >= 0):
        ZBarKoef = VectorBar[0]*z0 + VectorBar[1]*z1 + VectorBar[2]*z2

        if ZBarKoef <= ZBuffer[x][y]:
            ZBuffer[x][y] = ZBarKoef
            img[x, y] = (225*(VectorBar[0]*l1 + VectorBar[1]*l2  + VectorBar[2]*l3),
                        225*(VectorBar[0]*l1 + VectorBar[1]*l2  + VectorBar[2]*l3),
                       225*(VectorBar[0]*l1 + VectorBar[1]*l2  + VectorBar[2]*l3))


def GuroLightPolygon(polygonNumbers, polygonNormalsNumbers):

    line1 = VerticesCoordinates[int(polygonNumbers[0])-1]
    line2 = VerticesCoordinates[int(polygonNumbers[1])-1]
    line3 = VerticesCoordinates[int(polygonNumbers[2])-1]

    x0 = line1[0]
    y0 = line1[1]
    z0 = line1[2]
    x1 = line2[0]
    y1 = line2[1]
    z1 = line2[2]
    x2 = line3[0]
    y2 = line3[1]
    z2 = line3[2]

    rect = Rectangle(x0, y0, x1, y1, x2, y2)

    for i in range(rect[0], rect[1]):
        for j in range(rect[2], rect[3]):
            DrawPixelGuro(i, j, x0, y0, z0, x1, y1, z1, x2, y2, z2, polygonNormalsNumbers)


#with open("model_1.obj",'r') as f:

#    n = 1000
#    m = 1000
#    img = np.full((n, m, 3), (255, 255, 255), dtype=np.uint8)
#    Data = f.readlines()
#    ZBuffer = [[np.inf for x in range(n)] for x in range(n)]

#    LinesV = []
#    LinesVn = []
#    LinesF = []

#    VerticesCoordinates = []
#    VerticesCoordinatesN = []
#    VerticesNormals = []
#    VerticesNormalsOwn = []
#    VerticesLightGuro = []

#    PolygonsNumbers = []
#    PolygonsNormalsNumbers = []


#    for line in Data:
#        if(line[0]=='v' and line[1]==' '):
#            LinesV.append(line[2:-1])
#    for line in Data:
#        if((line[0]=='v')& (line[1]=='n') & (line[2]==' ')):
#            LinesVn.append(line[3:-1])
#    for line in Data:
#        if(line[0]=='f'):
#            LinesF.append(line[2:-1])

#    # V
#    for line in LinesV:
#        VerticeCoordinates = line.split()

#        x = int(5*n*float(VerticeCoordinates[0]))
#        y = int(5*m*float(VerticeCoordinates[1]))
#        z = int(5*n*float(VerticeCoordinates[2]))
#        VerticesCoordinatesN.append([x, y, z])

#        VerticeCoordinates = Rotate(float(VerticeCoordinates[0]), float(VerticeCoordinates[1]), float(VerticeCoordinates[2]))
#        VerticeCoordinates = Move(float(VerticeCoordinates[0]), float(VerticeCoordinates[1]), float(VerticeCoordinates[2]))
#        VerticeCoordinates = [float(VerticeCoordinates[1]), float(VerticeCoordinates[0]), float(VerticeCoordinates[2])]

#        VerticesCoordinates.append([int(VerticeCoordinates[0]), int(VerticeCoordinates[1]), int(VerticeCoordinates[2])])

#    # Vn
#    for line in LinesVn:
#        VerticeNormals = line.split()
#        VerticesNormals.append([float(VerticeNormals[0]), float(VerticeNormals[1]), float(VerticeNormals[2])])

#    # F
#    for line in LinesF:
#        VerticeCoordinates = line.split()
#        Numbers = []
#        NormVertices = []
#        for i in VerticeCoordinates:
#            Numbers.append(i.split('/')[0])
#            NormVertices.append(i.split('/')[2])
#        PolygonsNumbers.append(Numbers)
#        PolygonsNormalsNumbers.append(NormVertices)

#    # Calculate Normals
#    for vertice in VerticesCoordinates:
#        VerticesNormalsOwn.append([0, 0, 0])

#    for polygonNumbers in PolygonsNumbers:
#        NeighboringVerticesNormalsPolygon(polygonNumbers)

#    for VerticeNormalOwn in VerticesNormalsOwn:
#        LengthNormal = math.sqrt(VerticeNormalOwn[0]**2 + VerticeNormalOwn[1]**2 + VerticeNormalOwn[2]**2)

#        if (LengthNormal == 0):
#            LengthNormal = 1

#        VerticeNormalOwn[0] /= LengthNormal
#        VerticeNormalOwn[1] /= LengthNormal
#        VerticeNormalOwn[2] /= LengthNormal

#    # Guro Light
#    for verticeNormals in VerticesNormals:
#        VerticesLightGuro.append(GuroVertice(verticeNormals))

#    # Draw
#    for i in range(len(PolygonsNumbers)):
#        if (CosPolygon(PolygonsNumbers[i]) < 0):
#            GuroLightPolygon(PolygonsNumbers[i], PolygonsNormalsNumbers[i])
        
#    image = Image.fromarray(img,'RGB')
#    image.save("Task17Image.png")

#print("17 tasks ended up successfully")


#endregion

# Texturing
#region

TextureImg = np.array(ImageOps.flip(Image.open("bunny-atlas.jpg")))

def DrawPixelTexture(x, y, x0, y0, z0, x1, y1, z1, x2, y2, z2, polygonNormalsNumbers, polygonTextureNumbers):

    l1 = VerticesLightGuro[int(polygonNormalsNumbers[0]) - 1]
    l2 = VerticesLightGuro[int(polygonNormalsNumbers[1]) - 1]
    l3 = VerticesLightGuro[int(polygonNormalsNumbers[2]) - 1]

    u0 = VerticesTextureUV[int(polygonTextureNumbers[0]) - 1][0]
    v0 = VerticesTextureUV[int(polygonTextureNumbers[0]) - 1][1]
    u1 = VerticesTextureUV[int(polygonTextureNumbers[1]) - 1][0]
    v1 = VerticesTextureUV[int(polygonTextureNumbers[1]) - 1][1]
    u2 = VerticesTextureUV[int(polygonTextureNumbers[2]) - 1][0]
    v2 = VerticesTextureUV[int(polygonTextureNumbers[2]) - 1][1]

    VectorBar = Barycentering(x, y, x0, y0, x1, y1, x2, y2)

    if (VectorBar[0] >= 0 and VectorBar[1] >= 0 and VectorBar[2] >= 0):
        ZBarKoef = VectorBar[0]*z0 + VectorBar[1]*z1 + VectorBar[2]*z2

        if ZBarKoef <= ZBuffer[x][y]:
            ZBuffer[x][y] = ZBarKoef
            u = int(1024 * (VectorBar[0]*u0 + VectorBar[1]*u1 + VectorBar[2]*u2))
            v = int(1024 * (VectorBar[0]*v0 + VectorBar[1]*v1 + VectorBar[2]*v2))
            ShadowMult = 255*(VectorBar[0]*l1 + VectorBar[1]*l2  + VectorBar[2]*l3)
            Red = int(TextureImg[v, u][0]*(ShadowMult/255.0))
            Green = int(TextureImg[v, u][1]*(ShadowMult/255.0))
            Blue = int(TextureImg[v, u][2]*(ShadowMult/255.0))
            img[x, y] = [Red, Green, Blue]


def TexturePolygon(polygonNumbers, polygonNormalsNumbers, polygonTextureNumbers):

    line1 = VerticesCoordinates[int(polygonNumbers[0])-1]
    line2 = VerticesCoordinates[int(polygonNumbers[1])-1]
    line3 = VerticesCoordinates[int(polygonNumbers[2])-1]

    x0 = line1[0]
    y0 = line1[1]
    z0 = line1[2]
    x1 = line2[0]
    y1 = line2[1]
    z1 = line2[2]
    x2 = line3[0]
    y2 = line3[1]
    z2 = line3[2]

    rect = Rectangle(x0, y0, x1, y1, x2, y2)

    for i in range(rect[0], rect[1]):
        for j in range(rect[2], rect[3]):
            DrawPixelTexture(i, j, x0, y0, z0, x1, y1, z1, x2, y2, z2, polygonNormalsNumbers, polygonTextureNumbers)

with open("model_1.obj",'r') as f:

    n = 1000
    m = 1000
    img = np.full((n, m, 3), (255, 255, 255), dtype=np.uint8)
    Data = f.readlines()
    ZBuffer = [[np.inf for x in range(n)] for x in range(n)]

    LinesV = []
    LinesVt = []
    LinesVn = []
    LinesF = []

    VerticesCoordinates = []
    VerticesCoordinatesN = []
    VerticesNormals = []
    VerticesLightGuro = []
    VerticesTextureUV = []

    PolygonsNumbers = []
    PolygonsNormalsNumbers = []
    PolygonsTextureNumbers = []

    for line in Data:
        if(line[0]=='v' and line[1]==' '):
            LinesV.append(line[2:-1])
    for line in Data:
        if((line[0]=='v')& (line[1]=='t') & (line[2]==' ')):
            LinesVt.append(line[3:-1])
    for line in Data:
        if((line[0]=='v')& (line[1]=='n') & (line[2]==' ')):
            LinesVn.append(line[3:-1])
    for line in Data:
        if(line[0]=='f'):
            LinesF.append(line[2:-1])

    # V
    for line in LinesV:
        VerticeCoordinates = line.split()

        x = int(5*n*float(VerticeCoordinates[0]))
        y = int(5*m*float(VerticeCoordinates[1]))
        z = int(5*n*float(VerticeCoordinates[2]))
        VerticesCoordinatesN.append([x, y, z])

        VerticeCoordinates = Rotate(float(VerticeCoordinates[0]), float(VerticeCoordinates[1]), float(VerticeCoordinates[2]))
        VerticeCoordinates = Move(float(VerticeCoordinates[0]), float(VerticeCoordinates[1]), float(VerticeCoordinates[2]))
        VerticeCoordinates = [float(VerticeCoordinates[1]), float(VerticeCoordinates[0]), float(VerticeCoordinates[2])]

        VerticesCoordinates.append([int(VerticeCoordinates[0]), int(VerticeCoordinates[1]), int(5*n*VerticeCoordinates[2])])

    # Vt
    for line in LinesVt:
        VerticeTextureUV = line.split()
        VerticesTextureUV.append([float(VerticeTextureUV[0]), float(VerticeTextureUV[1])])

    # Vn
    for line in LinesVn:
        VerticeNormals = line.split()
        VerticesNormals.append([float(VerticeNormals[0]), float(VerticeNormals[1]), float(VerticeNormals[2])])

    # F
    for line in LinesF:
        SplitLine = line.split()
        PolygonNumbers = []
        PolygonTextureNumbers = []
        PolygonNormalsNumbers = []
        for i in SplitLine:
            PolygonNumbers.append(i.split('/')[0])
            PolygonTextureNumbers.append(i.split('/')[1])
            PolygonNormalsNumbers.append(i.split('/')[2])
        PolygonsNumbers.append(PolygonNumbers)
        PolygonsTextureNumbers.append(PolygonTextureNumbers)
        PolygonsNormalsNumbers.append(PolygonNormalsNumbers)

    # Guro Light
    for verticeNormals in VerticesNormals:
        VerticesLightGuro.append(GuroVertice(verticeNormals))

    # Draw
    for i in range(len(PolygonsNumbers)):
        if (CosPolygon(PolygonsNumbers[i]) < 0):
            TexturePolygon(PolygonsNumbers[i], PolygonsNormalsNumbers[i], PolygonsTextureNumbers[i])
        
    image = Image.fromarray(img,'RGB')
    image.save("Task18Image.png")

print("18 tasks ended up successfully")

#endregion