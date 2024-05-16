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
#        xyz = line.split()
#        x = int(5*n*float(xyz[1]) + n/2)
#        y = int(5*m*float(xyz[0]) + m/2)

#        if (x >= 0 and x < n and y >= 0 and y < m):
#            img[x,y] = 0

#    image = Image.fromarray(img,'L')
#    image.save("Task4Image.png")

#print("4 task ended up successfully")

def DrawLine(x0, y0, x1, y1):
    x_loop_line(x0, y0, x1, y1)

def DrawPolygon(polygon):

    line1 = lines[int(polygon[0])-1]
    line2 = lines[int(polygon[1])-1]
    line3 = lines[int(polygon[2])-1]

    DrawLine(line1[0], line1[1], line2[0], line2[1])
    DrawLine(line2[0], line2[1], line3[0], line3[1])
    DrawLine(line3[0], line3[1], line1[0], line1[1])

# Polygons
#with open("model_1.obj",'r') as f:

#    n = 1000
#    m = 1000
#    img = np.full((n, m), 255, dtype=np.uint8)
#    Data = f.readlines()
    
#    VectorsF = []
#    Polygons = []

#    for line in Data:
#        if(line[0]=='f'):
#            VectorsF.append(line[2:-1])

#    for line in VectorsF:
#        xyz = line.split()
#        numbers = []

#        for i in xyz:
#            numbers.append(i.split('/')[0])

#        Polygons.append(numbers)

#    print("5 task ended up successfully")

#    VectorsV = []
#    lines = []

#    for line in Data:
#        if(line[0]=='v' and line[1]==' '):
#            VectorsV.append(line[2:-1])

#    for line in VectorsV:
#        xyz = line.split()
#        x = int(5*n*float(xyz[1]) + n/2)
#        y = int(5*m*float(xyz[0]) + m/2)
#        lines.append([x,y])
        
#    for i in Polygons:
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

    line1 = lines[int(polygon[0])-1]
    line2 = lines[int(polygon[1])-1]
    line3 = lines[int(polygon[2])-1]

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
    
#    VectorsF = []
#    Polygons = []

#    for line in Data:
#        if(line[0]=='f'):
#            VectorsF.append(line[2:-1])

#    for line in VectorsF:
#        xyz = line.split()
#        numbers = []

#        for i in xyz:
#            numbers.append(i.split('/')[0])

#        Polygons.append(numbers)

#    VectorsV = []
#    lines = []

#    for line in Data:
#        if(line[0]=='v' and line[1]==' '):
#            VectorsV.append(line[2:-1])

#    for line in VectorsV:
#        xyz = line.split()
#        x = int(5*n*float(xyz[1]) + n/2)
#        y = int(5*m*float(xyz[0]) + m/2)
#        lines.append([x,y])
        
#    for i in Polygons:
#        ColorPolygon(i)
        
#    image = Image.fromarray(img,'RGB')
#    image.save("Task10Image.png")

#print("7-10 tasks ended up successfully")

#endregion

# Light Models
#region

def CosPolygon(polygon):
    
    light = [0, 0, 1]

    line1 = lines[int(polygon[0])-1]
    line2 = lines[int(polygon[1])-1]
    line3 = lines[int(polygon[2])-1]

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

    if (xn**2 + yn**2 + zn**2 != 0):
        return (light[0]*xn + light[1]*yn + light[2]*zn) / math.sqrt(xn**2 + yn**2 + zn**2)
    else:
        return 1

def DrawPixelWithBuffer(x, y, x0, y0, z0, x1, y1, z1, x2, y2, z2, color):

    VectorBar = Barycentering(x, y, x0, y0, x1, y1, x2, y2)

    if (VectorBar[0] >= 0 and VectorBar[1] >= 0 and VectorBar[2] >= 0):
        ZBarKoef = VectorBar[0]*z0 + VectorBar[1]*z1 + VectorBar[2]*z2

        if ZBarKoef <= ZBuffer[x][y]:
            ZBuffer[x][y] = ZBarKoef
            img[x, y] = color

def LightPolygon(polygon, color):

    line1 = lines[int(polygon[0])-1]
    line2 = lines[int(polygon[1])-1]
    line3 = lines[int(polygon[2])-1]

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
    
#    VectorsF = []
#    Polygons = []

#    for line in Data:
#        if(line[0]=='f'):
#            VectorsF.append(line[2:-1])

#    for line in VectorsF:
#        xyz = line.split()
#        numbers = []

#        for i in xyz:
#            numbers.append(i.split('/')[0])

#        Polygons.append(numbers)

#    VectorsV = []
#    lines = []

#    for line in Data:
#        if(line[0]=='v' and line[1]==' '):
#            VectorsV.append(line[2:-1])

#    for line in VectorsV:
#        xyz = line.split()
#        x = int(5*n*float(xyz[1]) + n/2)
#        y = int(5*m*float(xyz[0]) + m/2)
#        z = int(5*n*float(xyz[2]))
#        lines.append([x, y, z])
        
#    for i in Polygons:
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

    u0 = n/10
    v0 = m/5.5
    MultX = 8*n
    MultY = 8*m

    t = np.array([0.005, -0.005, 5.0])
    k = np.array([[MultX, 0, u0], [0, MultY, v0], [0, 0, 1]])

    X = np.array([x, y, z])
    X = X + t
    return np.dot(k, X)


#with open("model_1.obj",'r') as f:

#    n = 1000
#    m = 1000
#    img = np.full((n, m, 3), (255, 255, 255), dtype=np.uint8)
#    Data = f.readlines()
#    ZBuffer = [[np.inf for x in range(n)] for x in range(n)]
    
#    VectorsF = []
#    Polygons = []

#    for line in Data:
#        if(line[0]=='f'):
#            VectorsF.append(line[2:-1])

#    for line in VectorsF:
#        xyz = line.split()
#        numbers = []

#        for i in xyz:
#            numbers.append(i.split('/')[0])

#        Polygons.append(numbers)

#    VectorsV = []
#    lines = []

#    for line in Data:
#        if(line[0]=='v' and line[1]==' '):
#            VectorsV.append(line[2:-1])

#    for line in VectorsV:
#        xyz = line.split()

#        xyz = Rotate(float(xyz[0]), float(xyz[1]), float(xyz[2]))
#        xyz = Move(float(xyz[0]), float(xyz[1]), float(xyz[2]))
#        xyz = [float(xyz[1]), float(xyz[0]), float(xyz[2])]

#        lines.append([int(xyz[0]), int(xyz[1]), int(5*n*xyz[2])])
        
#    for i in Polygons:
#        if (CosPolygon(i) < 0):
#            LightPolygon(i, (150, 150, 150))
        
#    image = Image.fromarray(img,'RGB')
#    image.save("Task16Image.png")

#print("15-16 tasks ended up successfully")

#endregion

# Guro Light
#region

def guro(normalsPolygon):
    
    ilx = 0
    ily = 0
    ilz = 1
    
    nx = normalsPolygon[0]
    ny = normalsPolygon[1]
    nz = normalsPolygon[2]

    w = nx*nx + ny*ny +nz*nz
    s = ilx*ilx + ily*ily + ilz*ilz
    
    if (w != 0 and s != 0):
        return (ilx*nx + ily*ny + ilz*nz)/(math.sqrt(w)*math.sqrt(s))
    else:
        return 1

def DrawPixelGuro(x, y, x0, y0, z0, x1, y1, z1, x2, y2, z2, n):

    l1 = light[int(n[0]) - 1]
    l2 = light[int(n[1]) - 1]
    l3 = light[int(n[2]) - 1]
    VectorBar = Barycentering(x, y, x0, y0, x1, y1, x2, y2)

    if (VectorBar[0] >= 0 and VectorBar[1] >= 0 and VectorBar[2] >= 0):
        ZBarKoef = VectorBar[0]*z0 + VectorBar[1]*z1 + VectorBar[2]*z2

        if ZBarKoef <= ZBuffer[x][y]:
            ZBuffer[x][y] = ZBarKoef
            img[x, y] = (-250 + 200*(VectorBar[0]*l1 + VectorBar[1]*l2  + VectorBar[2]*l3),
                        -250 + 200*(VectorBar[0]*l1 + VectorBar[1]*l2  + VectorBar[2]*l3),
                       -250 + 200*(VectorBar[0]*l1 + VectorBar[1]*l2  + VectorBar[2]*l3))


def GuroLightPolygon(polygon, n):

    line1 = lines[int(polygon[0])-1]
    line2 = lines[int(polygon[1])-1]
    line3 = lines[int(polygon[2])-1]

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

    for i in range(rect[0], rect[1]):
        for j in range(rect[2], rect[3]):
            DrawPixelGuro(i, j, x0, y0, z0, x1, y1, z1, x2, y2, z2, n)


#with open("model_1.obj",'r') as f:

#    n = 1000
#    m = 1000
#    img = np.full((n, m, 3), (255, 255, 255), dtype=np.uint8)
#    Data = f.readlines()
#    ZBuffer = [[np.inf for x in range(n)] for x in range(n)]
    
#    VectorsF = []
#    Polygons = []
#    NormPolygons = []
#    norm = []
#    normals = []
#    light = []

#    for line in Data:
#        if(line[0]=='f'):
#            VectorsF.append(line[2:-1])

#    for line in VectorsF:
#        xyz = line.split()
#        Numbers = []
#        NormVertices = []

#        for i in xyz:
#            Numbers.append(i.split('/')[0])
#            NormVertices.append(i.split('/')[2])

#        Polygons.append(Numbers)
#        NormPolygons.append(NormVertices)

#    for line in Data:
#        if((line[0]=='v')& (line[1]=='n') & (line[2]==' ')):
#            norm.append(line[3:-1])

#    for i in norm:
#        norms = i.split()
#        n1 = float(norms[0])
#        n2 = float(norms[1])
#        n3 = float(norms[2])
#        normals.append([n1, n2, n3])

#    for i in normals:
#        light.append(guro(i))

#    VectorsV = []
#    lines = []

#    for line in Data:
#        if(line[0]=='v' and line[1]==' '):
#            VectorsV.append(line[2:-1])

#    for line in VectorsV:
#        xyz = line.split()

#        xyz = Rotate(float(xyz[0]), float(xyz[1]), float(xyz[2]))
#        xyz = Move(float(xyz[0]), float(xyz[1]), float(xyz[2]))
#        xyz = [float(xyz[1]), float(xyz[0]), float(xyz[2])]

#        lines.append([int(xyz[0]), int(xyz[1]), int(5*n*xyz[2])])
       
#    index = 0
#    for i in Polygons:
#        if (CosPolygon(i) < 0):
#            GuroLightPolygon(i, NormPolygons[index])
#        index += 1
        
#    image = Image.fromarray(img,'RGB')
#    image.save("Task17Image.png")

#print("17 tasks ended up successfully")


#endregion

# Texturing
#region

with open("model_1.obj",'r') as f:

    n = 1000
    m = 1000
    img = np.full((n, m, 3), (255, 255, 255), dtype=np.uint8)
    Data = f.readlines()
    ZBuffer = [[np.inf for x in range(n)] for x in range(n)]
    
    VectorsF = []
    Polygons = []
    NormPolygons = []
    norm = []
    normals = []
    light = []

    for line in Data:
        if(line[0]=='f'):
            VectorsF.append(line[2:-1])

    for line in VectorsF:
        xyz = line.split()
        Numbers = []
        NormVertices = []

        for i in xyz:
            Numbers.append(i.split('/')[0])
            NormVertices.append(i.split('/')[2])

        Polygons.append(Numbers)
        NormPolygons.append(NormVertices)

    for line in Data:
        if((line[0]=='v')& (line[1]=='n') & (line[2]==' ')):
            norm.append(line[3:-1])

    for i in norm:
        norms = i.split()
        n1 = float(norms[0])
        n2 = float(norms[1])
        n3 = float(norms[2])
        normals.append([n1, n2, n3])

    for i in normals:
        light.append(guro(i))

    VectorsV = []
    lines = []

    for line in Data:
        if(line[0]=='v' and line[1]==' '):
            VectorsV.append(line[2:-1])

    for line in VectorsV:
        xyz = line.split()

        xyz = Rotate(float(xyz[0]), float(xyz[1]), float(xyz[2]))
        xyz = Move(float(xyz[0]), float(xyz[1]), float(xyz[2]))
        xyz = [float(xyz[1]), float(xyz[0]), float(xyz[2])]

        lines.append([int(xyz[0]), int(xyz[1]), int(5*n*xyz[2])])
       
    index = 0
    for i in Polygons:
        if (CosPolygon(i) < 0):
            GuroLightPolygon(i, NormPolygons[index])
        index += 1
        
    image = Image.fromarray(img,'RGB')
    image.save("Task18Image.png")

print("18 tasks ended up successfully")


#endregion