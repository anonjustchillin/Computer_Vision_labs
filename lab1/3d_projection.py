import numpy as np
from graphics import *

width, height = 800, 600
line_colors = ["red", "orange", "yellow", "green", "blue", "purple"]
col_index = 0


def setWindow(w, h, color, title):
    win = GraphWin(title, w, h)
    win.setBackground(color)
    # win.setCoords(-300, -300, 300, 300)
    return win


def moveXYZ(object, l, m, n, x=1, y=1, z=1):
    moving = np.array([
        [x, 0, 0, l],
        [0, y, 0, m],
        [0, 0, z, n],
        [0, 0, 0, 1]
    ])

    total = object.dot(moving.T)
    return total


def rotateX(object, degree):
    a = np.radians(degree)
    Rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(a), -np.sin(a), 0],
                   [0, np.sin(a), np.cos(a), 0],
                   [0, 0, 0, 1]])

    total = object.dot(Rx.T)
    return total


def rotateY(object, degree):
    a = np.radians(degree)
    Ry = np.array([[np.cos(a), 0, np.sin(a), 0],
                   [0, 1, 0, 0],
                   [-np.sin(a), 0, np.cos(a), 0],
                   [0, 0, 0, 1]])

    total = object.dot(Ry.T)
    return total


def rotateZ(object, degree):
    a = np.radians(degree)
    Rz = np.array([[np.cos(a), -np.sin(a), 0, 0],
                   [np.sin(a), np.cos(a), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    total = object.dot(Rz.T)
    return total


def setColor():
    global col_index
    color = line_colors[col_index]
    if col_index == len(line_colors)-1:
        col_index = 0
    else:
        col_index += 1

    return color


def drawFace(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, color):
    curr_obj = Polygon(Point(p1x, p1y),
                       Point(p2x, p2y),
                       Point(p3x, p3y),
                       Point(p4x, p4y))
    curr_obj.setWidth(2)
    curr_obj.setOutline(color)
    curr_obj.draw(win)


def showObject(object, color=""):
    if color=="":
        color = setColor()

    Ax = object[0, 0]; Ay = object[0, 1]
    Bx = object[1, 0]; By = object[1, 1]
    Cx = object[2, 0]; Cy = object[2, 1]
    Dx = object[3, 0]; Dy = object[3, 1]

    Ex = object[4, 0]; Ey = object[4, 1]
    Fx = object[5, 0]; Fy = object[5, 1]
    Gx = object[6, 0]; Gy = object[6, 1]
    Hx = object[7, 0]; Hy = object[7, 1]

    drawFace(Ax, Ay, Bx, By, Cx, Cy, Dx, Dy, color)
    drawFace(Ex, Ey, Fx, Fy, Gx, Gy, Hx, Hy, color)
    drawFace(Ax, Ay, Bx, By, Fx, Fy, Ex, Ey, color)
    drawFace(Dx, Dy, Cx, Cy, Gx, Gy, Hx, Hy, color)
    drawFace(Bx, By, Fx, Fy, Gx, Gy, Cx, Cy, color)
    drawFace(Ax, Ay, Ex, Ey, Hx, Hy, Dx, Dy, color)


w, h, d = 50, 100, 150
obj = np.array([ [0, 0, 0, 1],
                  [w, 0, 0, 1],
                  [w, h, 0, 1],
                  [0, h, 0, 1],
                  [0, 0, d, 1],
                  [w, 0, d, 1],
                  [w, h, d, 1],
                  [0, h, d, 1]])

l = (width/2)-float(np.mean(obj[:, 0]))
m = (height/2)-float(np.mean(obj[:, 0]))
n = m


### Статичний паралелепіпед
win = setWindow(width, height, "white", "Паралелепіпед")

print('Статичний паралелепіпед')
print(obj)
showObject(obj, "black")

win.getMouse()
win.close()

print()

## Паралелепіпед в центрі
win = setWindow(width, height, "white", "Паралелепіпед")
moved_obj = moveXYZ(obj, l, m, n)
print('Паралелепіпед в центрі')
print(moved_obj)
showObject(moved_obj, "black")

win.getMouse()
win.close()

print()

### Ротація по x
win = setWindow(width, height, "white", "Паралелепіпед")

rotatedX = rotateX(obj, 45)
print('Ротація по x')
print(rotatedX)
rotatedX = moveXYZ(rotatedX, l, m, n, 1, 1, 0)
print('Переміщення в центр')
print(rotatedX)

showObject(rotatedX, "black")

win.getMouse()
win.close()

print()

### Ротація по y
win = setWindow(width, height, "white", "Паралелепіпед")

rotatedY = rotateY(obj, 45)
print('Ротація по y')
print(rotatedY)
rotatedY = moveXYZ(rotatedY, l, m, n, 1, 1, 0)
print('Переміщення в центр')
print(rotatedY)

showObject(rotatedY, "black")

win.getMouse()
win.close()

print()

### Ротація по z
win = setWindow(width, height, "white", "Паралелепіпед")

rotatedZ = rotateZ(obj, 120)
print('Ротація по z')
print(rotatedZ)
rotatedZ = moveXYZ(rotatedZ, l, m, n, 1, 1, 0)
print('Переміщення в центр')
print(rotatedZ)

showObject(rotatedZ, "black")

win.getMouse()
win.close()

print()

## Ротація по xyz
win = setWindow(width, height, "white", "Паралелепіпед")

rotated = rotateX(obj, 45)
print('Ротація по x')
print(rotated)
rotated = rotateY(rotated, 45)
print('Ротація по y')
print(rotated)
rotated = rotateZ(rotated, 45)
print('Ротація по z')
print(rotated)

rotated = moveXYZ(rotated, l, m, n, 1, 1, 0)

print('Переміщення в центр')
print(rotated)

showObject(rotated, "black")

win.getMouse()
win.close()

print()

# Обертання
win = setWindow(width, height, "white", "Паралелепіпед")
win.setCoords(-300, -300, 300, 300)

obj2 = obj.copy()

deg = 10
cycles = int(360/np.abs(deg))*5

for i in range(cycles):
    time.sleep(0.1)
    showObject(obj2, "white")

    obj2 = rotateX(obj2, deg)
    obj2 = rotateY(obj2, deg)
    obj2 = rotateZ(obj2, deg)

    showObject(obj2)

print('Закінчено')
win.getMouse()
win.close()
