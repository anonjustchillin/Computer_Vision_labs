import numpy as np
from graphics import *


def setWindow(w, h, color, title):
    win = GraphWin(title, w, h)
    win.setBackground(color)
    win.setCoords(-300, -300, 300, 300)
    return win


def action(p, T):
    total = p.dot(T.T)
    curr_p = [Point(x, y) for x, y, _ in total]
    p = np.array([
        [curr_p[0].x, curr_p[0].y, 1],
        [curr_p[1].x, curr_p[1].y, 1],
        [curr_p[2].x, curr_p[2].y, 1],
        [curr_p[3].x, curr_p[3].y, 1],
        [curr_p[4].x, curr_p[4].y, 1],
        [curr_p[5].x, curr_p[5].y, 1]
    ])
    return p, curr_p



#### Статичний шестикутник
height = 600
width = 600
win = setWindow(width, height, "white", "Статичний шестикутник")

r = 20
points = [Point(0,0),
          Point(0,0),
          Point(0,0),
          Point(0,0),
          Point(0,0),
          Point(0,0)]

for i in range(0, 6):
    points[i] = Point(r*np.cos(i*np.pi/3), r*np.sin(i*np.pi/3))

x1 = points[0].x; y1 = points[0].y
x2 = points[1].x; y2 = points[1].y
x3 = points[2].x; y3 = points[2].y
x4 = points[3].x; y4 = points[3].y
x5 = points[4].x; y5 = points[4].y
x6 = points[5].x; y6 = points[5].y

hexagon = Polygon(points)
hexagon.draw(win)
print("Координати шестикутника")
print(hexagon.getPoints())

win.getMouse()
win.close()

#### Переміщення шестикутника
win = setWindow(width, height, "white", "Шестикутник рухається")
hexagon = Polygon(points)
hexagon.draw(win)

p = np.array([
    [x1, y1, 1],
    [x2, y2, 1],
    [x3, y3, 1],
    [x4, y4, 1],
    [x5, y5, 1],
    [x6, y6, 1]
])

moving = np.array([
    [1, 0, r],
    [0, 1, -r],
    [0, 0, 1],
])

stop = int(width/r)
for i in range(stop):
    time.sleep(0.3)

    hexagon.setOutline("white")

    p, new_points = action(p, moving)

    hexagon = Polygon(*new_points)
    hexagon.draw(win)


print('Закінчено')
win.getMouse()
win.close()

#### Обертання шестикутника
win = setWindow(width, height, "white", "Шестикутник обертається")
hexagon = Polygon(points)
hexagon.draw(win)

p = np.array([
    [x1, y1, 1],
    [x2, y2, 1],
    [x3, y3, 1],
    [x4, y4, 1],
    [x5, y5, 1],
    [x6, y6, 1]
])

angle = 60
R = (3/14*angle)/180

rotating = np.array([
    [np.cos(R), np.sin(R), 0],
    [-np.sin(R), np.cos(R), 0],
    [0, 0, 1],
])

stop = int(width/r)
for i in range(stop):
    time.sleep(0.3)
    hexagon.setOutline("white")

    p, new_points = action(p, rotating)

    hexagon = Polygon(*new_points)
    hexagon.draw(win)

print('Закінчено')
win.getMouse()
win.close()


#### Масштабування шестикутника
win = setWindow(width, height, "white", "Шестикутник збільшується")
hexagon = Polygon(points)
hexagon.draw(win)

p = np.array([
    [x1, y1, 1],
    [x2, y2, 1],
    [x3, y3, 1],
    [x4, y4, 1],
    [x5, y5, 1],
    [x6, y6, 1]
])

size = 0.1

scaling = np.array([
    [r*size, 0, 0],
    [0, r*size, 0],
    [0, 0, 1],
])

#stop = int(width/r)
for i in range(6):
    time.sleep(0.3)
    hexagon.setOutline("white")

    p, new_points = action(p, scaling)

    hexagon = Polygon(*new_points)
    hexagon.draw(win)

print('Закінчено')
win.getMouse()
win.close()


# Все разом
#win = setWindow(width, height, "white", "Шестикутник")
#hexagon = Polygon(points)
#hexagon.draw(win)
#
#p = np.array([
#    [x1, y1, 1],
#    [x2, y2, 1],
#    [x3, y3, 1],
#    [x4, y4, 1],
#    [x5, y5, 1],
#    [x6, y6, 1]
#])
#
#stop = int(width/r)
#for i in range(stop):
#    time.sleep(0.3)
#    #hexagon.setOutline("white")
#
#    p, _ = action(p, moving)
#    p, new_points = action(p, rotating)
#    #p, new_points = action(p, scaling)
#
#    hexagon = Polygon(*new_points)
#    hexagon.draw(win)
#
#print('Закінчено')
#win.getMouse()
#win.close()
