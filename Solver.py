import cv2
import numpy as np
from PIL import Image
from queue import *
import math
import random


class Piece:
    def __init__(self, im, mk):
        self.image = im
        self.mask = mk
        self.outline = []
        self.bwmask = mk
        self.maleorgans = []
        self.femaleorgans = []
        self.femalelist = []
        self.malelist = []
        self.femalediff = []
        self.malediff = []
        self.femalecolors = []
        self.malecolors = []
        self.offset = [0,0]
        self.rotation = 0

    def __eq__(self, other):
        if self.outline == other.outline:
            return True
        return False

class Match:
    def __init__(self, fpeice, mpeice, forgan, morgan, diff):
        self.fpeice = fpeice
        self.mpeice = mpeice
        self.forgan = forgan
        self.morgan = morgan
        self.diff = diff

    def __lt__(self, other):
        if self.diff < other.diff:
            return True
        return False

    def __str__(self):
        return "[ " + str(self.fpeice) + ", " + str(self.mpeice) + ", " + str(self.forgan) + ", " + str(self.morgan) + ", " + str(self.diff) + " ]"


class Combo:
    def __init__(self, image, fpiece, fpart, fangle, mpiece, mpart, mangle, offset):

        fpiece.offset = offset
        fpiece.rotation = fangle
        mpiece.rotation = mangle

        self.image = image
        self.peices = [fpiece, mpiece]
        self.pque = Queue()
        self.pque.put(fpiece)
        #self.pque.put(mpiece)

    def addpiece(self, piece, ppart, pgender, ipiece, ipart):
        """
        addes peices to a single image
        :param piece: new peice to add
        :param ppart: part of peice
        :param pgender: gender of ppart
        :param ipiece: peice already in iname
        :param ipart: part of ipiece
        :return: updates combo class
        """
        #find point in combo image
        if pgender == 0:
            for p in self.peices:
                if ipiece.outline == p.outline:
                    for o in p.maleorgans:
                        if o == ipart:
                            newpoint = findpoint(ipiece, ipart, ipiece.rotation, 0)
                            newpoint[0] += ipiece.offset[0]
                            newpoint[1] += ipiece.offset[1]
        else:
            for p in self.peices:
                if ipiece.outline == p.outline:
                    for o in p.femaleorgans:
                        if o == ipart:
                            newpoint = findpoint(ipiece, ipart, ipiece.rotation, 0)
                            newpoint[0] += ipiece.offset[0]
                            newpoint[1] += ipiece.offset[1]
        #newpart = location in full image
        w2, h2 = piece.image.size
        #find left or up
        direction = 0 # 0 for left 1 for up
        if newpoint[0] > newpoint[1]:
            direction = 0
        else:
            direction = 1
        centerxoffset = newpoint[0] - w2 // 2
        centeryoffset = newpoint[1] - h2//2
        w, h = self.image.size
        if direction == 0:
            newangle = findangle((w2 // 2 +w, h2 // 2+centeryoffset), (ppart[0]+w, ppart[1] + centeryoffset), newpoint)
            ci = Image.new('RGB', (w + w2, h), (255, 255, 255))
            if ppart[1] > h2//2:
                newangle = 360 -newangle
        else:
            newangle = findangle((w2 // 2 + centerxoffset, h2 // 2 + h), (ppart[0] + centerxoffset, ppart[1] + h),
                                 newpoint)
            ci = Image.new('RGB', (w, h + h2), (255, 255, 255))
            if ppart[0] > w2//2:
                newangle = 360 - newangle
        #if ppart[1] + centeryoffset < h2 // 2 + centeryoffset:
           #newangle = 360 - newangle
        w1, h1 = piece.image.rotate(newangle, expand=True).size
        porgan = findpoint(piece, ppart, newangle, 0)
        print(newpoint, porgan)
        xoffset = newpoint[0] - porgan[0]
        yoffset = newpoint[1] - porgan[1]
        print("offset:", xoffset, yoffset)
        w2, h2 = self.image.size


        ci.paste(self.image, (0, 0))
        ci.paste(piece.image.rotate(newangle, expand=True), (0 + xoffset, 0 + yoffset),
                 piece.bwmask.rotate(newangle, expand=True))
        ci.save("combo.jpg")
        self.image = ci
        piece.rotation = newangle
        piece.offset = [xoffset, yoffset]
        self.peices.append(piece)
        self.pque.put(piece)
        #ci.show()

def getPieces(filename):
    """
    applies a mask to the input image to remove green screen then ques through the image searching for peices
    :param filename: filename containt an image of peices on a green screen
    :return: array of peices found
    """
    inputimage = cv2.imread(filename)


    #inputimage = cv2.resize(inputimage, (4032, 3024))

    u_green = np.array([120, 255, 95])#np.array([100, 255, 100])
    l_green = np.array([0, 100, 0])#np.array([0,90,0])
    mask = cv2.inRange(inputimage, l_green, u_green)
    #cv2.imwrite("mask.jpg", mask)


    masked_image = np.copy(inputimage)
    #cv2.imwrite("pre-mask.jpg", masked_image)
    masked_image[mask != 0] = [0, 0, 255]
    masked_image[mask == 0] = [0,255,0]
    cv2.imwrite("post-mask.jpg", masked_image)
    m = Image.fromarray(masked_image)

    m.save("post-mask.BMP")

    img = Image.open("post-mask.BMP")
    og = Image.open(filename)
    w, h = img.size
    print("Width: ", w, "\tHeight ", h)
    pixles = img.load()
    #pixles = masked_image
    piecesarr = []



    for i in range(1, w - 1):
        for j in range(1, h - 1):
            r, g, b = pixles[i, j]
            #print(r,g,b)
            if b - (r + g) != 255 and r - (g + b) != 255:
                fillq = Queue()
                maxx = -1
                minx = w + 1
                maxy = -1
                miny = h + 1
                fillq.put((i, j))
                pixles[i, j] = (255, 0, 0)
                while not fillq.empty():
                    x, y = fillq.get()
                    # get min/max
                    if x < minx:
                        minx = x
                    if x > maxx:
                        maxx = x
                    if y < miny:
                        miny = y
                    if y > maxy:
                        maxy = y

                    # check left
                    if x-1 > 0:
                        r, g, b = pixles[x - 1, y]
                        if b - (r + g) != 255 and r - (g + b) != 255 :
                            fillq.put((x - 1, y))
                            pixles[x - 1, y] = (255, 0, 0)
                    # check right
                    if x + 1 < w:
                        r, g, b = pixles[x + 1, y]
                        if b - (r + g) != 255 and r - (g + b) != 255 :
                            fillq.put((x + 1, y))
                            pixles[x + 1, y] = (255, 0, 0)
                    # check up
                    if y-1 > 0:
                        r, g, b = pixles[x, y - 1]
                        if b - (r + g) != 255 and r - (g + b) != 255 :
                            fillq.put((x, y - 1))
                            pixles[x, y - 1] = (255, 0, 0)
                    # check down
                    if y + 1 < h:
                        r, g, b = pixles[x, y + 1]
                        if b - (r + g) != 255 and r - (g + b) != 255:
                            fillq.put((x, y + 1))
                            pixles[x, y + 1] = (255, 0, 0)

                #print("MaxX: ", maxx, " | MinX: ", minx, " | MaxY: ", maxy, " | MinY: ", miny)
                # piecearr = ogpix[minx:maxx, miny:maxy]
                if(maxx-minx >40 or maxy-miny >40):
                    newpiece = og.crop((minx - 3, miny - 3, maxx + 3, maxy + 3))
                    newmask = img.crop((minx - 3, miny - 3, maxx + 3, maxy + 3))
                    # newpiece.show()
                    p1 = Piece(newpiece, newmask)
                    piecesarr.append(p1)
    print("number of Pieces:", len(piecesarr))


    return piecesarr


def CleanUp(pieces):
    """
    i dont think this does anything
    :param pieces:
    :return:
    """
    for p in pieces:
        w, h = p.mask.size
        pixles = p.mask.load()
        for i in range(w):
            for j in range(h):
                if pixles[i,j] != (255, 0, 0):
                    pixles[i,j] = (0, 0, 255)
    return pieces


def Outline(pieces):
    """
    creates an array of all point in the outline of a peice
    :param pieces: array of all peices
    :return: internally updates peices
    """
    for p in pieces:
        w, h = p.mask.size
        pixles = p.mask.load()
        outline = []
        for i in range(1, w - 1):
            for j in range(1, h - 1):
                r, g, b = pixles[i, j]
                if b - (r + g) == 255:
                    # check left
                    r, g, b = pixles[i - 1, j]
                    if b - (r + g) != 255 and g - (b + r) != 255:
                        pixles[i-1, j] = (0, 255, 0)
                        outline.append((i - 1, j))

                    # check right
                    r, g, b = pixles[i + 1, j]
                    if b - (r + g) != 255 and g - (r + b) != 255:
                        pixles[i+1, j] = (0, 255, 0)
                        outline.append((i+1, j))

                    # check up
                    r, g, b = pixles[i, j - 1]
                    if b - (r + g) != 255 and g - (b + r) != 255:
                        pixles[i, j-1] = (0, 255, 0)
                        outline.append((i, j-1))
                    # check down
                    r, g, b = pixles[i, j + 1]
                    if b - (r + g) != 255 and g - (b + r) != 255:
                        pixles[i, j+1] = (0, 255, 0)
                        outline.append((i, j+1))
        p.outline = outline
    return pieces





def findcenter(peice):
    """
    finds the center of mass of a peice
    :param peice: a single peice
    :return: x, y cords of the center
    """
    xsum = 0
    ysum = 0
    for point in peice.outline:
        xsum += point[0]
        ysum += point[1]
    return xsum//len(peice.outline), ysum//len(peice.outline)




def findfemalesexorgans(peice):
    """
    find male organs by looking for points furthest from the center
    :param peice: a single peice
    :return: internalyy updates peice
    """
    xsum = 0
    ysum = 0
    for point in peice.outline:
        xsum += point[0]
        ysum += point[1]
    cx = xsum // len(peice.outline)
    cy = ysum // len(peice.outline)


    #find above
    minabove = 99999
    pabove = (0, 0)
    minbelow = 99999
    pbelow = (0, 0)
    minleft = 99999
    pleft = (0, 0)
    minright = 99999
    pright = (0, 0)
    for point in peice.outline:
        distance = ((cx - point[0])**2 + (cy - point[1])**2)**.5
        if point[1] < cy and distance < minabove:
            minabove = distance
            pabove = point
        elif point[1] > cy and distance < minbelow:
            minbelow = distance
            pbelow = point
        elif point[0] < cx and distance < minleft:
            minleft = distance
            pleft = point
        elif point[0] > cx and distance < minright:
            minright = distance
            pright = point
    #peice.mask.putpixel(pabove, (255, 0, 255))
    #peice.mask.putpixel(pbelow, (255, 0, 255))
    #peice.mask.putpixel(pleft, (255, 0, 255))
    #peice.mask.putpixel(pright, (255, 0, 255))
    #peice.mask.show()
    print(pabove, pbelow, pleft, pright)
    peice.femaleorgans.append(pabove)
    peice.femaleorgans.append(pbelow)
    peice.femaleorgans.append(pleft)
    peice.femaleorgans.append(pright)

def findmalesexorgans(peice):
    """
    locates female organs by looking for points on the outline closest to the center
    :param peice: a single peice
    :return: internally updates peice
    """
    xsum = 0
    ysum = 0
    for point in peice.outline:
        xsum += point[0]
        ysum += point[1]
    cx = xsum // len(peice.outline)
    cy = ysum // len(peice.outline)


    #find above
    maxabove = 0
    pabove = (0, 0)
    maxbelow = 0
    pbelow = (0, 0)
    maxleft = 0
    pleft = (0, 0)
    maxright = 0
    pright = (0, 0)
    for point in peice.outline:
        distance = ((cx - point[0])**2 + (cy - point[1])**2)**.5
        if point[1] < cy and distance > maxabove:
            maxabove = distance
            pabove = point
        elif point[1] > cy and distance > maxbelow:
            maxbelow = distance
            pbelow = point
        elif point[0] < cx and distance > maxleft:
            maxleft = distance
            pleft = point
        elif point[0] > cx and distance > maxright:
            maxright = distance
            pright = point

    print(pabove, pbelow, pleft, pright)
    peice.maleorgans.append(pabove)
    peice.maleorgans.append(pbelow)
    peice.maleorgans.append(pleft)
    peice.maleorgans.append(pright)


def amputate(peice):
    """
    attempts to remove unneeded or duplicate peices
    :param peice: single peice
    :return: internally updates peice
    """
    # female
    cx, cy = findcenter(peice)
    remove = set()
    for point in peice.femaleorgans:
        for p2 in peice.femaleorgans:
            if point != p2:
                xdis = abs(point[0] - p2[0])
                ydis = abs(point[1] - p2[1])
                if xdis < 30 and ydis < 30:
                    d1 = ((cx - point[0]) ** 2 + (cy - point[1]) ** 2) ** .5
                    d2 = ((cx - p2[0]) ** 2 + (cy - p2[1]) ** 2) ** .5
                    if d1 < d2:
                        remove.add(p2)
                    else:
                        remove.add(point)
    for p in remove:
        if len(peice.maleorgans) > 1:
            peice.femaleorgans.remove(p)
            peice.mask.putpixel(p, (0, 255, 0))
    print(peice.femaleorgans)
    remove = set()
    for point in peice.maleorgans:
        for p2 in peice.maleorgans:
            if point != p2:
                xdis = abs(point[0]-p2[0])
                ydis = abs(point[1]-p2[1])
                if xdis < 30 and ydis < 30:
                    d1 = ((cx - point[0]) ** 2 + (cy - point[1]) ** 2) ** .5
                    d2 = ((cx - p2[0]) ** 2 + (cy - p2[1]) ** 2) ** .5
                    if d1 > d2:
                        remove.add(p2)
                    else:
                        remove.add(point)
    for p in remove:
        if len(peice.maleorgans) > 1:
            peice.maleorgans.remove(p)
            peice.mask.putpixel(p, (0, 255, 0))

    print(peice.maleorgans)


def outlineque(que, source, xdiff, ydiff, peice, gender):
    """
    helper function for growing the organs
    :param que:
    :param source:
    :param xdiff:
    :param ydiff:
    :param peice:
    :param gender:
    :return:
    """
    if peice.mask.getpixel((source[0] + xdiff, source[1] + ydiff)) == (0, 255, 0):
        que.put((source[0] + xdiff, source[1] + ydiff))
    return que


def groworgans(peice):
    """
    expands the point of a part to go along the outline of the peice by a size of organsize
    :param peice: a sing peice object
    :return: internally updates peice
    """
    organsize = 100
    for organ in peice.femaleorgans:
        fillque = Queue()
        fillque.put(organ)

        growth = []
        while not fillque.empty():
            #print(fillque)
            point = fillque.get()
            dist = ((point[0] - organ[0]) ** 2 + (point[1] - organ[1]) ** 2) ** .5
            if len(growth) <= organsize and point not in growth:
                # check up
                fillque = outlineque(fillque, point, 0, -1, peice, 0)
                # check down
                fillque = outlineque(fillque, point, 0, 1, peice, 0)
                # check left
                fillque = outlineque(fillque, point, -1, 0, peice, 0)
                # check right
                fillque = outlineque(fillque, point, 1, 0, peice, 0)
                # check up left
                fillque = outlineque(fillque, point, -1, -1, peice, 0)
                # check up right
                fillque = outlineque(fillque, point, 1, -1, peice, 0)
                # check down left
                fillque = outlineque(fillque, point, -1, 1, peice, 0)
                # check down right
                fillque = outlineque(fillque, point, 1, 1, peice, 0)
                growth.append(point)
        if (len(growth) > 3):
            peice.femalelist.append(growth)
        else:
            peice.femaleorgans.remove(organ)
    for organ in peice.maleorgans:
        fillque = Queue()
        fillque.put(organ)
        growth = []
        while not fillque.empty():
            point = fillque.get()
            dist = ((point[0] - organ[0]) ** 2 + (point[1] - organ[1]) ** 2) ** .5
            if len(growth) <= organsize and point not in growth:
                # check up
                fillque = outlineque(fillque, point, 0, -1, peice, 1)
                # check down
                fillque = outlineque(fillque, point, 0, 1, peice, 1)
                # check left
                fillque = outlineque(fillque, point, -1, 0, peice, 1)
                # check right
                fillque = outlineque(fillque, point, 1, 0, peice, 1)
                # check up left
                fillque = outlineque(fillque, point, -1, -1, peice, 1)
                # check up right
                fillque = outlineque(fillque, point, 1, -1, peice, 1)
                # check down left
                fillque = outlineque(fillque, point, -1, 1, peice, 1)
                # check down right
                fillque = outlineque(fillque, point, 1, 1, peice, 1)
                growth.append(point)
        if (len(growth) > 3):
            peice.malelist.append(growth)
        else:
            peice.maleorgans.remove(organ)
    #print(peice.femalelist)
    #print(peice.malelist)

def optimizeorgans(peice):
    """
    i dont think this does anything
    :param peice:
    :return:
    """
    xsum = 0
    ysum = 0
    for point in peice.outline:
        xsum += point[0]
        ysum += point[1]
    cx = xsum // len(peice.outline)
    cy = ysum // len(peice.outline)
    # male
    for i, organ in enumerate(peice.malelist):
        bestpoint = (0, 0)
        maxdistance = 0
        for point in organ:
            dist = ((point[0] - cx) ** 2 + (point[1] - cy) ** 2) ** .5
            if dist > maxdistance:
                bestpoint = point
                maxdistance = dist
        peice.maleorgans[i] = bestpoint

    # female
    for i, organ in enumerate(peice.femalelist):
        bestpoint = (0, 0)
        maxdistance = 9999
        for point in organ:
            dist = ((point[0] - cx) ** 2 + (point[1] - cy) ** 2) ** .5
            if dist < maxdistance:
                bestpoint = point
                maxdistance = dist
        peice.femaleorgans[i] = bestpoint




def findangle(center, organ, location):
    """
    uses law of cosines to find the rotation angle
    :param center: center of the peice
    :param organ: the organ point
    :param location: oint of where to rotate the organ to
    :return: angle of rotation still needs to be updateed to be counter clockwise
    """
    leg1 = ((center[0] - organ[0]) ** 2 + (center[1] - organ[1]) ** 2) ** .5
    leg2 = ((center[0] - location[0]) ** 2 + (center[1] - location[1]) ** 2) ** .5
    leg3 = ((location[0] - organ[0]) ** 2 + (location[1] - organ[1]) ** 2) ** .5
    #print(leg1, leg2, leg3)
    return math.degrees(math.acos((leg1**2+leg2**2-leg3**2)/(2 * leg1 * leg2)))

def findpoint(peice, organ, angle, gender):
    """
    find the new location of an organ after rotation
    :param peice: the working peice
    :param organ: which organ yo are looking for
    :param angle: angle ot rotation
    :param gender: unused
    :return: the new xy cords of the organ after rotation
    """
    copy = peice.mask.copy()
    copy.putpixel(organ, (255, 0, 255))
    copy.putpixel((organ[0] + 1, organ[1]), (255, 0, 255))
    copy.putpixel((organ[0] - 1, organ[1]), (255, 0, 255))
    copy.putpixel((organ[0], organ[1] + 1), (255, 0, 255))
    copy.putpixel((organ[0], organ[1] - 1), (255, 0, 255))

    px = copy.rotate(angle, expand=True).load()
    w, h = copy.rotate(angle, expand=True).size
    for i in range(w):
        for j in range(h):
            if px[i, j] == (255, 0, 255):
                px[i, j] = (0, 255, 0)
                return [i, j]




def combinep(p1, p1f, p2, p2m):
    """
    take two images and combines then so that they fit togeather
    :param p1: peice with female part
    :param p1f: part of p1
    :param p2: peice with male part
    :param p2m: part of p2
    :return: an initilized combo class to begin add all images
    """
    #p1 is timage with female
    #p1 if point of female
    #p2 is male
    w1, h1 = p1.image.size
    w2, h2 = p2.image.size
    fangle = findangle((w1//2, h1//2), p1f, (w1//2, 0)) # point pussy down
    mangle = findangle((w2//2, h2//2), p2m, (w2//2, h2)) # point penis up
    if p1f[0] < w1//2:
        fangle = 360 - fangle
    if p2m[0] > w2//2:
        mangle = 360 - mangle
    print(fangle, mangle)
    w1, h1 = p1.image.rotate(fangle, expand=True).size
    w2, h2 = p2.image.rotate(mangle, expand=True).size
    p1organ = findpoint(p1, p1f, fangle, 0)
    p2organ = findpoint(p2, p2m, mangle, 1)
    print("p1 old: ", p1f, " p1: ", p1organ, "  p2 old: ", p2m, " p2: ", p2organ)

    p2xoffset = 0
    p2yoffset = 0
    p1xoffset = p2organ[0] - p1organ[0]
    p1yoffset = p2organ[1] - p1organ[1]
    print("offset:", p2xoffset, p2yoffset)
    ci = Image.new('RGB', (w1+w2, h1+h2), (255, 255, 255))
    ci.paste(p1.image.rotate(fangle, expand=True), (0+p1xoffset, 0+p1yoffset), p1.bwmask.rotate(fangle, expand=True))
    ci.paste(p2.image.rotate(mangle, expand=True), (0+p2xoffset, 0+p2yoffset), p2.bwmask.rotate(mangle, expand=True))
    ci.save("combo.jpg")
    combo = Combo(ci, p1, p1f, fangle, p2, p2m, mangle, [p1xoffset, p1yoffset])
    #ci.show()
    return combo


def combinep2(p1, p1f, p2, p2m):
    #p1 is timage with female
    #p1 if point of female
    #p2 is male
    w1, h1 = p1.image.size
    w2, h2 = p2.image.size
    fangle = findangle((w1//2, h1//2), p1f, (w1//2, h1)) # point pussy down
    mangle = findangle((w2//2, h2//2), p2m, (w2//2, 0)) # point penis up
    if p1f[0] < w1//2:
        fangle = 360 - fangle
    if p2m[0] > w2//2:
        mangle = 360 - mangle
    print(fangle, mangle)
    w1, h1 = p1.image.rotate(fangle, expand=True).size
    w2, h2 = p2.image.rotate(mangle, expand=True).size
    p1organ = findpoint(p1, p1f, fangle, 0)
    p2organ = findpoint(p2, p2m, mangle, 1)
    print("p1 old: ", p1f, " p1: ", p1organ, "  p2 old: ", p2m, " p2: ", p2organ)

    p2xoffset = 0
    p2yoffset = 0
    p1xoffset = p1organ[0] - p2organ[0]
    p1yoffset = p1organ[1] - p2organ[1]
    print("offset:", p2xoffset, p2yoffset)
    ci = Image.new('RGB', (w1+w2, h1+h2), (255, 255, 255))
    ci.paste(p1.image.rotate(fangle, expand=True), (0+p2xoffset, 0+p2yoffset), p1.bwmask.rotate(fangle, expand=True))
    ci.paste(p2.image.rotate(mangle, expand=True), (0+p1xoffset, 0+p1yoffset), p2.bwmask.rotate(mangle, expand=True))
    ci.save("combo.jpg")
    combo = Combo(ci, p2, p2m, mangle, p1, p1f, fangle, [p1xoffset, p1yoffset])
    #ci.show()
    return combo



def bwmask(Pieces):
    """
    creates a black and white maske for all peices
    :param Pieces: array of all peices
    :return: edits class internally
    """
    for p in Pieces:
        w, h = p.mask.size
        maskdata = p.mask.load()

        ci = Image.new('1', (w, h), 0)
        bwdata = ci.load()
        for i in range(w):
            for j in range(h):
                if maskdata[i, j] == (255, 0, 0) or maskdata[i, j] == (0, 255, 0):
                    bwdata[i, j] = 1
        p.bwmask = ci
    return Pieces

def makediffrence(Pieces):
    """
    creates a score for the shape of the peice
    :param Pieces: array of all peices
    :return: eddits the class internally
    """
    for p in Pieces:
        #
        for female in p.femalelist:
            diff = []
            for i in range(1, len(female)-2):
                xdiff = female[i-1][0] + female[i+1][0] - female[i][0]
                ydiff = female[i-1][1] + female[i+1][1] - female[i][1]
                diff.append(xdiff + ydiff)
            p.femalediff.append(diff)

        for male in p.malelist:
            diff = []
            for i in range(1, len(male) - 2):
                xdiff = male[i - 1][0] + male[i + 1][0] - male[i][0]
                ydiff = male[i - 1][1] + male[i + 1][1] - male[i][1]
                diff.append(int(xdiff + ydiff))
            p.malediff.append(diff)

def getoverlap(p1, p1f, p2, p2m):
    """
    gets the number of overlapped pixles when combining two peices
    :param p1: peice class object with female part
    :param p1f: the part of p1
    :param p2: peice class object with male part
    :param p2m: the part of p2
    :return: the number of overlapped pixles
    """
    w1, h1 = p1.image.size
    w2, h2 = p2.image.size
    fangle = findangle((w1 // 2, h1 // 2), p1f, (w1 // 2, h1))  # point pussy down
    mangle = findangle((w2 // 2, h2 // 2), p2m, (w2 // 2, 0))  # point penis up
    if p1f[0] > w1 // 2:
        fangle = 360 - fangle
    if p2m[0] < w2 // 2:
        mangle = 360 - mangle
    #print(fangle, mangle)

    p1organ = findpoint(p1, p1f, fangle, 0)
    p2organ = findpoint(p2, p2m, mangle, 1)
    #print("p1 old: ", p1f, " p1: ", p1organ, "  p2 old: ", p2m, " p2: ", p2organ)

    p2xoffset = p1organ[0] - p2organ[0]
    p2yoffset = p1organ[1] - p2organ[1]
    p1xoffset = 0
    p1yoffset = 0
    overlap = 0
    p1data = p1.bwmask.copy().rotate(fangle, expand=True).load()
    p2data = p2.bwmask.copy().rotate(mangle, expand=True).load()
    w1, h1 = p1.bwmask.copy().rotate(fangle, expand=True).size
    w2, h2 = p2.bwmask.copy().rotate(mangle, expand=True).size
    for x in range(0, w1-p2xoffset, 5):
        for y in range(0, h1-p2yoffset, 5):
            if x < w2 and y < h2:
                if p1data[x + p2xoffset, y + p2yoffset] == 1 and p2data[x, y] == 1:
                    overlap += 1
    return overlap

def getcolordiff(peice, gender, organ):
    """
    returns the average rgb vale or an organ
    :param peice: a single peice class
    :param gender: gender of the organ to find the color score of
    :param organ: the point of the organ
    :return: the average rgb value of the organ
    """
    depth = 7
    xsum = 0
    ysum = 0
    for point in peice.outline:
        xsum += point[0]
        ysum += point[1]
    cx = xsum // len(peice.outline)
    cy = ysum // len(peice.outline)
    px = peice.image.load()
    r = 0
    g = 0
    b = 0
    total = 0
    if gender == 0:
        for female in peice.femalelist[organ]:

            if female[0] < cx:
                xdirection = 1
            else:
                xdirection = -1
            if female[1] < cy:
                ydirection = 1
            else:
                ydirection = -1
            for i in range(1, depth+1):
                rx, gx, bx = px[female[0] + (xdirection * i), female[1]]
                ry, gy, by = px[female[0], female[1] + (ydirection * i)]
                r += rx + ry
                g += gx + gy
                b += bx + by
                total += 2
    else:
        for male in peice.malelist[organ]:

            if male[0] < cx:
                xdirection = 1
            else:
                xdirection = -1
            if male[1] < cy:
                ydirection = 1
            else:
                ydirection = -1
            for i in range(1, depth + 1):
                rx, gx, bx = px[male[0] + (xdirection * i), male[1]]
                ry, gy, by = px[male[0], male[1] + (ydirection * i)]
                r += rx + ry
                g += gx + gy
                b += bx + by
                total += 2
    return [r/total, g/total, b/total]




def getmatcharray(Pieces):
    """
    calcutales the score for all possible combinations
    :param Pieces: array of all the peices
    :return: array withh all possible matches
    """
    match = []
    colorweight = 500
    for l, p1 in enumerate(Pieces):
        print(l, "of ", len(Pieces))
        for m, p2 in enumerate(Pieces):
            if p1.outline != p2.outline:

                i = 0

                for p1female in p1.femaleorgans:
                    j = 0
                    for p2male in p2.maleorgans:
                        difftotal = getoverlap(p1, p1female, p2, p2male)
                        if difftotal < 100:
                            difftotal = 9999
                        colorweight = difftotal
                        p1color = getcolordiff(p1, 0, i)
                        p2color = getcolordiff(p2, 1, j)
                        difftotal += colorweight*abs(p1color[0] - p2color[0])
                        difftotal += colorweight*abs(p1color[1] - p2color[1])
                        difftotal += colorweight*abs(p1color[2] - p2color[2])
                        match.append(Match(l, m, i, j, difftotal))
                        j += 1

                    i += 1

    return match




def findBestremaining(combo, matches, Pieces, used):
    """
    picks the best match to use
    :param combo: class of the combonation image
    :param matches: array cointing avalible matches
    :param Pieces: array of all the peices
    :param used: unneeded
    :return: the match and the gender of the selected match
    """
    goal = random.randrange(0, 9)
    current = 0
    currentpiece = combo.pque.get()
    while True:
        for m in matches:
            if Pieces[m.fpeice] == currentpiece and Pieces[m.mpeice] not in combo.peices:
                w, h = Pieces[m.fpeice].image.size
                if Pieces[m.fpeice].femaleorgans[m.forgan][0] > w // 2 or Pieces[m.fpeice].femaleorgans[m.forgan][
                    1] > h // 2:
                    if current == goal:
                        matches.remove(m)
                        return m, 0
                    current += 1
            if Pieces[m.mpeice] == currentpiece and Pieces[m.fpeice] not in combo.peices:
                w, h = Pieces[m.mpeice].image.size
                if Pieces[m.mpeice].maleorgans[m.morgan][0] > w // 2 or Pieces[m.mpeice].maleorgans[m.morgan][
                    1] > h // 2:
                    if current == goal:
                        matches.remove(m)
                        return m, 1
                    current += 1
        goal -= 1
        if goal < 0:
            goal = 10

def main():
    fname = input("Enter a filename of an image: ")
    Pieces = getPieces(fname)


    Pieces = CleanUp(Pieces)

    Pieces = bwmask(Pieces)

    Pieces = Outline(Pieces)

    for p in Pieces:
        print("finding organs")
        findfemalesexorgans(p)
        findmalesexorgans(p)
        print("amputating ligiments")
        #amputate(p)
        print("growing organs")
        groworgans(p)
        optimizeorgans(p)


    makediffrence(Pieces)
    matches = getmatcharray(Pieces)
    matches.sort()
    backup = matches[:]
    good = "no"
    while good == "no":
        matches = backup[:]
        if fname == "Green_Edge.jpg":
            combo = combinep(Pieces[9], Pieces[9].femaleorgans[2], Pieces[12], Pieces[12].maleorgans[0])  # good

            combo.addpiece(Pieces[2], Pieces[2].maleorgans[0], 1, Pieces[12], Pieces[12].femaleorgans[2])
            combo.image.show()
            used = []
            used.append(Match(9, 12, 2, 0, -1))
            used.append(Match(2, 12, 0, 2, -1))
        elif fname == "3by3.jpg": # works with 3by3
            combo = combinep2(Pieces[0], Pieces[0].maleorgans[1], Pieces[1], Pieces[1].femaleorgans[0])  # good
            combo.addpiece(Pieces[3], Pieces[3].maleorgans[0], 1, Pieces[0], Pieces[0].femaleorgans[3])
            # combo.image.show()
            used = []
            used.append(Match(1, 0, 0, 1, -1))
            used.append(Match(0, 3, 3, 0, -1))
        elif fname == "4by4.jpg":
            combo = combinep2(Pieces[0], Pieces[0].maleorgans[1], Pieces[1], Pieces[1].femaleorgans[0])  # good
            combo.addpiece(Pieces[4], Pieces[4].maleorgans[0], 1, Pieces[0], Pieces[0].femaleorgans[3])
            # combo.image.show()
            used = []
            used.append(Match(1, 0, 0, 1, -1))
            used.append(Match(0, 4, 3, 0, -1))
        else: #full
            combo = combinep2(Pieces[0], Pieces[0].maleorgans[1], Pieces[1], Pieces[1].femaleorgans[2])  # good
            combo.addpiece(Pieces[10], Pieces[10].maleorgans[0], 1, Pieces[0], Pieces[0].femaleorgans[3])
            # combo.image.show()
            used = []
            used.append(Match(1, 0, 0, 1, -1))
            used.append(Match(0, 10, 3, 0, -1))


        for u in used:
            for m in matches:
                if u.fpeice == m.fpeice and u.forgan == m.forgan:
                    matches.remove(m)
                elif u.mpeice == m.mpeice and u.morgan == m.morgan:
                    matches.remove(m)
        while len(combo.peices) < len(Pieces):
            bestmatch, gender = findBestremaining(combo, matches, Pieces, used)
            print(bestmatch)
            if gender == 0:#adding a new male part
                combo.addpiece(Pieces[bestmatch.mpeice], Pieces[bestmatch.mpeice].maleorgans[bestmatch.morgan], 1, Pieces[bestmatch.fpeice], Pieces[bestmatch.fpeice].femaleorgans[bestmatch.forgan] )
            else:
                combo.addpiece(Pieces[bestmatch.fpeice], Pieces[bestmatch.fpeice].femaleorgans[bestmatch.forgan], 0, Pieces[bestmatch.mpeice], Pieces[bestmatch.mpeice].maleorgans[bestmatch.morgan] )

            for u in used:
                for m in matches:
                    if u.fpeice == m.fpeice and u.forgan == m.forgan:
                        matches.remove(m)
                    elif u.mpeice == m.mpeice and u.morgan == m.morgan:
                        matches.remove(m)
        combo.image.show()
        good = input("Is this good? ")



main()

