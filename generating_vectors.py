import ctypes
import cv2
import numpy as np
import math
import time
import graph_tool.all as gt
from matplotlib import pyplot as plt
import numba as nb
import mahotas as mt
import pandas as pd
from skimage.feature import local_binary_pattern

@nb.jit
def filter_func(x, w, z):
    return ( w < x and x <= z)

# iterate over pixels and accumulate the closeness based on pixel intensity
@nb.jit
def filter_np_nb(arr, cc, width, height, w, z):
    result = 0.0
    for j in range(height):
        for i in range(width):
            if filter_func(arr[j][i], w, z):
                result += cc[j][i]
    return result

# generate image graph and return its closeness
def getCloseness(im, w, h, radius, threshold):
    r = radius*radius
    # creates empty graph
    graph = gt.Graph(directed=True)
    # add w*h vertices
    vlist = graph.add_vertex(w*h)

    # creates graph properties
    v_prop_position = graph.new_vertex_property("object")
    e_prop_distancia = graph.new_edge_property("double")  

    # iterate over the pixels
    for y in range(h):
        for x in range(w):
            v = graph.vertex(x+y*w)
            v_prop_position[v] = {"x": x, "y": y}
            for j in range(int(y-radius),int(y+radius)):
                for i in range(int(x-radius),int(x+radius)):
                    if(i >= 0 and i < w and j >= 0 and j < h and (i != x or j != y)):
                        
                        weightVertex = int(im[y][x]) - int(im[j][i])              
                        d = ((x-i)*(x-i) + (y-j)*(y-j))

                        if(weightVertex <= threshold and weightVertex > 0 and d <= r):
                            # creates the edge (x,y) -> (i,j)
                            e = graph.add_edge(graph.vertex(x+y*w), graph.vertex(i+j*w))
                            e_prop_distancia[e] = math.sqrt(d)

    # get closeness
    cc = gt.closeness(graph, weight=e_prop_distancia, harmonic=True)    
    cc = np.reshape(np.nan_to_num(cc.a), (h,w))
    return (cc)

# creates histogram of LBP feature
def lbpHist(lbp):
    hist, _ = np.histogram(lbp, density=True)
    return hist

# get the feature vector of the image
def getVvector(original_image, radius, t0):
    image = original_image

    # normalixe image to (0,1) interval
    image0_1 = cv2.normalize(original_image, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F) #float

    # get image shape
    (h, w) = image.shape
    h = len(image)
    w = len(image[0])
    
    # iterate over the thresholds and get closeness metric
    cc_array = []
    for t in range(t0,t0+(m+1)*ti,ti):
        cc = getCloseness(image,w,h,float(radius),float(t))
        cc_array.append(cc)

    # get mean of closeness
    cc_mean = np.mean(cc_array, axis=0)

    # intensity intervals
    m_intervalos = 75
    intervalos = []
    for i in range(0,m_intervalos):
        intervalos.append((i/m_intervalos,(i+1)/m_intervalos))

    # generate phi vector
    phi = []
    for (w,z) in intervalos:
        S = filter_np_nb(image0_1, cc_mean, len(image[0]),len(image),w,z)
        phi.append(S)
    phi = np.array(phi)
    # generate haralick vector
    haralick = mt.features.haralick(original_image,compute_14th_feature=True)
    haralick = haralick.mean(axis=0)

    # generate LBP vector
    lbpRadius = radius
    n_points = 3*radius
    lbp = local_binary_pattern(original_image,lbpRadius,n_points)
    retLBP = lbpHist(lbp)

    return (phi, haralick, retLBP)

# path to classes file
class_file = "./labels/groups_DPD.txt"

# path to image folder
path_images = "./images/"

# parameters
t0 = 15
ti = 40
m = 5
radius = 2

# results to be stored
results = []

# start timer
start = time.perf_counter()

# iterate over the images
for count, line in enumerate(open(class_file, 'r').readlines()):
    print("Line{}: >{}<".format(count+1, line.strip()))
    # image name
    name = str(count+1)+".jpg"
    # image class
    classification = line.strip()
    # read image
    image = cv2.imread(path_images+name,cv2.IMREAD_GRAYSCALE)

    # if there is not an image with given name, just skip it
    if image is None:
        continue

    # get image vectors
    phi,haralick,lbp = getVvector(image, radius, t0)

    # append result
    results.append((name, phi.tolist(), haralick.tolist(), lbp.tolist(), classification))


# store result as csv file
df = pd.DataFrame(results, columns=['name', 'phi', 'haralick', 'lbp', 'classification'])
df['phi'] = df['phi'].astype(object)
df['haralick'] = df['haralick'].astype(object)
df['lbp'] = df['lbp'].astype(object)
df['name'] = df['name'].astype(str)
df['classification'] = df['classification'].astype(str)
df.to_csv('./results/result.csv',index=False)


#end timer
end = time.perf_counter()
print("Ellapsed time: "+ str(end - start))