__authors__ = ['1638117', '1639392', '1550960']
__group__ = 'DM.12'

from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_k_means

import Kmeans as km
from Kmeans import *

import KNN as k
from KNN import *

import utils as ut
import time

def test(a):
    b = 2 + 2
    return b

def get_color_accuracy(array_imgs, comp_labels, mostres):
    start_time = time.time()

    count = 0
    max_k = 10
    current_k = 3
    
    for i in range(mostres):
        if i % 10 == 0:
            print('===========================> Marca de les', i, 'iteracions')
        km = KMeans(array_imgs[i], current_k) 
        km.find_bestK(max_k)
        print('Millor K trobada:', km.K)
        km.fit()  
        #visualize_k_means(km, [80, 60, 3])
        color = get_colors(km.centroids)
        if all(elem in color for elem in comp_labels[i]):
            print('Coincideix')
            
        else:
            print('Falla a la iteració', i)   
            print('Es:', list(set(color)))
            print('Hauria de ser:', test_color_labels[i])
            #visualize_k_means(km, [80, 60, 3])
            count = count + 1
        #print('Es:', list(set(color)))
        #print('Hauria de ser:', test_color_labels[i])
        print(' ')

    print('Tasa encert =', 1-(count/mostres))
    print('Tasa error =', count/mostres)
    print('Temps execucio:',time.time() - start_time)
    
    return  1-(count/mostres)

def retrieval_by_color(array_imgs, comp_labels, mostres, colors, acc = 0):
    
    """
    acc: 0 --> Any of the colors is in the image (default), returns dictionary with an entry for each color
         1 --> All of the colors are in the image, returns a list
    """
    
    start_time = time.time()
    imatges = []
    max_k = 10
    current_k = 3
    final_ret = 0
    
    ret = []
    print('Entra k-means')
    for i in range(mostres):
        km = KMeans(array_imgs[i], current_k) 
        km.fit()  
        color = get_colors(km.centroids)
        if all(elem in color for elem in comp_labels[i]):
            print(comp_labels[i])
            imatges.append([i, array_imgs[i], color])
    
    print('Acaba k-means')
    
    if acc == 0:
        ret_dict = {}
        for current_color in colors:
            list_dict = []
            for current_img in imatges:
                if current_color in current_img[2]:
                    list_dict.append(current_img)
            ret_dict[current_color] = list_dict
        final_ret = ret_dict  
        
    elif acc == 1:
        ret_list = []
        for current_img in imatges:
            if all(elem in color for elem in current_img[2]):
                ret_list.append(current_img)
                print('Entra')
        final_ret = ret_list
        
            
    return final_ret
    print('Temps execucio:',time.time() - start_time)

if __name__ == '__main__':

    print('Iniciant lectura dataset...')    

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')
        
    print('Inicialitzant classes...')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))
    
    print('Iniciant lectura dataset extendit...')

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    
    print('Creant versions retallades...')
    
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here
    print('Fins aqui no falla')
    
    #get_color_accuracy(test_imgs, test_color_labels, 50) 
    
    desired_colors = ['White', 'Black']
    acc = 1
    
    ret = retrieval_by_color(test_imgs, test_color_labels, 10, desired_colors, acc)
    print(ret)
    
    
    