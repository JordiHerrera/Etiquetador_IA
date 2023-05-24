__authors__ = ['1638117', '1639392', '1550960']
__group__ = 'DM.12'

from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_k_means

import Kmeans as km
from Kmeans import *

import KNN as k
from KNN import *

import utils as ut
import time


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
        color = get_colors(km.centroids)
        if all(elem in color for elem in comp_labels[i]):
            print('Coincideix')
            
        else:
            print('Falla a la iteració', i)   
            print('Es:', list(set(color)))
            print('Hauria de ser:', test_color_labels[i])
            mides = np.shape(array_imgs[i])
            visualize_k_means(km, [mides[0], mides[1], mides[2]])          #<--DESCOMENTAR PER GENERAR GRAFIC DELS ERRORS <!>
            count = count + 1
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
    for i in range(mostres):
        km = KMeans(array_imgs[i], current_k) 
        km.fit()  
        color = get_colors(km.centroids)
        if all(elem in color for elem in comp_labels[i]):
            imatges.append([i, array_imgs[i], color])
    
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
        final_ret = ret_list
        
    print('Temps execucio:',time.time() - start_time)
    return final_ret
    

if __name__ == '__main__':

    print('Iniciant lectura del dataset...')    

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

    print(' ')
    alg = input('Quin algorisme vols probar? (0 = K-Means, 1 = KNN) ')
    if alg == '0':
        kmean = input('Quina funcio vols provar? (0 = get_color_accuracy, 1 = retrieval_by_color) ')
        if kmean == '0':
            m_in = input('Mida de la mostra: ')
            mostra = int(m_in)
            print(' ')
            get_color_accuracy(test_imgs, test_color_labels, mostra)
        elif kmean == '1':
            m_in = input('Mida de la mostra: ')
            mostra = int(m_in)
            color_in = input('Indica els colors de les imatges que vols obtenir: ')
            desired_colors = color_in.split()
            acc_in = input('Vols que qualsevol dels color estigui a la imatge o tots? (0 = Qualsevol, 1 = Tots) ')
            acc = int(acc_in)
            print(' ')
            retrieval = retrieval_by_color(test_imgs, test_color_labels, mostra, desired_colors, acc)
            print(' ')
            if acc == 0:
                for current_k, current_v in retrieval.items():
                    print(current_k,':')
                    for im in current_v:
                        print(im[0],',', im[2])
                    print(' ')
            elif acc == 1:
                for current in retrieval:
                    print(current[0],',', current[2])
    elif alg == '1':
        print('Coses KNN')
