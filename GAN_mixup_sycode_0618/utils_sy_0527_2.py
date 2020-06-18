from sklearn.datasets import make_circles, make_moons
from torch.nn import init
from torch.nn import Parameter as P
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from torch.utils.data import TensorDataset, Dataset
from torchvision.datasets import ImageFolder
#import torchvision.utils as vutils
from torchvision.utils import save_image
#from torch.optim.lr_scheduler import StepLR, MultiStepLR
from matplotlib import pyplot
from pandas import DataFrame
from numpy import genfromtxt
import matplotlib.pyplot as plt

from models_sy_0527 import *

import random
import torch
import numpy as np
import os

from sklearn.datasets import make_circles, make_moons
#from models.mlp import MLP
#from models.mixup_utils import mixup_data, mixup_criterion
from mpl_toolkits.mplot3d import Axes3D

# ### Inverting GAN for a given image (Find optimal z for a given input image x)
    
def get_proj_distance_square(input_image, manifold_label_proj, data_,label_,args):
    '''
    Calculate the minimum distance between a data point "input_image" and a manifold (w/ label "manifold_label_proj")
    manifold_dim : the dimension of z noise
    num_epochs_z = How much step you make during finding the optimal projected image
    '''
    # saved_generator, manifold_label_proj, label_GAN, manifold_dim, data_type, data_class, device, distance_type, show_image, data_, label_, lr=0.05, num_epochs_z=500
    #self.gan, 0, label_GAN, manifold_dim, data_type, data_class, self.device, distance_type, False, data_, label_, lr=self.lr, num_epochs_z=num_epochs_z
    saved_generator = args.gan
    label_GAN = args.label_GAN
    manifold_dim = args.z_dim
    data_type = args.data_type
    data_class = args.data_class
    device = args.device
    distance_type = args.distance_type
    show_image = False
    #data_
    #label_
    lr = args.lr
    num_epochs_z = args.num_epochs_z
    num_random_z = args.num_random_z
    image_size_ref = args.image_size_ref


    #if data_class == 'Real':    num_random_z, image_size_ref = [10, 32]
    #else:    num_random_z = 5

    z_maxRad_coeff = 1.1 
    z_minRad_coeff = 0.9     

    if data_type == 'mnist_ext':
        lr = 0.01 
        num_epochs_z = 500 
    elif data_class == 'Synthetic':
        #lr = 0.005
        #num_epochs_z = 2000
        z_maxRad_coeff = 3
        z_minRad_coeff = 0

    if data_type in ['mnist', 'mnist_ext']: numCH = 1
    elif data_type in ['cifar10']: numCH = 3


    input_image = input_image.to(device)
    saved_generator.eval()
    y_label = torch.zeros([1,label_GAN])   # one-hot vector for indicating the label of images to generate
    y_label[0][manifold_label_proj]=1
    
    for random_z in range(num_random_z):
        #print('========= Random z generated at iter %d ===========' % random_z)
        if data_class == 'Real':
            z_Var = Variable(torch.randn([1, manifold_dim, 1, 1]).to(device), requires_grad=True)     # Set initial z value
            y_label = y_label.view(1,-1,1,1)

        else:
            z_Var = Variable(torch.randn([manifold_dim,1]).to(device), requires_grad=True)     # Set initial z value
        y_label = y_label.to(device)
        #pdb.set_trace()
        z_optimizer = optim.Adam([z_Var], lr, betas=(0.5, 0.999))                    # Optimizer
        for iter_z in range(num_epochs_z):
            z_maxRad = z_maxRad_coeff * np.sqrt(manifold_dim) 
            z_minRad = z_minRad_coeff * np.sqrt(manifold_dim)    
            if torch.norm(z_Var) > z_maxRad:    z_Var.data = z_Var.data/torch.norm(z_Var) * z_maxRad
            if torch.norm(z_Var) < z_minRad:    z_Var.data = z_Var.data/torch.norm(z_Var) * z_minRad

            z_Var = z_Var.reshape(-1,manifold_dim)
            #pdb.set_trace()

            # Check the projected image & calculate loss
            proj_image = saved_generator(z_Var, y_label)
            #pdb.set_trace()
            data_dim = proj_image.size()[1]
            if distance_type == 'L2':
                loss_z = nn.MSELoss()     # Loss function (L2-norm case)
                if data_class == 'Real':    recLoss = loss_z(proj_image, input_image.view(-1, numCH, image_size_ref,image_size_ref).float()) # L2-norm case
                else:    recLoss = loss_z(proj_image, input_image.view(1,data_dim).float()) # L2-norm case
            elif distance_type == 'L1':
                if data_class == 'Real':    recLoss = torch.sum(torch.abs(proj_image - input_image.view(-1, numCH, image_size_ref,image_size_ref).float())) # L1-norm case
                else:    recLoss = torch.sum(torch.abs(proj_image - input_image.view(1,data_dim).float())) # L1-norm case
            elif distance_type == 'Linf':
                #pdb.set_trace()
                if data_class == 'Real':   recLoss = torch.max(torch.abs(proj_image - input_image.view(-1, numCH, image_size_ref,image_size_ref).float())) # Linf-norm case
                else:    recLoss = torch.max(torch.abs(proj_image - input_image.view(1,data_dim).float())) # Linf-norm case
                
            # Update best z variable (with minimum loss)
            if iter_z == 0:
                best_loss_z = recLoss.data
                best_z_Var = z_Var.clone().detach()
            else:
                if recLoss.data < best_loss_z:
                    best_loss_z = recLoss.data
                    best_z_Var = z_Var.clone().detach()

            # Update z using gradient descent
            z_optimizer.zero_grad()
            recLoss.backward()
            z_optimizer.step()

            # Display status    
        #     if (iter_z) % 50 == 0: 
        #         print ('Epoch: %d, LR: %0.5f,  loss: %0.5f, z_Var: %0.3f' % (iter_z, lr, recLoss.data, torch.norm(z_Var.data))) 

        # print ('Smallest loss: %0.5f' % (best_loss_z))
        # if distance_type == 'L2':
        #     print ('Closest L2-distance (sqrt of MSE): %0.5f' % (np.sqrt(best_loss_z.cpu()))) # L2-norm case
        # elif distance_type == 'L1':
        #     print ('Closest L1-distance: %0.5f' % (best_loss_z)) # L1-norm case
        # elif distance_type == 'Linf':
        #     print ('Closest Linf-distance: %0.5f' % (best_loss_z)) # Linf-norm case
        # print('L2-norm of best_z_Var: ', torch.norm(best_z_Var).data)

        proj_image = saved_generator(best_z_Var, y_label).detach() # G_i (z*) << here, i depends on y_label          
        if random_z == 0: # the first time
            BoB_z_Var = best_z_Var
            BoB_loss_z = best_loss_z
        else:
            if best_loss_z < BoB_loss_z:
                BoB_z_Var = best_z_Var
                BoB_loss_z = best_loss_z
                        
    BoB_proj_image = saved_generator(BoB_z_Var, y_label).detach()
        
    if distance_type == 'L2':
        #pdb.set_trace()
        distance = torch.norm(BoB_proj_image - input_image.float())/32# L2-norm case (sqrt(MSE))
    elif distance_type == 'L1':        
        distance = torch.sum(torch.abs(BoB_proj_image - input_image.float())) # L1-norm case
    elif distance_type == 'Linf':        
        distance = torch.max(torch.abs(BoB_proj_image - input_image.float())) # Linf-norm case

    # print('[BEST_OF_BEST]_[Projected To Manifold %d] projected image: ' % manifold_label_proj, BoB_proj_image)
    # print('torch.max(projected image)', torch.max(BoB_proj_image) )
    # print('torch.min(projected image)', torch.min(BoB_proj_image) )
    # print('Red point image: ', input_image)
    # print('BoB Distance: ', distance)
    
#    if show_image == True:
#        if data_class == 'Synthetic': 
#            fig1, ax1 = plt.subplots()
#            fig2, ax2 = plt.subplots()
#            df = DataFrame(dict(x=data_[:,0].cpu(), y=data_[:,1].cpu(), label=label_.cpu()))
#            colors = {0:'black', 1:'blue'}
#            grouped = df.groupby('label')                
#
#            if num_class_total == 1:
#                ax1.scatter(data_[:,0], data_[:,1], c=label_, cmap='gray', edgecolor='k')
#                ax2.scatter(data_[:,0], data_[:,1], c=label_, cmap='gray', edgecolor='k')
#
#            for key, group in grouped:
#                group.plot(ax=ax1, kind='scatter', x='x', y='y', label=key, color=colors[key])
#                group.plot(ax=ax2, kind='scatter', x='x', y='y', label=key, color=colors[key])
#            ax1.scatter(input_image[0][0].cpu(), input_image[0][1].cpu(), c='red')
#            ax2.scatter(BoB_proj_image[0][0].cpu(), BoB_proj_image[0][1].cpu(), c='red')
#
#            if data_type in ['circle', 'v_shaped']:
#                ax1.axis('equal')          
#                ax2.axis('equal')              
#                plt.axis('square')         
#            plt.show()    
#            
#        elif data_type in ['mnist', 'mnist_ext']:
#            plt.subplot(121)
#            plt.imshow(input_image.view(32,32), cmap='Greys')
#            plt.subplot(122)
#            plt.imshow(BoB_proj_image.view(32,32), cmap='Greys')
#            plt.show()    
    
    return BoB_z_Var, BoB_proj_image, distance

def seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #TODO: Do we need deterministic in cudnn ? Double check
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print ("Seeded everything")


def load_synthetic_data(data_type, data_dim, n_samples_train, n_samples_test, show_image, device, train_noise = False, train_seed=42, test_seed=24):

    ## Specify parameters for synthetic dataset (follow these values for our setup)

    circle_factor = 0.5
    circle_noise = 0.1 #0.05 #0.01
    high_circle_noise = 0.1 #0.05 #0.01
    moon_noise =  0.1 #0.05 #0.05 #0.1
    v_shape_noise = 0 
    braket_noise = 0
    
    
    if data_type == 'moon': 
        x_max, x_min, y_max, y_min = (2, -1, 1, 0.5)

        seed(train_seed) # seed before generating training data 
        data_, label_ = make_moons(n_samples=n_samples_train) # data_: 2D position vector, label_: label (scalar)
        if train_noise == True: data_ += np.random.laplace(scale=moon_noise, size=data_.shape) # add noise to data_  #data_ += np.random.exponential(scale=moon_noise, size=data_.shape)
        seed(test_seed) # seed before generating test data
        test_data_, test_label_ = make_moons(n_samples=n_samples_test)
        test_data_ += np.random.laplace(scale=moon_noise, size=test_data_.shape) # add noise to data_
        
    elif data_type == 'circle': 
        x_max, x_min, y_max, y_min = (1, -1, 1, -1)
        
        seed(train_seed) # seed before generating training data 
        data_, label_ = make_circles(n_samples=n_samples_train, factor=circle_factor) # data_: 2D position vector, label_: label (scalar)
        if train_noise == True: data_ += np.random.laplace(scale=circle_noise, size=data_.shape) # add noise to data #data_ += np.random.exponential(scale=circle_noise, size=data_.shape)
        seed(test_seed) # seed before generating test data
        test_data_, test_label_ = make_circles(n_samples=n_samples_test, factor=circle_factor)
        test_data_ += np.random.laplace(scale=circle_noise, size=test_data_.shape) # add noise to data_

    elif data_type == 'high_circle':

        #x_max, x_min, y_max, y_min = (1, -1, 1, -1)

        seed(train_seed) # seed before generating training data 
        data_, label_ = make_high_circle(n_samples=n_samples_train, factor=circle_factor, dimension= data_dim) # data_: n-dim position vector, label_: label (scalar)
        if train_noise == True: data_ += torch.from_numpy(np.random.laplace(scale=high_circle_noise, size=data_.shape)) # add noise to data #data_ += np.random.exponential(scale=circle_noise, size=data_.shape)
        seed(test_seed) # seed before generating test data
        test_data_, test_label_ = make_high_circle(n_samples=n_samples_test, factor=circle_factor, dimension = data_dim)
        #pdb.set_trace()
        test_data_ += torch.from_numpy(np.random.laplace(scale=high_circle_noise, size=test_data_.shape)) # add noise to data_

        #pdb.set_trace()


    elif data_type == 'v_shaped': 
        x_min, x_max, slope  = (0.5, 2, 2) # determines the shape of dataset

        seed(train_seed) # seed before generating training data 
        data_, label_ = make_v_shape(n_samples_train, v_shape_noise, x_max, x_min, slope)

        seed(test_seed) # seed before generating test data 
        test_data_, test_label_ = make_v_shape(n_samples_test, v_shape_noise, x_max, x_min, slope)

    elif data_type == 'bracket': 
        x_min, x_max = (1, 2) # determines the shape of dataset
        data_, label_ = make_braket(n_samples_train, braket_noise, x_max, x_min)

    # convert to tensor
    if data_type in ['moon', 'circle']:
        data_, label_ = torch.from_numpy(data_), torch.from_numpy(label_)  
        test_data_, test_label_ = torch.from_numpy(test_data_), torch.from_numpy(test_label_)  



    # plot the synthetic data
    if data_dim == 2:

        fig=plot_synthetic_data(data_,'Synthetic', label_,show_image)
        fig.savefig('train_data_{}.png'.format(data_type))

        fig=plot_synthetic_data(test_data_,'Synthetic', test_label_,show_image)
        fig.savefig('test_data_{}.png'.format(data_type))

        num_test = int(10000) #examples #10000 #500000
        if data_type == 'v_shaped':
            X_test = (2 * x_max + 1) * torch.arange(num_test).float()/num_test - (x_max + 0.5)
            X_test = X_test.view(num_test,1)
            Y_test = (slope * (x_max - x_min) + 1) * torch.rand(num_test,1) - 0.5  
            X_test = torch.cat((X_test, Y_test), dim=1)
        elif data_type == 'circle':
            X_test = (x_max - x_min) * torch.rand(num_test,data_dim) + x_min
        elif data_type == 'moon':
            X_test = (x_max - x_min + 0.5) * torch.arange(num_test).float()/num_test + x_min - 0.25
            X_test = X_test.view(num_test,1)        
            Y_test = (y_max - (-y_max) + 0.5) * torch.rand(num_test,1) - y_max - 0.25        
            X_test = torch.cat((X_test, Y_test), dim=1)   
        elif data_type == 'bracket':
            X_test = (2 * x_max + 1) * torch.arange(num_test).float()/num_test - (x_max + 0.5)
            X_test = X_test.view(num_test,1)
            Y_test = (x_max + 2) * torch.rand(num_test,1) - 1       
            X_test = torch.cat((X_test, Y_test), dim=1)  

    else:
        plot_synthetic_data(data_,'Real', label_,show_image)

        X_test = torch.zeros([0,data_dim])

    X_test = X_test.to(device)

    return data_, label_, test_data_, test_label_, X_test


def load_synthetic_middle_points(data_type, distance_type, early_stop):

    if data_type == 'circle':
        if distance_type == 'L2':
            #my_middle_data = genfromtxt('./data/arxiv_middle_excel/L2/middle_points_circle_0.01_Dec3_total.csv', delimiter=',') # circle L2 latest
            my_middle_data = genfromtxt('./data/arxiv_middle_excel/L2/middle_points_circle_1000_laplace_0.1.csv', delimiter=',') # circle L2 latest (laplacian)

    elif data_type == 'moon':
        if distance_type == 'L2':
            #my_middle_data = genfromtxt('./data/arxiv_middle_excel/L2/middle_points_moon_noise_0.05_total_Dec15_added.csv', delimiter=',') # moon latest + Dec.15 added
            my_middle_data = genfromtxt('./data/arxiv_middle_excel/L2/middle_points_moon_1000_laplace_0.1.csv', delimiter=',') # moon latest (laplacian)
        elif distance_type == 'L1':
            my_middle_data = genfromtxt('./data/arxiv_middle_excel/L1/middle_points_moon_total.csv', delimiter=',') # moon latest
        elif distance_type == 'Linf':
            my_middle_data = genfromtxt('./data/arxiv_middle_excel/Linf/middle_points_moon_Linf_total_alpha_10.csv', delimiter=',') # moon latest

    elif data_type == 'v_shaped':
        if distance_type == 'L2':
            if early_stop == True:
                #my_middle_data = genfromtxt('./data/arxiv_middle_excel/L2/middle_points_v_shaped_early_stop_Apr20.csv', delimiter=',') # v_shaped 
                #my_middle_data = genfromtxt('./data/arxiv_middle_excel/L2/middle_points_v_shaped_early_stop_2_iter_Apr20.csv', delimiter=',') # v_shaped 
                my_middle_data = genfromtxt('./data/arxiv_middle_excel/L2/middle_points_v_shaped_early_stop_2_iter_Apr20_only32.csv', delimiter=',') # v_shaped 
            else:
                my_middle_data = genfromtxt('./data/arxiv_middle_excel/L2/middle_points_v_shape_total_Dec11_append.csv', delimiter=',') # v_shape Dec.11 (appended - total 35 points) >> used in training!

    elif data_type == 'bracket':
        if distance_type == 'L2':
            my_middle_data = genfromtxt('./data/arxiv_middle_excel/L2/middle_points_braket_total_Dec17_9pm.csv', delimiter=',') # braket Dec.17 9pm  >> used in training!

    return my_middle_data

def load_real_middle_points(data_type, num_mnist_train_for_GAN, image_size_ref, distance_type, show_image):
    mnist_compose_augment = transforms.Compose(
            [
            transforms.Resize((image_size_ref,image_size_ref)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
    if distance_type == 'Linf':
        if num_mnist_train_for_GAN == 2000:
            MNIST_AUG_DATA_PATH = "./data/mnist_middle/arxiv/Linf/May_10_dual_direction" #   400 (7->9) + 400 (9->7) points
        elif num_mnist_train_for_GAN == 500:
            MNIST_AUG_DATA_PATH = "./data/mnist_middle/arxiv/Linf(500_points)/May_16_all" #   3100 (start from middle, May.16)
    elif distance_type == 'L2':
            MNIST_AUG_DATA_PATH = "./data/mnist_middle/arxiv/L2/May_11_dual_direction" #   200 (7->9) + 200 (9->7) points
    elif distance_type == 'Linf+L2':
        MNIST_AUG_DATA_PATH = "./data/mnist_middle/arxiv/Linf+L2/May_10_11" #   600 (7->9) + 600 (9->7) points (= May_10_dual + May_11_dual)

    mnist_train_aug = ImageFolder(root=MNIST_AUG_DATA_PATH, transform=mnist_compose_augment)
    mnist_train_aug.targets = torch.tensor(mnist_train_aug.targets)
    num_augment = len(mnist_train_aug)

    return mnist_train_aug, num_augment

def load_GAN(data_type, data_dim, noise_dim, num_class, device, train_add_noise, label_GAN):
    #load_GAN(data_type, data_dim, noise_dim, num_class, device, train_add_noise, label_GAN) 
    if data_type == 'circle':
        saved_generator = generator_circle(noise_dim, num_class, data_dim)
        
        if train_add_noise == True:
            saved_generator = generator_moon(noise_dim, num_class, data_dim)
            saved_generator.load_state_dict(torch.load('./data/models/cGAN/generator_param_circle_epoch_793.pkl', map_location=device)) #

        else:
            saved_generator.load_state_dict(torch.load('./data/models/cGAN/generator_param_circle_reduced_gan_epoch_162.pkl', map_location=device)) # after KW (Dec.2. 10pm)

    elif data_type == 'high_circle':
        saved_generator = generator_moon_complex(noise_dim, num_class, data_dim)

        if train_add_noise == True:
            print("we didn't add noise")
        else:
            saved_generator.load_state_dict(torch.load('./data/models/cGAN/generator_param_high_circle_epoch_121.pkl', map_location=device)) #

    elif data_type == 'moon':
        saved_generator = generator_moon_complex(noise_dim, num_class, data_dim)

        if train_add_noise == True:
            print("we didn't add noise")
            saved_generator.load_state_dict(torch.load('./data/models/cGAN/generator_param_moon_epoch_617.pkl', map_location=device)) 

        else:
            print("we didn't add noise")
            saved_generator.load_state_dict(torch.load('./data/models/cGAN/generator_param_moon_epoch_400_simple_noise_0.05.pkl', map_location=device)) # Dec.5. 2pm)  -> data noise = 0.05 given

    elif data_type == 'v_shaped':
        saved_generator = generator_braket_complex(noise_dim, num_class, data_dim)
        saved_generator.load_state_dict(torch.load('./data/models/cGAN/generator_param_v_shape_epoch_175.pkl', map_location=device)) # Dec.28. 10pm  -> data noise = 0 & x_min, x_max, slope  = (0.5, 2, 2)

    elif data_type == 'bracket':
        saved_generator = generator_braket_complex(noise_dim, num_class, data_dim)
        saved_generator.load_state_dict(torch.load('./data/models/cGAN/generator_param_braket_epoch_364.pkl', map_location=device)) # Dec.16. 6pm  -> data noise = 0 & x_min, x_max = (1, 2) >> Dec.28 10pm current best
    
    elif data_type == 'mnist':
        saved_generator = conditional_DC_generator(label_GAN = label_GAN)

        if label_GAN == 2:
            saved_generator.load_state_dict(torch.load('./data/models/cDCGAN/MNIST_cDCGAN_generator_param_labels_7_9_num_samples_2000_per_class.pkl', map_location=device)) # trained for 2000 data points at each label \in [7,9]
        elif label_GAN == 10: 
            saved_generator.load_state_dict(torch.load('./data/models/cDCGAN/MNIST_cDCGAN_generator_param_labels_all_num_samples_500_per_class.pkl', map_location=device)) # trained for 500 data points at each label
            print("load_GAN() in utils.py: I am using GAN with 10 input labels")

    elif data_type == 'mnist_ext':
        saved_generator = conditional_DC_generator()
        saved_generator.load_state_dict(torch.load('./data/models/cDCGAN/MNIST_Extended_four_cDCGAN_generator_param_14.pkl', map_location=device)) # for all digits 7 & 9, with extention ratio 4

    return saved_generator

class SoftDataset(Dataset):
    def __init__(self, data_type, numData, data_dim):
        if data_type in ['circle', 'moon', 'v_shaped', 'bracket']:
            self.data = torch.zeros(numData, data_dim)
        elif data_type in ['mnist']:
            self.data = torch.zeros(numData, 1, data_dim, data_dim) # data_dim = image_size_ref, numCH =1 
        elif data_type in ['cifar10']:
            self.data = torch.zeros(numData, 3, data_dim, data_dim) # data_dim = image_size_ref, numCH = 3
        self.targets = torch.zeros(numData, 3) # store (target_a, target_b, lambda)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def load_synthetic_train_loader(data_, label_, data_type, n_samples, data_dim, train_type, device, my_middle_data = 0, early_stop = False):

    orig_dataset = TensorDataset(data_, label_)

    if train_type in ['GAN_mixup']:
        num_augment = len(my_middle_data)
        suggested_dataset = SoftDataset(data_type, n_samples + num_augment, data_dim)

        # fill in suggested_dataset
        suggested_dataset.data[:n_samples] = data_
        suggested_dataset.data[n_samples:] = torch.from_numpy(my_middle_data).to(device)

        suggested_dataset.targets[:n_samples, 0] = label_
        suggested_dataset.targets[:n_samples, 1] = label_
        suggested_dataset.targets[:n_samples, 2] = torch.ones([n_samples])
        #if early_stop == True:  suggested_dataset.data[n_samples:] = torch.from_numpy(my_middle_data[:,0:2]).to(device)  
        #else:   suggested_dataset.data[n_samples:] = torch.from_numpy(my_middle_data).to(device)
        suggested_dataset.targets[n_samples:, 0] = torch.zeros([num_augment])
        suggested_dataset.targets[n_samples:, 1] = torch.ones([num_augment])
        suggested_dataset.targets[n_samples:, 2] = 0.5 * torch.ones([num_augment])
        #if early_stop == True:  suggested_dataset.targets[n_samples:, 2] = torch.from_numpy(1-my_middle_data[:,2]).to(device)
        ##### my middle data is in ndim=2,,,, change function if we want to insert lambda in the my_middle_data
        #else:   suggested_dataset.targets[n_samples:, 2] = 0.5 * torch.ones([num_augment])    

        return suggested_dataset #train_loader, total_data_, total_label_


def train_GAN_mixup(model,optimizer,epoch,train_loader,n_samples,num_augment,num_class,device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    avg_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        
        target_a = target[:, 0].long()
        target_b = target[:, 1].long()
        lam = target[:, 2]
        loss = 0

        if num_augment != 0: weight = n_samples/num_augment

        #weight = (num_augment != 0)*n_samples/num_augment + (num_augment==0)*1
        indices_augment = np.where(lam.cpu()==0.5)[0]

        for idx in range(lam.size()[0]):
            if lam[idx] > 0 and lam[idx] < 1:
                coeff = weight
            else:
                coeff = 1
            loss += coeff * mixup_criterion(criterion, output[idx].view(-1,num_class), target_a[idx].view(-1), target_b[idx].view(-1), lam[idx])             
            #  loss = lam[0] * criterion(output, target_a) + (1 - lam[0]) * criterion(output, target_b)
            
        loss = loss/(lam.size()[0])
        avg_loss += loss.detach().data
        loss.backward()
        optimizer.step()
        
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))
            
    return avg_loss/len(train_loader)



def find_middle_points(saved_generator, data_type, data_class, curr_date, train_loader, num_class, show_image, num_class_total, z_dim, device, early_stop, distance_type, label_GAN, data_dim = 0, data_ = 0, label_ = 0):
    # ### For multiple trials, Generate Intermediate Data Points btw "label_first" and "label_second"
    print("find_middle_points() in utils.py: GAN has {} labels".format(label_GAN))

    label_first = target_label[0]
    label_second = target_label[1]

    num_trials = 1000
    out_dir = './data/%s_middle/%s/%s' % (data_type, curr_date, distance_type)
    Logger._make_dir(out_dir)

    if data_class == 'Synthetic':
        X_middle = torch.zeros([0,data_dim])

    approx_param = 10.   # alpha used in approximating max function to alpha-softmax
    
    for trial_iter in range(num_trials):
        print('======================= Trial %d =======================' % trial_iter)
        trial_start_time = time.time()

        ### Step 1. randomly choose a point from the dataset     
        for epoch in range(1):
            for n_batch,(data_batch,label_batch) in enumerate(train_loader):
                temp_first_data = data_batch[np.where(label_batch==label_first)]
                input_image = temp_first_data[0].reshape(1,data_dim)
                break
        
        else:
            if label_GAN == 2:
                if label_first==9: label_to_generate=1
                if label_first==7: label_to_generate=0
            
            elif label_GAN == 10:
                label_to_generate=label_first

            z_vec_test = torch.randn(1, z_dim, 1, 1)
            y_label_test = torch.zeros(label_GAN)
            y_label_test[label_to_generate]=1
            y_label_test = y_label_test.view(1,-1,1,1)
            z_vec_test, y_label_test = z_vec_test.to(device), y_label_test.to(device)
            print('norm of z_vec_test is: ', torch.norm(z_vec_test))
            input_image = saved_generator(z_vec_test, y_label_test).detach()
            print('torch.max(input_image[0]): ', torch.max(input_image[0]))
            print('torch.min(input_image[0]): ', torch.min(input_image[0]))

        ### Step 2. Search for equi-distance points between two classes, class "label_first" and "label_second"
        image_x = input_image
        num_epochs_x = 40

        for iter_x in range(num_epochs_x):
            print('iter: ', iter_x)
            start_epoch_time = time.time()

            # calculate "difference of distances" & model update
            ## Calculate z_1^*, z_2^*, r_1^*, r_2^*
            z_Var_1, proj_image_x_1, dist_1 = get_proj_distance_square(image_x, saved_generator, label_first, num_class_total, z_dim, data_type, data_class, device, distance_type, show_image, data_, label_)
            z_Var_1, proj_image_x_2, dist_2 = get_proj_distance_square(image_x, saved_generator, label_second, num_class_total, z_dim, data_type, data_class, device, distance_type, show_image, data_, label_) 

            print('image_x has norm ', torch.norm(image_x))
            print('point projected to the first manifold has norm ', torch.norm(proj_image_x_first))
            print('point projected to the second manifold has norm ', torch.norm(proj_image_x_second))

            ####### image_x, proj_image_x_1, proj_image_x_2 비교 plot line 345~398


            ## Calculate d_{12}= |r_1^* - r_2^*|
            ratio = dist_first/(dist_first + dist_second)
            dist_diff = np.absolute(dist_first.cpu() - dist_second.cpu())
            print('difference of distance (of LHS image): ', dist_diff.data)   
            print('ratio of d_0 / (d_0 + d_1): ', ratio.data)


            diff_max = 0.01


            if dist_diff < diff_max:
                finish_time = time.time() - trial_start_time  
                X_middle = torch.cat((X_middle.float(), image_x.float()), 0)
                print('The saved image has distance ', dist_diff, ' and point is ', image_x, ' which has norm ', torch.norm(image_x), 'the distance to the first manifold is ', dist_first, 'the distance to the second manifold is', dist_second)

                print('Trial %d ends at time %.4f' % (trial_iter, finish_time))
                
                break

            if distance_type == 'L2':
                grad_vec = -4 * (dist_first**2 - dist_second**2) * (proj_image_x_first.cpu()-proj_image_x_second.cpu())

                norm_grad = torch.norm(grad_vec)

            elif distance_type == 'L1':
                dist_vec_first = proj_image_x_first.cpu() - image_x   # G_0(z_0^*) - x
                dist_vec_second = proj_image_x_second.cpu() - image_x # G_1(z_1^*) - x
                sign_first = torch.sign(-dist_vec_first) # g^(0) = grad(dist_first)
                sign_second = torch.sign(-dist_vec_second) # g^(1) = grad(dist_second)
                grad_vec = 2 * (dist_first - dist_second) * (sign_first - sign_second)        
            elif distance_type == 'Linf':
                #pdb.set_trace()
                dist_vec_first = proj_image_x_first - image_x.to(device)   # G_0(z_0^*) - x
                sign_vec_first = - torch.sign(dist_vec_first)
                abs_dist_vec_first = torch.abs(dist_vec_first)
                summ_first = torch.sum(torch.exp(approx_param * abs_dist_vec_first))
                weight_summ_first = torch.sum( torch.mul(abs_dist_vec_first, torch.exp(approx_param * abs_dist_vec_first))  )
                grad_first = summ_first * torch.mul( torch.exp(approx_param * abs_dist_vec_first), 1 + approx_param *abs_dist_vec_first) - weight_summ_first * approx_param * torch.exp(approx_param * abs_dist_vec_first) 
                grad_first = torch.mul(grad_first, sign_vec_first)/(summ_first**2)

                dist_vec_second = proj_image_x_second - image_x.to(device)   # G_1(z_1^*) - x
                sign_vec_second = - torch.sign(dist_vec_second)
                abs_dist_vec_second = torch.abs(dist_vec_second)
                summ_second = torch.sum(torch.exp(approx_param * abs_dist_vec_second))
                weight_summ_second = torch.sum( torch.mul(abs_dist_vec_second, torch.exp(approx_param * abs_dist_vec_second))  )
                grad_second = summ_second * torch.mul( torch.exp(approx_param * abs_dist_vec_second), 1 + approx_param *abs_dist_vec_second) - weight_summ_second * approx_param * torch.exp(approx_param * abs_dist_vec_second) 
                grad_second = torch.mul(grad_second, sign_vec_second)/(summ_second**2)
                grad_vec = 2 * (dist_first - dist_second) * (grad_first - grad_second)

                norm_grad = torch.max(torch.abs(grad_vec))

            
            lr_x = lr_x_init * (0.8)**(divmod(iter_x,10)[0])
            image_x = image_x.float() - torch.tensor([lr_x]) * grad_vec #.cpu()

            if data_type in ['mnist', 'mnist_ext']:    image_x = torch.clamp(image_x, 0., 1.)

            print('Updated data point is ', image_x, 'having norm ', torch.norm(image_x))

            epoch_time = time.time() - start_epoch_time
            print('Trial %d: Epoch %d ends within %.4f' % (trial_iter, iter_x, epoch_time))

        #### store images and middle point information




def plot_synthetic_data(data,dclass,label,show_image):
    if dclass == 'Real': #np.size(data,1) != 2:
        fig = plt.subplot()
        plt.imshow(data.view(32,32),cmap='Greys', vmin=0, vmax=1)
    
    elif dclass == 'Synthetic':
        ##### yet we only have ways to show 0/1 2-class
        if np.ndim(data)==2:
            fig,ax = plt.subplots()
            plt.scatter(data[:,0].cpu(),data[:,1].cpu(),c=np.where(label.cpu()==0,'k',np.where(label.cpu()==1,'b','r')))
            plt.axis('equal')
        else:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(data[:,0].cpu(), data[:,1].cpu(), data[:,2].cpu(), c=np.where(label.cpu()==0,'k',np.where(label.cpu()==1,'b','r')))
        ####### MUST INCLUDE utils 270~297
    if show_image == True: plt.show()

    return fig