import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F

class generator_circle(nn.Module):
    # initializers
    def __init__(self, noise_dim, num_class, data_dim):
        super(generator_circle, self).__init__()
        self.fc1_1 = nn.Linear(noise_dim, 256)
        self.fc1_1_bn = nn.BatchNorm1d(256)
        self.fc1_2 = nn.Linear(num_class, 256)
        self.fc1_2_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(512, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, data_dim)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.relu(self.fc1_1_bn(self.fc1_1(input)))
        y = F.relu(self.fc1_2_bn(self.fc1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        #x = torch.tanh(self.fc4(x))
        x = self.fc4(x)

        return x

class generator_moon(nn.Module):
    # initializers
    def __init__(self, noise_dim, num_class, data_dim):
        super(generator_moon, self).__init__()
        self.fc1_1 = nn.Linear(noise_dim, 512)
        self.fc1_1_bn = nn.BatchNorm1d(512)
        self.fc1_2 = nn.Linear(num_class, 512)
        self.fc1_2_bn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc2_bn = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, data_dim)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.relu(self.fc1_1_bn(self.fc1_1(input)))
        y = F.relu(self.fc1_2_bn(self.fc1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        #x = torch.tanh(self.fc4(x))
        x = self.fc4(x)

        return x    

    
class generator_moon_complex(nn.Module):
    # initializers
    def __init__(self, noise_dim, num_class, data_dim):
        super(generator_moon_complex, self).__init__()
        self.fc1_1 = nn.Linear(noise_dim, 256)
        self.fc1_1_bn = nn.BatchNorm1d(256)
        self.fc1_2 = nn.Linear(num_class, 256)
        self.fc1_2_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(512, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc3_bn = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, data_dim)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.relu(self.fc1_1_bn(self.fc1_1(input)))
        y = F.relu(self.fc1_2_bn(self.fc1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = self.fc4(x)
        #x = torch.tanh(self.fc4(x))

        return x

class generator_braket_complex(nn.Module):
    # initializers
    def __init__(self, noise_dim, num_class, data_dim):
        super(generator_braket_complex, self).__init__()
        self.fc1_1 = nn.Linear(noise_dim, 256)
        self.fc1_1_bn = nn.BatchNorm1d(256)
        self.fc1_2 = nn.Linear(num_class, 256)
        self.fc1_2_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(512, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc3_bn = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, data_dim)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.relu(self.fc1_1_bn(self.fc1_1(input)))
        y = F.relu(self.fc1_2_bn(self.fc1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = self.fc4(x)
        #x = F.tanh(self.fc4(x))

        return x    

# G(z)
class conditional_DC_generator(nn.Module): ## Conditional DC-GAN (c-DCGAN)
    # initializers
    def __init__(self, d=128, label_GAN=10):
        super(conditional_DC_generator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(100, d*2, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d*2)
        self.deconv1_2 = nn.ConvTranspose2d(label_GAN, d*2, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d*2)
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x))
        # x = F.relu(self.deconv4_bn(self.deconv4(x)))
        # x = F.tanh(self.deconv5(x))

        
        x = (x+1)/2
        
        return x
                
    
def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
        
        
        
# class simple_Ndim_Net(torch.nn.Module):
#     def __init__(self, num_layer, data_dim, num_class):
#         super(simple_Ndim_Net, self).__init__()
        
#         self.num_layer = num_layer
#         self.data_dim = data_dim
#         self.num_class = num_class
        
#         self.fc1 = torch.nn.Linear(self.data_dim, 64)
#         self.fc2 = torch.nn.Linear(64, 128)
#         if self.num_layer >= 4:
#             self.fc3 = torch.nn.Linear(128, 128)
#             self.fc4 = torch.nn.Linear(128, 128)
#         if self.num_layer == 6:
#             self.fc5 = torch.nn.Linear(128, 128)
#             self.fc6 = torch.nn.Linear(128, 128)        
#         self.fc7 = torch.nn.Linear(128, self.num_class)
        
#     def forward(self, x, target=None, device = None, mixup_hidden = False,  mixup_alpha = 0.1, layer_mix=None):


#         if mixup_hidden == True:

#             if layer_mix == None:
#                 layer_mix = random.randint(0,2) 
#             out = x
            
#             if layer_mix == 0:
#                 out, y_a, y_b, lam = mixup_data(out, target, device, mixup_alpha)
#             out = F.relu(self.fc1(out))
    
#             if layer_mix == 1:
#                 out, y_a, y_b, lam = mixup_data(out, target, device, mixup_alpha)
#             out = F.relu(self.fc2(out))
    
#             if layer_mix == 2:
#                 out, y_a, y_b, lam = mixup_data(out, target, device, mixup_alpha)
#             out = F.relu(self.fc3(out))
#             out = F.relu(self.fc4(out))


#             if self.num_layer == 6:
#                 out = F.relu(self.fc5(out))
#                 out = F.relu(self.fc6(out))
#             out = self.fc7(out)

#             lam = torch.tensor(lam).to(device)
#             lam = lam.repeat(y_a.size())
#             #print (out.shape)
#             #print (y_a.shape)
#             #print (y_b.size()) 
#             #print (lam.size())
#             return out, y_a, y_b, lam

#         else:
#             x = F.relu(self.fc1(x))
#             x = F.relu(self.fc2(x))
#             x = F.relu(self.fc3(x))
#             x = F.relu(self.fc4(x))
#             if self.num_layer == 6:
#                 x = F.relu(self.fc5(x))
#                 x = F.relu(self.fc6(x))
#             x = self.fc7(x)

#             return x

    
class LeNet(nn.Module):
    def __init__(self, num_class):
        super(LeNet, self).__init__()
        # input channel = 1, output channel = 6, kernel_size = 5
        # input size = (32, 32), output size = (28, 28)
        self.conv1 = nn.Conv2d(1, 6, 5)
        # input channel = 6, output channel = 16, kernel_size = 5
        # input size = (14, 14), output size = (10, 10)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # input dim = 16*5*5, output dim = 120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # input dim = 120, output dim = 84
        self.fc2 = nn.Linear(120, 84)
        # input dim = 84, output dim = num_class (10 or 2)
        self.fc3 = nn.Linear(84, num_class)

    def forward(self, x):
        # pool size = 2
        # input size = (28, 28), output size = (14, 14), output channel = 6
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # pool size = 2
        # input size = (10, 10), output size = (5, 5), output channel = 16
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # flatten as one dimension
        x = x.view(x.size()[0], -1)
        # input dim = 16*5*5, output dim = 120
        x = F.relu(self.fc1(x))
        # input dim = 120, output dim = 84
        x = F.relu(self.fc2(x))
        # input dim = 84, output dim = num_class (10 or 2)
        x = self.fc3(x)
        return x