import os
import glob
import random
import torch
import itertools
import datetime
import time
import sys
import argparse
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
from torchvision.utils import save_image, make_grid

#########################################################################
############################  datasets.py  ###################################

## If the input dataset consists of grayscale images, convert them to RGB images (not needed for the facades dataset used in this case)
def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

## Build the dataset
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        # List of file paths for domains A and B
        self.files_A = sorted(glob.glob(os.path.join(root, "%sA" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%sB" % mode) + "/*.*"))

        print("Found files in {}: A - {}, B - {}".format(root, len(self.files_A), len(self.files_B)))

        # If files are not found, print the paths to aid in debugging
        if len(self.files_A) == 0 or len(self.files_B) == 0:
            print("No files found in directories: {}A and {}B".format(root, root))

    def __getitem__(self, index):
        # Load images from domains A and B
        image_A = Image.open(self.files_A[index % len(self.files_A)])  ## Take one photo from A

        if self.unaligned:  ## If using non-paired data, randomly select one from B
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert grayscale images to RGB
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        # Apply transformations and return as dictionary
        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

    ## Get the length of A and B data
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

#########################################################################
############################  models.py  ###################################

## Define parameter initialization function
def weights_init_normal(m):
    classname = m.__class__.__name__  ## m as a parameter, in principle, it can pass a lot of content. To implement multi-parameter passing, each module needs to provide its name. So this sentence returns the name of m.
    if classname.find("Conv") != -1:  ## find(): Check if the classname contains the string "Conv", return -1 if not found; return 0 if found.
        torch.nn.init.normal_(m.weight.data, 0.0,
                              0.02)  ## m.weight.data represents the weights that need to be initialized. nn.init.normal_(): Initializes using a normal distribution with mean 0 and standard deviation 0.02.
        if hasattr(m, "bias") and m.bias is not None:  ## hasattr(): Used to determine if m contains the bias attribute, and whether the bias attribute is not empty.
            torch.nn.init.constant_(m.bias.data, 0.0)  ## nn.init.constant_(): Set bias as a constant 0.
    elif classname.find("BatchNorm2d") != -1:  ## find(): Check if the classname contains the string "BatchNorm2d", return -1 if not found; return 0 if found.
        torch.nn.init.normal_(m.weight.data, 1.0,
                              0.02)  ## m.weight.data represents the weights that need to be initialized. nn.init.normal_(): Initializes using a normal distribution with mean 1.0 and standard deviation 0.02.
        torch.nn.init.constant_(m.bias.data, 0.0)  ## nn.init.constant_(): Set bias as a constant 0.

##############################
## Residual Block ResidualBlock
##############################
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(  ## block = [pad + conv + norm + relu + pad + conv + norm]
            nn.ReflectionPad2d(1),  ## ReflectionPad2d(): Pads the input tensor boundaries using reflection of the input tensor
            nn.Conv2d(in_features, in_features, 3),  ## Convolution
            nn.InstanceNorm2d(in_features),  ## InstanceNorm2d(): Normalization over HW on image pixels, used in style transfer
            nn.ReLU(inplace=True),  ## Non-linear activation
            nn.ReflectionPad2d(1),  ## ReflectionPad2d(): Pads the input tensor boundaries using reflection of the input tensor
            nn.Conv2d(in_features, in_features, 3),  ## Convolution
            nn.InstanceNorm2d(in_features),  ## InstanceNorm2d(): Normalization over HW on image pixels, used in style transfer
        )

    def forward(self, x):  ## Input: One image
        return x + self.block(x)  ## Output: Image plus the residual output of the network

##############################
## Generator Network GeneratorResNet
##############################
class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):  ## (input_shape = (3, 256, 256), num_residual_blocks = 9)
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]  ## Input channel count, channels = 3

        ## Initialize the network structure
        out_features = 64  ## Output feature count, out_features = 64
        model = [  ## model = [Pad + Conv + Norm + ReLU]
            nn.ReflectionPad2d(channels),  ## ReflectionPad2d(3): Pads the input tensor boundaries using reflection of the input tensor
            nn.Conv2d(channels, out_features, 7),  ## Conv2d(3, 64, 7)
            nn.InstanceNorm2d(out_features),  ## InstanceNorm2d(64): Normalization over HW on image pixels, used in style transfer
            nn.ReLU(inplace=True),  ## Non-linear activation
        ]
        in_features = out_features  ## in_features = 64

        ## Downsampling, loop 2 times
        for _ in range(2):
            out_features *= 2  ## out_features = 128 -> 256
            model += [  ## (Conv + Norm + ReLU) * 2
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features  ## in_features = 256

        # Residual blocks, loop 9 times
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]  ## model += [pad + conv + norm + relu + pad + conv + norm]

        # Upsampling two times
        for _ in range(2):
            out_features //= 2  ## out_features = 128 -> 64
            model += [  ## model += [Upsample + conv + norm + relu]
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features  ## out_features = 64

        ## Network output layer                                                ## model += [pad + conv + tanh]
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7),
                  nn.Tanh()]  ## Maps each data point in (3) to the range [-1, 1]

        self.model = nn.Sequential(*model)

    def forward(self, x):  ## Input: (1, 3, 256, 256)
        return self.model(x)  ## Output: (1, 3, 256, 256)

##############################
#        Discriminator
##############################
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape  ## input_shape:(3, 256, 256)

        # Calculate the output shape of the image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)  ## output_shape = (1, 16, 16)

        def discriminator_block(in_filters, out_filters, normalize=True):  ## Discriminator block
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]  ## layers += [conv + norm + relu]
            if normalize:  ## Every convolution reduces the size by half, a total of 4 convolutions
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),  ## layers += [conv(3, 64) + relu]
            *discriminator_block(64, 128),  ## layers += [conv(64, 128) + norm + relu]
            *discriminator_block(128, 256),  ## layers += [conv(128, 256) + norm + relu]
            *discriminator_block(256, 512),  ## layers += [conv(256, 512) + norm + relu]
            nn.ZeroPad2d((1, 0, 1, 0)),  ## layers += [pad]
            nn.Conv2d(512, 1, 4, padding=1)  ## layers += [conv(512, 1)]
        )

    def forward(self, img):  ## Input: (1, 3, 256, 256)
        return self.model(img)  ## Output: (1, 1, 16, 16)

#########################################################################
############################  utils.py  ###################################

## Previously generated sample buffer
class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):  ## Put in one image and then retrieve one from the buffer
        """
        Push a data element into the buffer and pop a random element if the buffer is full.
        This helps to maintain diversity in the generated samples.
        """
        to_return = []  ## Ensure the randomness of the data for discriminator recognition of real and fake images
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:  ## Up to 50 images, keep adding if not full
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:  ## If full, with a 1/2 probability, retrieve from the buffer, or use the current input image
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

## Learning rate scheduler
class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        """
        Lambda function for adjusting learning rates over epochs.
        It reduces the learning rate linearly after a certain epoch.
        """
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

###########################################################################
############################  cycle_gan.py  ###################################

## Hyperparameters configuration
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="./datasets/facades", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0003, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first-order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first-order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=3, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in the generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
opt = parser.parse_args()
print(opt)

## Create folders
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("save/%s" % opt.dataset_name, exist_ok=True)

## Input shape: (3, 256, 256)
input_shape = (opt.channels, opt.img_height, opt.img_width)

## Create generator and discriminator objects
G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

## Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

## Move models to GPU if available
if torch.cuda.is_available():
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

## Load pre-trained models if epoch != 0, otherwise, initialize model parameters
if opt.epoch != 0:
    # Load pretrained models trained up to epoch n
    G_AB.load_state_dict(torch.load("save/%s/G_AB_%d.pth" % (opt.dataset_name, opt.epoch)))
    G_BA.load_state_dict(torch.load("save/%s/G_BA_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_A.load_state_dict(torch.load("save/%s/D_A_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_B.load_state_dict(torch.load("save/%s/D_B_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # Initialize model parameters
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

## Define optimizers with a learning rate of 0.0003
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

## Learning rate schedulers
# The above code is a comment in Python. It describes the logic for initializing model parameters if
# the epoch is 0 and loading a pre-trained model trained up to the nth epoch if the epoch is equal to
# n.
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

## Previously generated sample buffer
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

## Image transformations
transforms_ = [
transforms.Resize(int(opt.img_height * 1.12)),  ## Enlarge the image by 1.12 times
transforms.RandomCrop((opt.img_height, opt.img_width)),  ## Randomly crop the image to its original size
transforms.RandomHorizontalFlip(),  ## Randomly flip the image horizontally
transforms.ToTensor(),  ## Convert the image to a Tensor
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  ## Normalize the image
]

## Training data loader
dataloader = DataLoader(  ## Change to the directory where your files are stored
    ImageDataset("./datasets/facades", transforms_=transforms_, unaligned=True),
    ## "./datasets/facades", unaligned: Set to True for non-aligned data
    batch_size=opt.batch_size,  ## batch_size = 1
    shuffle=True,
    num_workers=opt.n_cpu,
)
## Test data loader
val_dataloader = DataLoader(
    ImageDataset("./datasets/facades", transforms_=transforms_, unaligned=True, mode="test"),  ## "./datasets/facades"
    batch_size=5,
    shuffle=True,
    num_workers=1,
)

## Print images every 100 iterations
def sample_images(batches_done):  ## (100/200/300/400...)
    """Save samples generated from the test set"""
    imgs = next(iter(val_dataloader))  ## Take one image
    G_AB.eval()
    G_BA.eval()
    real_A = Variable(imgs["A"]).cuda()  ## Take a real A
    fake_B = G_AB(real_A)  ## Generate fake B from real A
    real_B = Variable(imgs["B"]).cuda()  ## Take a real B
    fake_A = G_BA(real_B)  ## Generate fake A from real B
    # Arrange images along x-axis
    ## make_grid(): Used to arrange several images in a grid-like fashion for visualization
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    # Arrange images along y-axis
    ## Concatenate the above images to save as one large image
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, "images/%s/%s.png" % (opt.dataset_name, batches_done), normalize=False)


def train():
    # ----------
    #  Training
    # ----------
    prev_time = time.time()  ## Record the start time of training
    for epoch in range(opt.epoch, opt.n_epochs):  ## Loop through multiple epochs of training
        for i, batch in enumerate(dataloader):  ## Iterate through the training dataset
            ## batch is a dictionary, batch['A']:(1, 3, 256, 256), batch['B']:(1, 3, 256, 256)
            # Read real images
            #       print('here is %d' % i)
            ## Read real images from the dataset
            ## Convert tensors to Variables and put them into the computation graph. After conversion, tensors become variables and can be used for backpropagation to calculate gradients.

            real_A = Variable(batch["A"]).cuda()  ## Real image A
            real_B = Variable(batch["B"]).cuda()  ## Real image B

            # Define real and fake labels
            valid = Variable(torch.ones((real_A.size(0), *D_A.output_shape)),
                             requires_grad=False).cuda()  ## Define the label for real images as 1
            fake = Variable(torch.zeros((real_A.size(0), *D_A.output_shape)),
                            requires_grad=False).cuda()  ## Define the label for fake images as 0
            ## -----------------
            ##  Train Generator
            ## Principle: The goal is to have the generated fake images classified as real by the discriminator.
            ## In this process, the discriminator is kept fixed. Fake images are passed into the discriminator, and the results are compared with real labels.
            ## Backpropagation updates the parameters of the generator network.
            ## This process trains the network to generate images that the discriminator perceives as real, achieving the adversarial goal.
            ## -----------------
            G_AB.train()
            G_BA.train()

            ## Identity loss                                              ## Images in the style of A put into the B -> A generator should result in images with the style of A
            loss_id_A = criterion_identity(G_BA(real_A),
                                        real_A)  ## loss_id_A ensures that when image A1 is put into the B2A generator, the generated image A2 should have the style of A, minimizing the difference between A1 and A2
            loss_id_B = criterion_identity(G_AB(real_B), real_B)

            loss_identity = (loss_id_A + loss_id_B) / 2  ## Identity loss

            ## GAN loss
            fake_B = G_AB(real_A)  ## Fake image B generated from real image A
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)  ## Discriminator B classifies the fake image B. The goal is to train the generator so that the discriminator thinks fake is real, making it difficult for the discriminator to distinguish between them.
            fake_A = G_BA(real_B)  ## Fake image A generated from real image B
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)  ## Discriminator A classifies the fake image A. The goal is to train the generator so that the discriminator thinks fake is real, making it difficult for the discriminator to distinguish between them.

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2  ## GAN loss

            # Cycle loss
            recov_A = G_BA(fake_B)  ## Previously, real A through A -> B generates fake image B, then through B -> A, resulting in the cyclic image recov_A.
            loss_cycle_A = criterion_cycle(recov_A, real_A)  ## The difference between real A and recov_A should be small, ensuring that styles change between A and B while preserving corresponding details.
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss                                              
            ## The total loss is the sum of all the individual losses above
            loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity
            optimizer_G.zero_grad()  ## Reset gradients to zero before backpropagation
            loss_G.backward()  ## Backpropagate the error
            optimizer_G.step()  ## Update parameters

            ## -----------------------
            ## Train Discriminator A
            ## Divided into two parts: 1. Real images classified as real; 2. Fake images classified as fake
            ## -----------------------
            ## Real images classified as real
            loss_real = criterion_GAN(D_A(real_A), valid)
            ## Fake images classified as fake (randomly sampled from the buffer cache)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2
            optimizer_D_A.zero_grad()  ## Reset gradients to zero before backpropagation
            loss_D_A.backward()  ## Backpropagate the error
            optimizer_D_A.step()  ## Update parameters

            ## -----------------------
            ## Train Discriminator B
            ## Divided into two parts: 1. Real images classified as real; 2. Fake images classified as fake
            ## -----------------------
            # Real images classified as real
            loss_real = criterion_GAN(D_B(real_B), valid)
            ## Fake images classified as fake (randomly sampled from the buffer cache)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2
            optimizer_D_B.zero_grad()  ## Reset gradients to zero before backpropagation
            loss_D_B.backward()  ## Backpropagate the error
            optimizer_D_B.step()  ## Update parameters
            loss_D = (loss_D_A + loss_D_B) / 2

            ## ----------------------
            ##  Log Progress
            ## ----------------------

            ## Estimate remaining time: Assuming current epoch = 5, i = 100
            batches_done = epoch * len(dataloader) + i  ## Total batches trained so far: 5 * 400 + 100
            batches_left = opt.n_epochs * len(dataloader) - batches_done  ## Batches remaining: 50 * 400 - 2100
            time_left = datetime.timedelta(
                seconds=batches_left * (time.time() - prev_time) 
            ) ## Calculate the estimated remaining time: time_left = remaining iterations * time per iteration

            prev_time = time.time()
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_cycle.item(),
                    loss_identity.item(),
                    time_left,
                )
            )
            # Save a set of images from the test set every 100 training iterations

            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)

        # Update learning rate
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

    # Save models after training
    torch.save(G_AB.state_dict(), "save/%s/G_AB_%d.pth" % (opt.dataset_name, epoch))
    torch.save(G_BA.state_dict(), "save/%s/G_BA_%d.pth" % (opt.dataset_name, epoch))
    torch.save(D_A.state_dict(), "save/%s/D_A_%d.pth" % (opt.dataset_name, epoch))
    torch.save(D_B.state_dict(), "save/%s/D_B_%d.pth" % (opt.dataset_name, epoch))
    print("\nModel saved!")
    # Save the model every few epochs
    #     if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
    #         # Save model checkpoints
    #         torch.save(G_AB.state_dict(), "saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, epoch))
    #         torch.save(G_BA.state_dict(), "saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, epoch))
    #         torch.save(D_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (opt.dataset_name, epoch))
    #         torch.save(D_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (opt.dataset_name, epoch))

def test():
    ## Hyperparameter settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=4, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='./datasets/facades', help='root directory of the dataset')
    parser.add_argument('--channels', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of channels of output data')
    parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
    parser.add_argument('--cuda', type=bool, default=True, help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of CPU threads to use during batch generation')
    parser.add_argument('--generator_A2B', type=str, default='save/dataset/facades/G_AB_4.pth',
                        help='A2B generator checkpoint file')
    parser.add_argument('--generator_B2A', type=str, default='save/dataset/facades/G_BA_4.pth',
                        help='B2A generator checkpoint file')
    opt = parser.parse_args()
    print(opt)

    #################################
    ##       Preparation for Testing       ##
    #################################

    ## input_shape: (3, 256, 256)
    input_shape = (opt.channels, opt.size, opt.size)
    ## Create generator, discriminator objects
    netG_A2B = GeneratorResNet(input_shape, opt.n_residual_blocks)
    netG_B2A = GeneratorResNet(input_shape, opt.n_residual_blocks)

    ## Use CUDA
    if opt.cuda:
        netG_A2B.cuda()
        netG_B2A.cuda()

    ## Load trained model parameters
    netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
    netG_B2A.load_state_dict(torch.load(opt.generator_B2A))

    ## Set to evaluation mode
    netG_A2B.eval()
    netG_B2A.eval()

    ## Create a tensor array
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batchSize, opt.channels, opt.size, opt.size)
    input_B = Tensor(opt.batchSize, opt.channels, opt.size, opt.size)

    '''Build the test dataset'''
    transforms_ = [transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test'),
                            batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)

    #################################
    ##       Testing Starts        ##
    #################################

    '''Create directories if they don't exist (to store test output images)'''
    if not os.path.exists('output/A'):
        os.makedirs('output/A')
    if not os.path.exists('output/B'):
        os.makedirs('output/B')

    for i, batch in enumerate(dataloader):
        ## Input data: real
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))
        ## Fake images generated by the generator
        fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)
        fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)
        ## Save images
        save_image(fake_A, 'output/A/%04d.png' % (i + 1))
        save_image(fake_B, 'output/B/%04d.png' % (i + 1))
        print('processing (%04d)-th image...' % (i))
    print("Testing completed")

## Function starts here
if __name__ == '__main__':
    print("Training set size:", len(dataloader))
    print("Validation set size:", len(val_dataloader))
    train()  ## Train
    test()   ## Test
