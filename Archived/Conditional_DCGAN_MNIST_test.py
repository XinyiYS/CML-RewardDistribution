import torch
import matplotlib.pyplot as plt

# entrypoints = torch.hub.list('facebookresearch/pytorch_GAN_zoo', force_reload=True)
# print(entrypoints)


# load the models
from models.Conditional_DCGAN_MNIST import Discriminator, Generator

D = Discriminator(ngpu=1,nc=1+10).eval()
G = Generator(ngpu=1).eval()

# load weights
D.load_state_dict(torch.load('checkpoints/netD_epoch_49.pth'))
G.load_state_dict(torch.load('checkpoints/netG_epoch_49.pth'))

batch_size = 50
latent_size = 100

fixed_noise = torch.randn(batch_size, latent_size, 1, 1)

labels = torch.tensor( [[i]*5 for i in range(10)] ,dtype=int).reshape(50)
G_target = torch.nn.functional.one_hot(labels, 10)
G_target = G_target.unsqueeze(2).unsqueeze(3)

if torch.cuda.is_available():
    D = D.cuda()
    G = G.cuda()
    fixed_noise = fixed_noise.cuda()
    G_target = G_target.cuda()
fake_images = G(fixed_noise, G_target)


fake_images_np = fake_images.cpu().detach().numpy()
fake_images_np = fake_images_np.reshape(fake_images_np.shape[0], 28, 28)
R, C = 10, 5
for i in range(batch_size):
    plt.subplot(R, C, i + 1)
    plt.imshow(fake_images_np[i], cmap='gray')
plt.show()