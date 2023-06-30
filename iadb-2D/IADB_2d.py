import imageio 
import matplotlib.pyplot
import numpy as np
import torch
from tqdm import tqdm

# load PNG image
def loadImage(filename):
    image = imageio.imread(filename).astype("float32")[:, :, 0:3] / 255.0
    image = image[np.newaxis, ...]
    return image

# generate Ndata from PDF p represented as an image
# using rejection sampling
def generateSamplesFromImage(p, Ndata):
    maxPDFvalue = np.max(p)
    samples = torch.zeros(Ndata, 2).to("cuda")
    for n in tqdm(range(Ndata), "generateSamplesFromImage"):
        while True:
            # random location in [0, 1]²
            x = np.random.rand()
            y = np.random.rand()
            # discrete pixel coordinates of (x,y)
            i = int(x * p.shape[1])
            j = int(y * p.shape[2])
            # random value
            u = np.random.rand()
            # keep or reject?
            if p[0,i,j,0]/maxPDFvalue >= u: 
                samples[n,0] = x
                samples[n,1] = y
                break
    return samples

# create plot with samples
def export(x, filename):
    axes = matplotlib.pyplot.gca()
    axes.set_xlim([0, 1])
    axes.set_ylim([0, 1])
    axes.set_aspect('equal', adjustable='box')
    x_numpy = x.detach().cpu().clone().numpy()
    for i in range(x_numpy.shape[0]):
        matplotlib.pyplot.plot(x_numpy[i,1], 1-x_numpy[i,0], 'bo', markersize=1)
    matplotlib.pyplot.savefig(filename)
    matplotlib.pyplot.clf()


# data loading
p_0 = loadImage("p0.png")
p_1 = loadImage("p1.png")
Ndata = 65536
x_0_data = generateSamplesFromImage(p_0, Ndata)
x_1_data = generateSamplesFromImage(p_1, Ndata)
export(x_0_data[0:2048], "x0.png")
export(x_1_data[0:2048], "x1.png")

# architecture
class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2+1,64)  # input = (x_alpha, alpha)
        self.linear2 = torch.nn.Linear(64, 64)  
        self.linear3 = torch.nn.Linear(64, 64)   
        self.linear4 = torch.nn.Linear(64, 64)   
        self.output  = torch.nn.Linear(64, 2) 
        self.relu = torch.nn.ReLU()

    def forward(self, x, alpha):
        res = torch.cat([x, alpha], dim=1)
        res = self.relu(self.linear1(res))
        res = self.relu(self.linear2(res))
        res = self.relu(self.linear3(res))
        res = self.relu(self.linear4(res))
        res = self.output(res)
        return res

# allocating the neural network D
D = NN().to("cuda")
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.001)

# training loop
batchsize = 256
for iteration in tqdm(range(65536), "training loop"):

    
    x_0 = x_0_data[np.random.randint(0, Ndata, batchsize), :]
    x_1 = x_1_data[np.random.randint(0, Ndata, batchsize), :]
    alpha = torch.rand(batchsize, 1, device="cuda")
    x_alpha = (1-alpha) * x_0 + alpha * x_1

    loss = torch.sum( (D(x_alpha, alpha) - (x_1-x_0))**2 )
    optimizer_D.zero_grad()
    loss.backward()
    optimizer_D.step()

# sampling loop
batchsize = 2048
with torch.no_grad():
    # starting points x_alpha = x_0
    x_alpha = x_0_data[np.random.randint(0, Ndata, batchsize), :]

    # loop
    T = 128
    for t in tqdm(range(T), "sampling loop"):

        # export plot
        export(x_alpha, "x_" + str(t) + ".png")

        # current alpha value
        alpha = t / T * torch.ones(batchsize, 1, device="cuda")

        # update 
        x_alpha = x_alpha + 1/T * D(x_alpha, alpha)

    export(x_alpha, "x_" + str(T) + ".png")