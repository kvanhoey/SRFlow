import numpy as np


def create_mesh_coordinates(w, h):
    x,y = np.mgrid[0:w:1, 0:h:1]
    coords = np.empty([2,w,h])
    coords[0,...] = x / w
    coords[1,...] = y / h
    return coords

# # shadow
# with torch.no_grad():
#     Ncosines = 16
#     image_shadow = torch.zeros(B, 1, WIDTH, WIDTH, device=device)
#     directions = 10.0*torch.randn(B, 2, Ncosines, device=device)
#     phase = 6.2831 * torch.rand(B, Ncosines, device=device)
#     image_shadow = torch.sum(torch.cos(phase[:,:,None,None]+torch.einsum('btn,btwh->bnwh', directions, image_coordinates)), dim=1, keepdim=True)
#     image_shadow = 0.2 + 0.8 * torch.heaviside(image_shadow, torch.tensor([0.0]).to(device)) * (0.5+0.5*torch.rand(B, 1, 1, 1, device=device))
#     image_shadowed = image_shadow * image

def add_shadow(img):
    Ncosines = 16
    image_coordinates = create_mesh_coordinates(img.shape[0],img.shape[1])
    image_shadow = np.zeros_like(img)
    directions = 10.0 * np.random.randn(2, Ncosines)
    phase = 6.2831 * np.random.rand(Ncosines)
    image_shadow = np.sum((np.cos(phase[...,None,None]+np.einsum('tn,twh->nwh', directions, image_coordinates))),axis=0,keepdims=True)
    image_shadow = 0.2 + 0.8 * np.heaviside(image_shadow, 0.0) * (0.5+0.5*np.random.rand(1, 1, 1))
    image_shadowed = np.transpose(image_shadow,(1,2,0)) * img
    return image_shadowed



import matplotlib.pyplot as plt
I = 0.9*np.ones([128,128,3])

plt.imshow(I)
plt.imshow(add_shadow(I))
