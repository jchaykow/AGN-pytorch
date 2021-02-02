from imports import *
from archs import *

def train_AGN(netG, netD, model_ft, dataloader, dataloader_me, class_names, nz, num_epochs:int=10, kap:float=0.2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Initialize BCEWithLogitsLoss function (better to use without softmax on end)
    criterion = nn.BCEWithLogitsLoss()
    criterionCW = CWLoss
    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0
    me_label = [i for i, el in enumerate(class_names) if el == 'Michael_Chaykowsky'][0]
    print('me_label: ', me_label)
    beta1 = 0.5
    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=5e-6, betas=(beta1, 0.99))
    optimizerG = optim.Adam(netG.parameters(), lr=5e-6, betas=(beta1, 0.99))
    # Un-normalizing
    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    renorm = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    # Get masks
    imgGlass = cv2.imread("data/glasses_mask.png", -1)
    r = 160.0 / imgGlass.shape[1]
    dim = (160, int(imgGlass.shape[0] * r))
    imgGlass = cv2.resize(imgGlass, dim, interpolation = cv2.INTER_AREA)
    imgGlass = imgGlass[39:81, 21:138]
    alpha_data = imgGlass[:,:,0] + imgGlass[:,:,1] + imgGlass[:,:,2]
    alpha_data[alpha_data < 200] = 0
    alpha_data[alpha_data > 20] = 255
    orig_mask_g = alpha_data
    orig_mask_inv_g = cv2.bitwise_not(orig_mask_g)
    orig_mask_g = orig_mask_g / 255
    orig_mask_inv_g = orig_mask_inv_g / 255

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    d1s = []
    d3s = []
    num_fooled = []
    iters = 0
    counter = 0

    # for p in netG.parameters(): 
    #     p.register_hook(lambda grad: print(torch.norm(grad)))
    #     break
    
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader):
            if data.shape[0] != 64: continue
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            # (2) Check if F(.) is fooled
            # Get a batch of faces and affix glasses to them in correct positions
            # all image transformations must be done using pytorch functions only
            model_ft.eval()
            fakes = unorm_glasses(fake)
            fakes = fakes[:,:,39:81,21:138]
            for j in range(fakes.size(0)):
                for k in range(fakes.size(1)):
                    fakes[j,k,:,:][orig_mask_g == 0] = 0
            faces, landmarks, labels = next(iter(dataloader_me))
            for j in range(faces.size(0)):
                img = unorm(faces[j,:,:,:]).cpu()
                glassHeight,glassWidth = landmarks[j,-2:].int()
                x1,x2,y1,y2 = landmarks[j,:-2].int()
                glass = F.interpolate(fakes, (glassHeight,glassWidth)).cpu()
                mask = F.interpolate(T(orig_mask_g[None,None,:,:]), (glassHeight,glassWidth))
                mask_inv = F.interpolate(T(orig_mask_inv_g[None,None,:,:]), (glassHeight,glassWidth))
                roi1 = img[None,:,y1:y2, x1:x2]
                roi_bg = roi1 - mask
                roi_bg = torch.clamp(roi_bg, 0)
                roi_fg = glass + mask_inv
                img[:,y1:y2, x1:x2] = glass[j] + roi_bg[0]
                faces[j,:,:,:] = img
            faces = renorm(faces).to(device)
            # Check to see how the generator is doing in a nograd environment
            with torch.no_grad():
                outputs = model_ft(faces)
                _, preds = torch.max(outputs, 1)
                if i % 100 == 0: img_list.append(utils.make_grid(faces.detach().cpu(), padding=2, normalize=True))
                print('Num Fooled: ', torch.sum(preds != me_label).item(), 
                      'Sum prob me: ', outputs[:,me_label].sum().item(), 
                      'Sum prob targ: ', outputs[:,7].sum().item())
                if torch.all(preds != me_label):
                    return netG, img_list, G_losses, D_losses, d1s, d3s, num_fooled

            # (3) Update G network
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            d1 = criterion(output, label)
            d1.backward(retain_graph=True)
            # Take faces with glasses and forward pass through F()
            labelXsb = torch.full((b_size,), me_label, device=device)    # CW Loss Label
            outputs = model_ft(faces)
            d3 = criterionCW(outputs, labelXsb, b_size, is_targeted=False, num_classes=len(class_names))
            d3.backward()
            errG = d1 + d3
            # Update G
            optimizerG.step()
            D_G_z2 = output.mean().item()
            # Output training stats
            if i % 100 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            d1s.append(d1.item())
            d3s.append(d3.item())
            num_fooled.append(torch.sum(preds != me_label).item())
            
            if iters == 20:
                return netG, img_list, G_losses, D_losses, d1s, d3s, num_fooled

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 1000 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(utils.make_grid(fake, padding=2, normalize=True))

            iters += 1
    return netG, img_list, G_losses, D_losses, d1s, d3s, num_fooled
