from imports import *
from archs import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def run_test_adv_G():
    """Function to test the adversarial generator after it is trained."""
    netG = Generator(1).to(device)
    checkpointA = torch.load(args.generator, map_location=lambda storage, loc: storage) 
    netG.load_state_dict(checkpointA['state_dict'])
    # Test out resulting netG
    # make guassian noice output to pass into generator
    fixed_noise = torch.randn(1, nz, 1, 1, device=device)
    # pass noise through generator to output adversarial glasses
    fake = netG(fixed_noise).detach().cpu()
    # view output
    plt.imshow(np.transpose(utils.make_grid(fake, padding=2, normalize=True),(1,2,0)))
    # transformations for the images
    t_img = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    # transformations for the bounding boxes
    t_landmarks = transforms.Compose([
        transforms.ToTensor()
    ])
    # create dataloader of my face images with bounding box information for where my eyes are
    data_dir = 'data/agn_me'
    image_datasets_me = MeDataset(
        'data/bboxes_fnames.csv', 
        'data/agn_me_extras160/Michael_Chaykowsky', 
        bs=1, 
        transform_img=t_img, 
        transform_land=t_landmarks)
    dataloader_me_test = torch.utils.data.DataLoader(image_datasets_me, batch_size=1, shuffle=True)
    # un-Normalize the image
    fakes = unorm_glasses(fake)
    fakes = fakes[:,:,39:81,21:138]
    for k in range(3):
        fakes[0,k,:,:][orig_mask_g == 0] = 0
    faces, landmarks, labels = next(iter(dataloader_me_test))
    # affix glasses to face using the bounding box values from the dlib output
    for j in range(faces.size(0)):
        img = unorm(faces[j,:,:,:])
        glassHeight,glassWidth = landmarks[j,-2:].int()
        x1,x2,y1,y2 = landmarks[j,:-2].int()
        glass = F.interpolate(fakes, (glassHeight,glassWidth))
        mask = F.interpolate(T(orig_mask_g[None,None,:,:]), (glassHeight,glassWidth))
        mask_inv = F.interpolate(T(orig_mask_inv_g[None,None,:,:]), (glassHeight,glassWidth))
        roi1 = img[None,:,y1:y2, x1:x2]
        roi_bg = roi1 - mask
        roi_bg = torch.clamp(roi_bg, 0)
        roi_fg = glass + mask_inv
        print(glass.shape, roi_bg[0].shape, img[:,y1:y2, x1:x2].shape)
        img[:,y1:y2, x1:x2] = glass[j] + roi_bg[0]
        faces[j,:,:,:] = img
    # show image of your face with glasses attached
    plt.imshow(np.transpose(utils.make_grid(faces.cpu(), padding=2, normalize=True),(1,2,0)))
    # check against the classifier
    model_ft.eval()
    faces = faces.to(device)
    outputs = model_ft(faces)
    _, preds = torch.max(outputs, 1)
    print([class_names[p] for p in preds])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--generator',
        type=str,
        help='Path to generator file')
    
    run_test_adv_G()