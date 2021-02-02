from imports import *
from pretrain_gans import *
from vid_processing import *
from dataset import *
from train import *
from archs import *
from finetune_face_classifier import *
from models.inception_resnet_v1 import InceptionResnetV1

def run_AGN(last_layers:str='n', ngpu:int=0, testing:str='n', write_vid:str='y'):
    """
    Function to run full AGN training from scratch.
        1. Train GAN to produce realistic eyeglass frames
        2. Finetune facial recognition classifier on images of my face
        3. Adversarially train generator to produce eyeglasses that trick the classifier
    """
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    # Initialize dataset of eyeglasses
    tensor_dataset = EyeglassesDataset(
        csv_file='data/files_sample.csv',
        root_dir='data/eyeglasses/',
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(160),
            transforms.ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]))
    # Initialize D & G
    bs = 64
    nc, ndf, ngf, nz = 3, 160, 160, 100
    netD = Discriminator(ngpu, nc, ndf, ngf, nz).to(device)
    netG = Generator(ngpu, nc, ndf, ngf, nz).to(device)
    # Apply weights
    netD.apply(weights_init)
    netG.apply(weights_init)
    # Initialize BCELoss function - BCEWithLogitsLoss is used due to removal of Sigmoid in D
    criterion = nn.BCEWithLogitsLoss() 
    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    # Build dataloader of eyeglass frames
    eye_dl = DataLoader(tensor_dataset, batch_size=64, shuffle=True)
    # Pre-train the GAN on the eyeglasses
    img_list, G_losses, D_losses, netG, netD = pretrain_gan(netD, netG, eye_dl, criterion, fixed_noise, nz, testing=testing)
    # Data augmentation and normalization for training - Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    # Specify path for location of images of my face
    data_dir = 'data/test_me'
    # Build datasets and loaders for both train and validation sets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8, shuffle=True) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    # Get pretrained resnet on vggface2 dataset
    model_ft = InceptionResnetV1(pretrained='vggface2', classify=False, num_classes=len(class_names))
    # Remove the last layers after conv block
    layer_list = list(model_ft.children())[-5:] # all final layers
    # All beginning layers
    model_ft = nn.Sequential(*list(model_ft.children())[:-5])
    if last_layers == 'y':
        for param in model_ft.parameters():
            param.requires_grad = False
    # Re-initialize layers to set requires_grad to True
    model_ft.avgpool_1a = nn.AdaptiveAvgPool2d(output_size=1)
    model_ft.last_linear = nn.Sequential(
        Flatten(),
        nn.Linear(in_features=1792, out_features=512, bias=False),
        normalize()
    )
    model_ft.logits = nn.Linear(layer_list[3].in_features, len(class_names))
    model_ft.softmax = nn.Softmax(dim=1)
    # Place model on device
    model_ft = model_ft.to(device)
    # Define loss function for finetuning
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=1e-2, momentum=0.9)
    # Decay LR by a factor of *gamma* every *step_size* epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    # Finetune the facial recognition classifier
    model_ft, FT_losses = train_ft_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=1)
    # Save the model
    torch.save({'state_dict': model_ft.state_dict()}, f"tmp/model_ft_loss{FT_losses[-1]}_{dt.datetime.today().strftime('%Y%m%d')}.pth.tar")

    ### AGN training ###

    # If you have not already saved the frames of the video of you can do that here
    ### Perform data pre-processing step on frames from video described in readme using command line after this step ###
    if write_vid == 'y':
        vidcap = cv2.VideoCapture('data/IMG_2411.MOV')
        success,image = vidcap.read()
        count = 0
        success = True
        while success:
            cv2.imwrite(f"data/agn_me_extras160/Michael_Chaykowsky/Michael_Chaykowsky_{format(count, '04d')}.png", image)
            success,image = vidcap.read()
            print('Read a new frame: ', success)
            count += 1

    # Transforms for the images and the coordinates (where eyeglasses go) seperately
    t_img = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        # transforms.RandomHorizontalFlip(), # couldn't get to work
        # transforms.RandomRotation(5), # couldn't get to work
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    t_landmarks = transforms.Compose([
        transforms.ToTensor()
    ])
    # Custom dataset for images of my face to affix glasses
    image_datasets_me = MeDataset(
        'data/bboxes_fnames.csv', 
        'data/agn_me_extras160/Michael_Chaykowsky', 
        bs=1, 
        transform_img=t_img, 
        transform_land=t_landmarks)
    # Build dataloader for face images
    dataloader_me = torch.utils.data.DataLoader(image_datasets_me, batch_size=64, shuffle=True)
    # Pre-process video data of my face to get images to adversarially train the generator
    coords = vid_preprocess()
    # Full training of AGN
    netG, img_list, G_losses, D_losses, d1s, d3s, num_fooled = train_AGN(
        netG, netD, model_ft, eye_dl, 
        dataloader_me, class_names, nz, num_epochs=1)
    # Save adversarial generator for testing later
    torch.save({'state_dict': netG.state_dict()}, f"tmp/adv_G_{dt.datetime.today().strftime('%Y%m%d')}.pth.tar")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--last_layers',
        type=str,
        default='n',
        help='Path to model file')
    parser.add_argument(
        '--ngpu',
        type=int,
        default=0,
        help='How many GPUs do you have?')
    parser.add_argument(
        '--write_vid',
        type=str,
        default='y',
        help='Do you want to save IMG_2411.MOV file frames as png?')
    parser.add_argument(
        '--testing',
        type=str,
        default='n',
        help='Run all epochs?')
    args = parser.parse_args()

    run_AGN(args.last_layers, args.ngpu, args.testing, args.write_vid)
