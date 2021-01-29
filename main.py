from imports import *
from pretrain_gans import *
from models.inception_resnet_v1 import InceptionResnetV1

def run_AGN(last_layers:bool=False):
    """"""
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    tensor_dataset = EyeglassesDataset(
        csv_file='data/files_sample.csv',
        root_dir='data/eyeglasses/',
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(160),
            transforms.ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]))
    
    dataloader_test = DataLoader(tensor_dataset, batch_size=4, shuffle=True, num_workers=2)

    bs,nz = 64,100
    nc = 3; ndf = 160; ngf = 160
    netD = Discriminator(ngpu).to(device)
    netG = Generator(ngpu).to(device)

    netD.apply(weights_init)
    netG.apply(weights_init)

    # Initialize BCELoss function
    criterion = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss is used due to removal of Sigmoid in D

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0
    beta1 = 0.5

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=1e-5, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-5, betas=(beta1, 0.999))

    eye_dl = DataLoader(tensor_dataset, batch_size=64, shuffle=True)

    img_list, G_losses, D_losses, netG, netD = pretrain_gan(dataloader=eye_dl)

    # Load Data
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

    data_dir = 'data/test_me'

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8, shuffle=True) for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    # Get pretrained resnet on vggface2 dataset
    model_ft = InceptionResnetV1(pretrained='vggface2', classify=False, num_classes=len(class_names))
    # Remove the last layers after conv block
    layer_list = list(model_ft.children())[-5:] # all final layers
    # all beginning layers
    model_ft = nn.Sequential(*list(model_ft.children())[:-5])
    if last_layers:
        for param in model_ft.parameters():
            param.requires_grad = False
    # Ree-initialize layers to set requires_grad to True
    model_ft.avgpool_1a = nn.AdaptiveAvgPool2d(output_size=1)
    model_ft.last_linear = nn.Sequential(
        Flatten(),
        nn.Linear(in_features=1792, out_features=512, bias=False),
        normalize()
    )
    model_ft.logits = nn.Linear(layer_list[3].in_features, len(class_names))
    model_ft.softmax = nn.Softmax(dim=1)

    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=1e-2, momentum=0.9)

    # Decay LR by a factor of *gamma* every *step_size* epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft, FT_losses = train_ft_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=500)
    torch.save({'state_dict': model_ft.state_dict()}, f'model_ft_loss{FT_losses[-1]}_{dt.datetime.today().strftime('%Y%m%d')}.pth.tar')