from imports import *


class MeDataset(Dataset):
    """Eyeglasses dataset."""

    def __init__(self, csv_file, root_dir, bs, transform_img=None, transform_land=None):
        """
        Args:
            csv_file (string): Path to the csv file with labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.label = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform_img = transform_img
        self.transform_land = transform_land
        self.bs = bs

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.label.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.label.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks, 'label': np.zeros(self.bs).astype('float').reshape(-1, 1)}
        # sample = image

        if self.transform_img:
            sample['image'] = self.transform_img(sample['image'])
            
        if self.transform_land:
            sample['landmarks'] = self.transform_land(sample['landmarks'])
            sample['label'] = self.transform_land(sample['label'])
        
        #sample['label'] = sample['label'].reshape(-1).view(-1)

        return sample['image'], sample['landmarks'].reshape(-1), sample['label'].reshape(-1)


def vid_preprocess():
    """
    Takes a video captured from an iPhone and pre-processes it to:
        1. Obtain the frames of the video as `.png` files and rotate/crop/align
        2. Perform facial landmark analysis to determine position of eyes in image
        3. Finds and returns specific coordinates for each image of where to place eyeglass frames
    """
    t_img = transforms.Compose([transforms.ToTensor()])
    t_land = transforms.Compose([transforms.ToTensor()])
    image_datasets = MeDataset('data/bboxes_fnames.csv', 'data/agn_me_extras160/Michael_Chaykowsky', 
                            bs = 1, transform_img=t_img, transform_land=t_land)
    dataloader_me = torch.utils.data.DataLoader(image_datasets, batch_size=1, shuffle=False)
    # dlib face alignment (example notebook available in repo)
    predictor_path = "./data/shape_predictor_68_face_landmarks.dat"
    face_rec_model_path = "./data/dlib_face_recognition_resnet_model_v1.dat"
    # Load pre-trained facial landmark analysis models
    cnn_face_detector = dlib.cnn_face_detection_model_v1("./data/mmod_human_face_detector.dat")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    # Get original glass height and width for calculations
    imgGlass = cv2.imread("data/eyeglasses/glasses000002-2.png", -1)
    r = 160.0 / imgGlass.shape[1]
    dim = (160, int(imgGlass.shape[0] * r))
    # perform the actual resizing of the image and show it
    imgGlass = cv2.resize(imgGlass, dim, interpolation = cv2.INTER_AREA)
    imgGlass = imgGlass[39:81, 21:138]
    origGlassHeight, origGlassWidth = imgGlass.shape[:2]
    # # Make orig_mask_inv_g
    # alpha_data = imgGlass[:,:,2]
    # alpha_data[alpha_data < 255] = 1
    # orig_mask_g = alpha_data
    # orig_mask_inv_g = cv2.bitwise_not(orig_mask_g)
    # Capture the specific areas of the face around the eyes and save to list
    coords = []
    for iter, (frame, _, labels) in enumerate(dataloader_me):
        if iter % 20 == 0: print('Iteration: ', iter)
        frame = np.transpose(to_np(frame[0]), (1,2,0)) * 255
        frame = frame.astype('uint8')
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = cnn_face_detector(frame, 1)
        d = dets[0]
        shape = predictor(frame, d.rect)
        # Determine the width and height of the eyeglass frames
        glassWidth = abs(shape.part(16).x - shape.part(1).x)
        glassHeight = int(glassWidth * origGlassHeight / origGlassWidth)
        # Determine the bounding box coords for the given image/frame
        y1 = int(shape.part(24).y)
        y2 = int(y1 + glassHeight)
        x1 = int(shape.part(27).x - (glassWidth/2))
        x2 = int(x1 + glassWidth)
        # If the bounding box for the eyes falls outside of the image resize it to make it fit
        if y1 < 0: 
            glassHeight = glassHeight - abs(y1)
            y1 = 0
        if y2 > frame.shape[0]: glassHeight = glassHeight - (y2 - frame.shape[0])
        if x1 < 0: 
            glassWidth = glassWidth - abs(x1)
            x1 = 0
        if x2 > frame.shape[1]: glassWidth = glassWidth - (x2 - frame.shape[1])
        
        coords.append([x1, x2, y1, y2, glassHeight, glassWidth])
    return coords
