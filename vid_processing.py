from imports import *

def vid_preprocess():
    # Capture the frames of the video as `.png` files and rotate/crop/align
    vidcap = cv2.VideoCapture('IMG_2411.MOV')
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
        cv2.imwrite(f"./Michael_Chaykowsky/Michael_Chaykowsky_{format(count, '04d')}.png", image)
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

    # dlib face alignment (example notebook available in repo)
    predictor_path = "data/shape_predictor_68_face_landmarks.dat"
    face_rec_model_path = "data/dlib_face_recognition_resnet_model_v1.dat"

    cnn_face_detector = dlib.cnn_face_detection_model_v1("data/mmod_human_face_detector.dat")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # Capture the specific areas of the face around the eyes and save to list
    coords = []
    for frame, _, labels in dataloader_me:
        print(frame[0].shape)
        frame = np.transpose(to_np(frame[0]), (1,2,0)) * 255
        frame = frame.astype('uint8')
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(frame.shape)
        dets = cnn_face_detector(frame, 1)
        d = dets[0]
        shape = predictor(frame, d.rect)

        glassWidth = abs(shape.part(16).x - shape.part(1).x)
        glassHeight = int(glassWidth * origGlassHeight / origGlassWidth)

        y1 = int(shape.part(24).y)
        y2 = int(y1 + glassHeight)
        x1 = int(shape.part(27).x - (glassWidth/2))
        x2 = int(x1 + glassWidth)

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

