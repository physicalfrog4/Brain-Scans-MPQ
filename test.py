class YoloModel(YOLO):
     def __init__(self, *args, **kwargs):
         super(YoloModel, self).__init__(*args, **kwargs)
     def __str__(self):
         return ""
     def __repr__(self):
         return ""

 #Currently Using
 class roiVGGYolo(torch.nn.Module):
     def __init__(self, numROIs: int, tsfms):
         super(roiVGGYolo, self).__init__()
         #Make VGG Instance for feature extraction
         self.vgg = vgg19(weights = "DEFAULT")
         #Get layers to use for feature extraction and freeze the model params so they aren't updated during training
         self.vggConvFeatures = self.vgg.features[:35]
         for params in self.vgg.parameters():
             params.requires_grad = False
         #Make yolo instance and freeze parameters so they aren't updated during training
         self.yolo = YoloModel("yolov8n.pt")
         for params in self.yolo.parameters():
             params.requires_grad = False
         #Create the MLP which is just a series of linear layers with relu
         self.MLP = torch.nn.Sequential(
             torch.nn.Linear(25088, 4096),
             torch.nn.ReLU(),
             torch.nn.Linear(4096, 1024),
             torch.nn.ReLU(),
             torch.nn.Linear(1024, numROIs),
         )
         #Save the torch transforms for the YOLO model
         self.tsfms = tsfms
     def forward(self, img, imgPaths):
         #extract vgg features from image
         convFeatures = self.vggConvFeatures(img)
         #transform the original images such as it's compatible with the YOLO model
         yoloInput = [self.tsfms(Image.open(image)) for image in imgPaths]
         #Make YOLO predictions on images and get bounding box data
         yoloResults = [results.boxes for results in self.yolo.predict(torch.stack(yoloInput), verbose=False)]
         boundingBoxDataAllImages = self.getMappedBoundingBox(yoloResults)
         finalFMRIs = []
         indices = [] #saved indexes to use that have bounding box info from yolo
         count = 0
         #for each detected bounding box
         for boundingBoxData in boundingBoxDataAllImages:
             #if no bounding boxes, continue (ignore image)
             if len(boundingBoxData) == 0:
                 continue
             #pool the features in the regions that overlap with a bounding box. returns a result for each detected object
             objectROIPools = ops.roi_pool(convFeatures, [boundingBoxData], output_size = (7,7)) #output shape (num objects, 512, 7, 7)
             fmriPieces = []
             #for the pooled results for each object, predict the partial fmri data
             for objectROIPool in objectROIPools:
                 input = torch.flatten(objectROIPool) #flatten data to pass into mlp, shape (1, 25088)
                 fmriPieces.append(self.MLP(input)) #save partial fmri prediction
             #sum over all partial fmri data
             totalFMRI = torch.sum(torch.stack(fmriPieces), dim=0)
             finalFMRIs.append(totalFMRI)
             indices.append(count)
             count+=1
         return torch.stack(finalFMRIs), indices
     #extract bounding box data for each of the yolo results objects
     def getMappedBoundingBox(self, yoloResults):
         mappedBoxes = []
         for result in yoloResults:
             #get normalized top left and bottom right coordinates for the bounding box
             boundingBoxData = result.xyxyn
             #Get interested cells in 7x7 matrix for ROI pooling                      topLeftX               topLeftY               bottomRightX           #bottomRightY
             boundingBoxStartX, boundingBoxStartY, boundingBoxEndX, boundingBoxEndY = boundingBoxData[:, 0], boundingBoxData[:, 1], boundingBoxData[:, 2], boundingBoxData[:, 3]
             #Transform normalized range (0 to 1) to (0 to 7)
             transformedBoundingBoxStartX, transformedBoundingBoxStartY, transformedBoundingBoxEndX, transformedBoundingBoxEndY = boundingBoxStartX * 7, boundingBoxStartY * 7, boundingBoxEndX * 7, boundingBoxEndY * 7
             startCellX = torch.floor(transformedBoundingBoxStartX)
             startCellY = torch.floor(transformedBoundingBoxStartY)
             endCellX = torch.ceil(transformedBoundingBoxEndX)
             endCellY = torch.ceil(transformedBoundingBoxEndY)
             mappedBoxes.append(torch.hstack((startCellX.reshape(-1,1), startCellY.reshape(-1,1), endCellX.reshape(-1,1), endCellY.reshape(-1,1))))
         return mappedBoxes
     #Overwrite functions so that the model isn't printed on server side after each call to train() or eval()
     def __str__(self):
         return ""
     def __repr__(self):
         return ""

 platform = 'jupyter_notebook'
 device = 'cuda:1'
 device = torch.device(device)
 # setting up the directories and ARGS
 data_dir = './algonauts_2023_challenge_data/'#../MQP/algonauts_2023_challenge_data/'
 parent_submission_dir = '../submission'
 subj = 1 # @param ["1", "2", "3", "4", "5", "6", "7", "8"] {type:"raw", allow-input: true}
 # args

  args = argObj(data_dir, parent_submission_dir, subj)
 words = ['furniture', 'food', 'kitchenware', 'appliance', 'person', 'animal', 'vehicle', 'accessory',
             'electronics', 'sports', 'traffic', 'outdoor', 'home', 'clothing', 'hygiene', 'toy', 'plumbing',
              'computer', 'fruit', 'vegetable', 'tool']

 #define transforms for the images to be passed into the vgg model
 tsfms = transforms.Compose([
     transforms.Resize((224,224)),
     transforms.ToTensor(),
 ])

 #define transforms for the images to be passed into the YOLO model
 yoloTsfms = transforms.Compose([
     transforms.Resize((640, 640)),
     transforms.ToTensor()
 ])

 trainingDataset = AlgonautsDataset(args.data_dir, args.subj, transform=tsfms)
-2.515819 2.4762304
-2.5166664 2.4293716
 trainIdxs, validIdxs = train_test_split(range(len(trainingDataset.imagePaths)), train_size=0.9, random_state = 42)
 lhNumROIs = len(trainingDataset.lhFMRI[0])
 rhNumROIs = len(trainingDataset.rhFMRI[0])
 trainSubset = Subset(trainingDataset, trainIdxs)
 trainDataLoader = DataLoader(trainSubset, batch_size = 128, shuffle = True)
 validSubset = Subset(trainingDataset, validIdxs)
 validDataLoader = DataLoader(validSubset, batch_size = 128, shuffle = True)#change
 #Create VGG and YOLO model
 lhModel = roiVGGYolo(lhNumROIs, yoloTsfms).to(device)
 # rhModel = roiVGGYolo(rhNumROIs, yoloTsfms).to(device)
 #Define optimzer params and loss function
 learningRate = 0.0001
 lhOptim = torch.optim.Adam(lhModel.parameters(), learningRate)#,  weight_decay=1e-4
 lhScheduler = lr_scheduler.StepLR(lhOptim, step_size=10, gamma=0.1)
 # rhOptim = torch.optim.Adam(rhModel.parameters(), learningRate)#,  weight_decay=1e-4
 # rhScheduler = lr_scheduler.StepLR(rhOptim, step_size=10, gamma=0.1)
 criterion = torch.nn.MSELoss()
 #Training for specified epochs
 epochs = 15

P # lhPredictions = []
 # allLhIndices = np.array([])
 # rhPredictions = []
 # allRhIndices = np.array([])
 # first = True
 # with torch.no_grad():
 #     for data in tqdm(validDataLoader, desc="Evaluating", unit="batch"):
 #         #Get data in batch
 #         img, imgPaths, _, _, _, _ = data
 #         img = img.to(device)
 #         # normalLhFMRI = normalLhFMRI.to(device)
 #         # normalRhFMRI = normalRhFMRI.to(device)
 #         #make predictions
 #         lhPred, lhIndices = lhModel(img, imgPaths)
 #         rhPred, rhIndices = rhModel(img, imgPaths)
 #         print(lhIndices)
 #         print()
 #         #
 #         if first:
 #             first = False
 #             lhPredictions.extend(lhPred.cpu().numpy())
 #             allLhIndices = np.concatenate((allLhIndices, np.array(lhIndices)))
 #             # allLhIndices.extend(lhIndices )
 #             rhPredictions.extend(rhPred.cpu().numpy())
 #             allRhIndices = np.concatenate((allRhIndices, np.array(rhIndices)))
 #             # allRhIndices.extend(rhIndices)
 #         else:
 #             lhPredictions.extend(lhPred.cpu().numpy())
 #             allLhIndices = np.concatenate((allLhIndices, np.array(lhIndices) + allLhIndices[-1] + 1))
 #             # allLhIndices.extend(lhIndices )
 #             rhPredictions.extend(rhPred.cpu().numpy())
 #             allRhIndices = np.concatenate((allRhIndices, np.array(rhIndices) + allRhIndices[-1] + 1))
 #             # allRhIndices.extend(rhIndices)




 # lhPredictions = np.array(lhPredictions)
 # rhPredictions = np.array(rhPredictions)






 for epoch in range(epochs):
     #Variables to hold information on training progress
     print(f"Epoch {epoch}")
     avgLhTrainingLoss = 0
     avgLhEvalLoss = 0
     avgLhTrainingR2Score = 0
     avgLhEvalR2Score = 0
     # avgRhTrainingLoss = 0
     # avgRhEvalLoss = 0
     # avgRhTrainingR2Score = 0
     # avgRhEvalR2Score = 0
     for data in tqdm(trainDataLoader, desc="Training", unit="batch"):
         #Get data in batch
         img, imgPaths, _, _, normalLhFMRI, normalRhFMRI = data
         img = img.to(device)
         normalLhFMRI = normalLhFMRI.to(device)
         # normalRhFMRI = normalRhFMRI.to(device)
         lhOptim.zero_grad()
         # rhOptim.zero_grad()
         #make predictions
         lhPred, lhIndices = lhModel(img, imgPaths)
         # rhPred, rhIndices = rhModel(img, imgPaths)
         #only use data for images that had bounding box data
         normalLhFMRI = normalLhFMRI[lhIndices]
         # normalRhFMRI = normalRhFMRI[rhIndices]
         #evaluate based on loss function
         lhLoss = criterion(lhPred, normalLhFMRI)
         lhLoss.backward()
         lhOptim.step()
         # rhLoss = criterion(rhPred, normalRhFMRI)
         # rhLoss.backward()
         # rhOptim.step()
         #sum r2 scores and loss values for batch
         avgLhTrainingLoss += lhLoss.item()
         avgLhTrainingR2Score += r2_score(lhPred.detach().cpu().numpy(), normalLhFMRI.detach().cpu().numpy())
         # avgRhTrainingLoss += rhLoss.item()
         # avgRhTrainingR2Score += r2_score(rhPred.detach().cpu().numpy(), normalRhFMRI.detach().cpu().numpy())
     with torch.no_grad():
         for data in tqdm(validDataLoader, desc="Evaluating", unit="batch"):
             #Get data in batch
             img, imgPaths, _, _, normalLhFMRI, normalRhFMRI = data
             img = img.to(device)
             normalLhFMRI = normalLhFMRI.to(device)
             # normalRhFMRI = normalRhFMRI.to(device)
             #make predictions
             lhPred, lhIndices = lhModel(img, imgPaths)
             # rhPred, rhIndices = rhModel(img, imgPaths)
             #only use data for images that had bounding box data
             normalLhFMRI = normalLhFMRI[lhIndices]
             # normalRhFMRI = normalRhFMRI[rhIndices]
             #evaluate based on loss function
             lhEvalLoss = criterion(lhPred, normalLhFMRI)
             # rhEvalLoss = criterion(rhPred, normalRhFMRI)
             #sum r2 scores and loss values for batch
             avgLhEvalLoss += lhEvalLoss.item()
             avgLhEvalR2Score += r2_score(lhPred.detach().cpu().numpy(), normalLhFMRI.detach().cpu().numpy())
             # avgRhEvalLoss += rhEvalLoss.item()
             # avgRhEvalR2Score += r2_score(rhPred.detach().cpu().numpy(), normalRhFMRI.detach().cpu().numpy())
     lhScheduler.step()
     # rhScheduler.step()
     #calculate metrics for epoch
     validLhMse = avgLhEvalLoss / len(validDataLoader)
     validLhR2 = avgLhEvalR2Score / len(validDataLoader)
     # validRhMse = avgRhEvalLoss / len(validDataLoader)
     # validRhR2 = avgRhEvalR2Score / len(validDataLoader)
     print(f"lh using lr = {lhScheduler.get_last_lr()} TrainingMSE: {avgLhTrainingLoss / len(trainDataLoader)}, ValidMSE: {validLhMse}, trainR2 = {avgLhTrainingR2Score / len(trainDataLoader)}, evalR2= {validLhR2}")
     # print(f"rh using lr = {rhScheduler.get_last_lr()} TrainingMSE: {avgRhTrainingLoss / len(trainDataLoader)}, ValidMSE: {validRhMse}, trainR2 = {avgRhTrainingR2Score / len(trainDataLoader)}, evalR2= {validRhR2}")


     Epoch 0
Training: 100%|██████████████████████████████████████████████████████████████████████| 70/70 [07:06<00:00,  6.09s/batch]
Evaluating: 100%|██████████████████████████████████████████████████████████████████████| 8/8 [00:34<00:00,  4.34s/batch]
lh using lr = [0.0001] TrainingMSE: 0.6171573162078857, ValidMSE: 0.2749865688383579, trainR2 = -126403110.32687958, evalR2= -5437840.159811192
Epoch 1
Training: 100%|██████████████████████████████████████████████████████████████████████| 70/70 [06:56<00:00,  5.95s/batch]
Evaluating: 100%|██████████████████████████████████████████████████████████████████████| 8/8 [00:33<00:00,  4.20s/batch]
lh using lr = [0.0001] TrainingMSE: 0.26512725949287413, ValidMSE: 0.25418172404170036, trainR2 = -4581003.274572415, evalR2= -180251.88091374066
Epoch 2
Training: 100%|██████████████████████████████████████████████████████████████████████| 70/70 [06:57<00:00,  5.97s/batch]
Evaluating: 100%|██████████████████████████████████████████████████████████████████████| 8/8 [00:33<00:00,  4.19s/batch]
lh using lr = [0.0001] TrainingMSE: 0.24545944971697672, ValidMSE: 0.23461440578103065, trainR2 = -23302823.87792561, evalR2= -3812728.6043471782
Epoch 3
Training: 100%|██████████████████████████████████████████████████████████████████████| 70/70 [06:56<00:00,  5.95s/batch]
Evaluating: 100%|██████████████████████████████████████████████████████████████████████| 8/8 [00:33<00:00,  4.20s/batch]
lh using lr = [0.0001] TrainingMSE: 0.22728079919304167, ValidMSE: 0.21756133250892162, trainR2 = -61283562.208520986, evalR2= -9855359.272617662
Epoch 4
Training: 100%|██████████████████████████████████████████████████████████████████████| 70/70 [06:57<00:00,  5.97s/batch]
Evaluating: 100%|██████████████████████████████████████████████████████████████████████| 8/8 [00:33<00:00,  4.25s/batch]
lh using lr = [0.0001] TrainingMSE: 0.2113634486283575, ValidMSE: 0.20264874398708344, trainR2 = -53103639.773944594, evalR2= -1430904.4871995915
Epoch 5
Training: 100%|██████████████████████████████████████████████████████████████████████| 70/70 [06:57<00:00,  5.97s/batch]
Evaluating: 100%|██████████████████████████████████████████████████████████████████████| 8/8 [00:33<00:00,  4.17s/batch]
lh using lr = [0.0001] TrainingMSE: 0.19826264253684453, ValidMSE: 0.19074797630310059, trainR2 = -138150998.41316468, evalR2= -238743.869432803
Epoch 6
Training: 100%|██████████████████████████████████████████████████████████████████████| 70/70 [06:58<00:00,  5.97s/batch]
Evaluating: 100%|██████████████████████████████████████████████████████████████████████| 8/8 [00:33<00:00,  4.18s/batch]
lh using lr = [0.0001] TrainingMSE: 0.18715660401753018, ValidMSE: 0.18106965348124504, trainR2 = -17533127.81373673, evalR2= -8223.870479353278
Epoch 7
Training: 100%|██████████████████████████████████████████████████████████████████████| 70/70 [06:56<00:00,  5.95s/batch]
Evaluating: 100%|██████████████████████████████████████████████████████████████████████| 8/8 [00:33<00:00,  4.14s/batch]
lh using lr = [0.0001] TrainingMSE: 0.177986889226096, ValidMSE: 0.17316096276044846, trainR2 = -451987.12742406357, evalR2= -3704.9436959300356
Epoch 8
Training: 100%|██████████████████████████████████████████████████████████████████████| 70/70 [06:51<00:00,  5.88s/batch]
Evaluating: 100%|██████████████████████████████████████████████████████████████████████| 8/8 [00:29<00:00,  3.74s/batch]
lh using lr = [0.0001] TrainingMSE: 0.17065899627549308, ValidMSE: 0.16632726602256298, trainR2 = -34135.19261315329, evalR2= -14.006261324576855
Epoch 9
Training: 100%|██████████████████████████████████████████████████████████████████████| 70/70 [06:48<00:00,  5.83s/batch]
Evaluating: 100%|██████████████████████████████████████████████████████████████████████| 8/8 [00:29<00:00,  3.69s/batch]
lh using lr = [1e-05] TrainingMSE: 0.1647493072918483, ValidMSE: 0.16258666664361954, trainR2 = -10.846779972723416, evalR2= -6.870876646716905
Epoch 10
Training: 100%|██████████████████████████████████████████████████████████████████████| 70/70 [06:49<00:00,  5.85s/batch]
Evaluating: 100%|██████████████████████████████████████████████████████████████████████| 8/8 [00:30<00:00,  3.83s/batch]
lh using lr = [1e-05] TrainingMSE: 0.1620670314346041, ValidMSE: 0.1621967926621437, trainR2 = -7.823157078846005, evalR2= -6.384219885329811
Epoch 11
Training: 100%|██████████████████████████████████████████████████████████████████████| 70/70 [06:56<00:00,  5.95s/batch]
Evaluating: 100%|██████████████████████████████████████████████████████████████████████| 8/8 [00:32<00:00,  4.12s/batch]
lh using lr = [1e-05] TrainingMSE: 0.16153633445501328, ValidMSE: 0.16171522624790668, trainR2 = -7.525114675371148, evalR2= -6.143365612400881
Epoch 12
Training: 100%|██████████████████████████████████████████████████████████████████████| 70/70 [06:58<00:00,  5.98s/batch]
Evaluating: 100%|██████████████████████████████████████████████████████████████████████| 8/8 [00:32<00:00,  4.11s/batch]
lh using lr = [1e-05] TrainingMSE: 0.16110469635043825, ValidMSE: 0.16054522059857845, trainR2 = -7.159005663444848, evalR2= -5.756406569720187
Epoch 13
Training: 100%|██████████████████████████████████████████████████████████████████████| 70/70 [06:58<00:00,  5.98s/batch]
Evaluating: 100%|██████████████████████████████████████████████████████████████████████| 8/8 [00:33<00:00,  4.15s/batch]
lh using lr = [1e-05] TrainingMSE: 0.16209917281355177, ValidMSE: 0.1599310953170061, trainR2 = -7.06038276992769, evalR2= -5.829012697353271
Epoch 14
Training: 100%|██████████████████████████████████████████████████████████████████████| 70/70 [07:01<00:00,  6.02s/batch]
Evaluating: 100%|██████████████████████████████████████████████████████████████████████| 8/8 [00:33<00:00,  4.20s/batch]
lh using lr = [1e-05] TrainingMSE: 0.16054844643388475, ValidMSE: 0.16066600196063519, trainR2 = -6.842499213943924, evalR2= -5.314793296516097