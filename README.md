1. Noting
    We recommend using python3.
    Before running, unzip the folder "UMFiber\DataProcessing\DataSource.zip".
    If you want to run "UMFiber\DataProcessing\.py", please first run "UMFiber\DataProcessing\logitistic.py" to generate the dataset.

2. Data processing
    The optical microfiber dynamic response data is stored in power_intensity folder, and the data dimension of each sample is 4*2000. 
    4 represents 4 fingers, and 2000 represents the collection time is 2000ms. Each gesture should be done within 2 seconds.     
    If you want to change the normalization way, adjust in Lines 27 - 30, "trans_pic.py".
    If you want to adjust the proportion of the test set, adjust in Line 30, "generate_dataset.py".
    The pictures in OMDRdatasets is the dataset of our CNNs model.

3. Training the networks
    If you want to train the VGGNet, simply run "VGGNet.py".
        If you want to change the optimizer, adjust in Line 157.
        If you want to change the image preprocessing method, adjust in Lines 90 - 95.
    If you want to train the shallow CNNS model, simply run "shallow_cnn.py".
        If you want to change the network architecture of CNNs, adjust in Lines 31 - 36, and 38 - 39.

4. Results
    The training results of "VGGNet.py" is as follow:
        The model has 134309962 trainable parameters
        Number of training examples:810
        Number of validation examples:90
        Number of testing examples:100
        Epoch: 1
                Train Loss: 2.8920043890292826|Train Acc: 0.3072916681949909
                 Val. Loss: 1.2294911742210388|Val. Acc:0.4813701957464218
        Epoch: 2
                Train Loss: 0.4998496977182535|Train Acc: 0.841804027557373
                 Val. Loss: 0.6671781241893768|Val. Acc:0.7986778914928436
        Epoch: 3
                Train Loss: 0.3646262717934755|Train Acc: 0.9031021044804499
                 Val. Loss: 0.3472229093313217|Val. Acc:0.8882211446762085
        Epoch: 4
                Train Loss: 0.16731591981190902|Train Acc: 0.9524954190621009
                 Val. Loss: 0.17869116738438606|Val. Acc:0.9302884638309479
        Epoch: 5
                Train Loss: 0.12536185062848604|Train Acc: 0.9614812273245591
                 Val. Loss: 0.1926010763272643|Val. Acc:0.9615384638309479
        Epoch: 6
                Train Loss: 0.06017235949492225|Train Acc: 0.9807692307692307
                 Val. Loss: 0.13057442707940936|Val. Acc:0.9807692170143127
        Epoch: 7
                Train Loss: 0.044278962251085505|Train Acc: 0.9897550344467163
                 Val. Loss: 0.03910090634599328|Val. Acc:0.9807692170143127
        Epoch: 8
                Train Loss: 0.023835336574568197|Train Acc: 0.9897550344467163
                 Val. Loss: 0.2017540684901178|Val. Acc:0.9807692170143127
        Epoch: 9
                Train Loss: 0.04152647156017618|Train Acc: 0.9819711538461539
                 Val. Loss: 0.12359786487650126|Val. Acc:0.9807692170143127
        Epoch: 10
                Train Loss: 0.03505192253102835|Train Acc: 0.9867216119399438
                 Val. Loss: 0.07676350232213736|Val. Acc:0.9729567170143127
        Epoch Time: 3m 57s
        epoch_mins:0,epoch_secs:9
        Test Loss:0.05436124140396714|Test Acc:0.9921875
