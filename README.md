# MA_Detection_Recocnition_Endoscopic_Videos
Repository for the master thesis with the title: "Detection and recognition in neurosurgical endoscopic videos" done by Julia Zahner, March - September 2021, CVL ETH Zurich, Supervisor: Prof. Dr. Ender Konukoglu

The Code is seperated in three parts:
    - 'net' directory with all the files defining the neural net structure and losses
    - 'preproc' directory with all the files used for data and preprocessing
    - runable files used for training, prediction and visualization + configuration file

Other directories are:
    - 'pred_lab', holding all the saved checkpoints, predictions, heatmaps, visualizations (visualization + gif_overlays)


Overview of files:
    - ./net/modules.py : definition of the CNN structure (torch.nn.Module)
    - ./net/loss_fn.py : definition of loss functions used in the code
    - ./preproc/load_data.py : definition & generation of datasets from data
    - ./preproc/distortion.py : definitions of non-torchvision transformation for image preprocessing
    - ./preproc/create_csv_annot.py : script to generate csv-files with path and label information (saved in 'preproc/data_lists') 

    ./config.py : configuration file for the code
    ./train_baseline.py : execution file for baseline 
    ./train.py : execution file for pretraining
    ./train_whole.py : execution file for main training
    ./predict.py : execution file for prediction seperate from training
    ./visualize.py : execution file to visualize results and generate gifs

Weights and Biases is used for experiment tracking in the code. Create an account under https://wandb.ai/site to use it.

To run a full training with pretraining follow the following procedure:
    1. Adjust save paths in the config.py file to your needs
    2. Set cycle = 0 in the config.py file (start of pretraining)
    3. Set login key for wandb in the train.py file (you can find it in your account info from wandb)
    4. Run train.py on a computer (GPU needed) or cluster
    5. Pretraining is cut into train cycles of four epochs, increase cycle in the config.py file after ending a cycle and repeat step 4. 
    6. When satisfied with the pretraining performance main training can be started.
    7. Adjust loss configuration in the train_whole.py file ('step' function of U_net class) to your needs
    8. Run train_whole.py on a computer (GPU needed) or cluster 
    9. train_whole.py also includes prediction on the test set, to visualize your results use visualize.py 
    10. Done!

Tu run the Baseline follow the following procedure:
    1. Adjust save paths in the config.py file to your needs
    2. Set login key for wandb in the train.py file (you can find it in your account info from wandb)
    3. Run train_baseline.py on a computer (GPU needed) or cluster
    4. train_baseline.py also includes prediction on the test set, to visualize your results use visualize.py 
    5. Done!






