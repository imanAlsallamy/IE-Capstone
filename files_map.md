# Project Structure

/root  
├── **legacy_versions**: Folder of previous versions  
├── **model_pickles**: Folder of saved models  
├── **divide_dataset_train_test_sim.sql**: Script used to divide the data for train and simulate  
├── **config.json**: DB Parameters  
├── **data_loading.py**: Contains connection to DB, simple functions for read and write on DB  
├── **eda.ipynb**: Used to explore and understand the dataset with no manipulation done  
├── **preprocessing.py**: Data engineering and prepare the data for the model  
├── **training.py**: This file used to train our models and save it in pickles  
├── **predict.py**: This file used to predict the probabilities of both exited (0, 1) on the simulation data  
├── **used-models.py**: Contains the models we explored
└── **main.ipynb**: Main notebook to run the entire project  
