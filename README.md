# **NOTE 1**

    - Edit file 
    'DeepLearning-CodeSmell\DeepSmells\program\ease_deepsmells\utils.py' 
    
    To change the path: 
    sys.path.insert(0, r'/content/drive/MyDrive/LabRISE/DeepLearning-CodeSmell/DeepSmells/program/dl_models')
    
    By the absolute path of the folder:
    'DeepLearning-CodeSmell/DeepSmells/program/dl_models'

# **NOTE 2**

    - File pickle includes 3 columns: 'embedding', 'name', 'label'
        + 'embedding': the embedding of the file.java
        + 'name': the name of the file.java without extension .java
        + 'label': the label of the file.java (0: non-smelly, 1: smelly)