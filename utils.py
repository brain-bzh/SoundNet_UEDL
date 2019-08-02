#camila
import os
from tqdm import tqdm
#Preprocessing audio folder

def audiopath_resample(audiopath,outputpath = 'audio22k' ,sr = 22050):
    """This method changes the sampling rate and the number of channels for each .wav file in the audiopath
    """
    resultdir = os.path.join(outputpath)
    os.makedirs(resultdir,exist_ok=True)
    listwav = []
    for c in os.listdir(audiopath):
        if c[-3:] == 'wav':
            listwav.append(c)           
    
    if(len(listwav) == 0):
        print('There are not wav files in :',audiopath)
        return 
    
    for c in tqdm(listwav):
        command = "sox {} -r {} -c 1 {}/{}_.wav".format(audiopath+c,sr,outputpath,c[:-4])
        print(command)
        if (os.system(command) != 0): return                
    print("Audio saved in ",resultdir)
    
def audio_resample(audiopath ,sr = 22050):
    command = "sox {} -r {} -c 1 {}/{}_.wav".format(audiopath,sr,outputpath,c[:-4])
    print(command)
    if (os.system(command) != 0): return     

        
    
