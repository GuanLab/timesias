import glob
import numpy as np

def load_challenge_data(file):
    with open(file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')
    return data

NEW=open('eight_hour_original_cutoff.txt','w')
FILE=open('train_gs.dat','r')
all_train=glob.glob('../../preprocess_code/physionet_format/*')
for the_file in all_train:
    try:
        whole_train=load_challenge_data(the_file)
        print(whole_train[:,0][-1])
        NEW.write('%.1f\n' % whole_train[:,0][-1])
    except:
        whole_train=load_challenge_data(the_file)
        print(whole_train[0])
        NEW.write('%.1f\n' % whole_train[0])
        
NEW.close()
FILE.close()
