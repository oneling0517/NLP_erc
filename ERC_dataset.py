from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence
import random
import pandas as pd
import numpy as np
    
class MELD_loader(Dataset):
    def __init__(self, txt_file, dataclass):
        self.dialogs = []
        
        f = open(txt_file, 'r')
        dataset = f.readlines()
        f.close()
        
        temp_speakerList = []
        context = []
        context_speaker = []
        self.speakerNum = []
        # 'anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'
        emodict = {'anger': "anger", 'disgust': "disgust", 'fear': "fear", 'joy': "joy", 'neutral': "neutral", 'sadness': "sad", 'surprise': 'surprise'}
        self.sentidict = {'positive': ["joy"], 'negative': ["anger", "disgust", "fear", "sadness"], 'neutral': ["neutral", "surprise"]}
        self.emoSet = set()
        self.sentiSet = set()
        for i, data in enumerate(dataset):
            if i < 2:
                continue
            if data == '\n' and len(self.dialogs) > 0:
                self.speakerNum.append(len(temp_speakerList))
                temp_speakerList = []
                context = []
                context_speaker = []
                continue
            speaker, utt, emo, senti = data.strip().split('\t')
            context.append(utt)
            if speaker not in temp_speakerList:
                temp_speakerList.append(speaker)
            speakerCLS = temp_speakerList.index(speaker)
            context_speaker.append(speakerCLS)
            
            self.dialogs.append([context_speaker[:], context[:], emodict[emo], senti])
            self.emoSet.add(emodict[emo])
            self.sentiSet.add(senti)
        
        self.emoList = sorted(self.emoSet)  
        self.sentiList = sorted(self.sentiSet)
        if dataclass == 'emotion':
            self.labelList = self.emoList
        else:
            self.labelList = self.sentiList        
        self.speakerNum.append(len(temp_speakerList))
        
    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return self.dialogs[idx], self.labelList, self.sentidict
    
    
class NLP_loader(Dataset):
    def __init__(self, txt_file, dataclass):
        self.dialogs = []
        
        train_path = "/content/dataset/fixed_train.csv"
        test_path = "/content/dataset/fixed_test.csv"
        val_path = "/content/dataset/fixed_valid.csv"

        data_train = pd.read_csv(train_path, encoding='utf-8')
        data_valid = pd.read_csv(val_path, encoding='utf-8')
        data_test = pd.read_csv(test_path, encoding='utf-8')

        data_train = np.array(data_train) #np.ndarray()
        dataset = data_train.tolist()
        
        """sentiment"""
        pos = ["joyful", "trusting", "faithful", "excited", "anticipating", "content", "confident", "grateful", "hopeful"]
        neg = ["sad", "terrified", "disappointed", "jealous", "disgusted", "ashamed", "afraid", "sentimental", "devastated", "annoyed", "anxious", "furious", "lonely", "angry", "apprehensive", "guilty"]
        neu = ["caring", "surprised", "impressed", "embarrassed", "proud", "prepared", "nostalgic"]
        emodict = {'sad': 0, 'trusting': 1, 'terrified': 2, 'caring': 3, 'disappointed': 4,'faithful': 5, 'joyful': 6, 'jealous': 7, 'disgusted': 8, 'surprised': 9,
        'ashamed': 10, 'afraid': 11, 'impressed': 12, 'sentimental': 13, 'devastated': 14, 'excited': 15, 'anticipating': 16, 'annoyed': 17, 'anxious': 18,
        'furious': 19, 'content': 20, 'lonely': 21, 'angry': 22, 'confident': 23, 'apprehensive': 24, 'guilty': 25, 'embarrassed': 26, 'grateful': 27,
        'hopeful': 28, 'proud': 29, 'prepared': 30, 'nostalgic': 31}
        
        #temp_speakerList = []
        context_speaker = []        
        self.speakerNum = []
        
        context = []
        self.emoSet = set()
        #self.sentiSet = set()
        
        for i in range(len(dataset)):
            if dataset[i][1] == 1: #一輪新的對話
                #speakerNum.append(len(temp_speakerList))
                context = []
                context_speaker = []

                speaker = 0
                context_speaker.append(speaker)
                context.append(dataset[i][2]) #prompt
                #continue
              #speaker, utt, emo = data.strip().split('\t')
            if dataset[i][1] % 2 != 0:
                speaker = 1
            else:
                speaker = 2
            utt = dataset[i][3]
            #print("utt:",utt)
            emo = dataset[i][4]
            #print("emo:",emo)
            context.append(utt)
            #print("context:",context)

            sentiment = list(emodict.keys())[list(emodict.values()).index(emo)]
            if sentiment in pos:
                senti = "positive"
            elif sentiment in neg:
                senti = "negative"
            elif sentiment in neu:
                senti = "neutral"
            else:
                print('ERROR emotion&sentiment')
                
#             if speaker not in temp_speakerList:
#                 temp_speakerList.append(speaker)
#             speakerCLS = temp_speakerList.index(speaker)
            context_speaker.append(speaker)
            
            self.dialogs.append([context_speaker[:], context[:], emo, senti])
#             self.dialogs.append([context[:], emo, senti])
            self.emoSet.add(emo)
            self.sentiSet.add(senti)
            
        self.emoList = sorted(self.emoSet)
        #self.sentiList = sorted(self.sentiSet)
        
        if dataclass == 'emotion':
            self.labelList = self.emoList
#         else:
#             self.labelList = self.sentiList        
#         self.speakerNum.append(len(temp_speakerList))
        
    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return self.dialogs[idx], self.labelList
#         return self.dialogs[idx], self.labelList, self.sentidict
    
    
class IEMOCAP_loader(Dataset):
    def __init__(self, txt_file, dataclass):
        self.dialogs = []
        
        f = open(txt_file, 'r')
        dataset = f.readlines()
        f.close()
        
        temp_speakerList = []
        context = []
        context_speaker = []
        self.speakerNum = []
        pos = ['ang', 'exc', 'hap']
        neg = ['fru', 'sad']
        neu = ['neu']
        emodict = {'ang': "angry", 'exc': "excited", 'fru': "frustrated", 'hap': "happy", 'neu': "neutral", 'sad': "sad"}
        self.sentidict = {'positive': pos, 'negative': neg, 'neutral': neu}
        # use: 'hap', 'sad', 'neu', 'ang', 'exc', 'fru'
        # discard: disgust, fear, other, surprise, xxx        
        self.emoSet = set()
        self.sentiSet = set()
        for i, data in enumerate(dataset):
            if data == '\n' and len(self.dialogs) > 0:
                self.speakerNum.append(len(temp_speakerList))
                temp_speakerList = []
                context = []
                context_speaker = []
                continue
            speaker = data.strip().split('\t')[0]
            utt = ' '.join(data.strip().split('\t')[1:-1])
            emo = data.strip().split('\t')[-1]
            context.append(utt)
            
            if emo in pos:
                senti = "positive"
            elif emo in neg:
                senti = "negative"
            elif emo in neu:
                senti = "neutral"
            else:
                print('ERROR emotion&sentiment')                        
            
            if speaker not in temp_speakerList:
                temp_speakerList.append(speaker)
            speakerCLS = temp_speakerList.index(speaker)
            context_speaker.append(speakerCLS)
            
            self.dialogs.append([context_speaker[:], context[:], emodict[emo], senti])
            self.emoSet.add(emodict[emo])
        
        self.emoList = sorted(self.emoSet)   
        self.sentiList = sorted(self.sentiSet)
        if dataclass == 'emotion':
            self.labelList = self.emoList
        else:
            self.labelList = self.sentiList        
        self.speakerNum.append(len(temp_speakerList))
        
    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return self.dialogs[idx], self.labelList, self.sentidict
    
class DD_loader(Dataset):
    def __init__(self, txt_file, dataclass):
        self.dialogs = []
        
        f = open(txt_file, 'r')
        dataset = f.readlines()
        f.close()
        
        temp_speakerList = []
        context = []
        context_speaker = []
        self.speakerNum = []      
        self.emoSet = set()
        self.sentiSet = set()
        # {'anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'}
        pos = ['happiness']
        neg = ['anger', 'disgust', 'fear', 'sadness']
        neu = ['neutral', 'surprise']
        emodict = {'anger': "anger", 'disgust': "disgust", 'fear': "fear", 'happiness': "happy", 'neutral': "neutral", 'sadness': "sad", 'surprise': "surprise"}
        self.sentidict = {'positive': pos, 'negative': neg, 'neutral': neu}
        for i, data in enumerate(dataset):
            if data == '\n' and len(self.dialogs) > 0:
                self.speakerNum.append(len(temp_speakerList))
                temp_speakerList = []
                context = []
                context_speaker = []
                continue
            speaker = data.strip().split('\t')[0]
            utt = ' '.join(data.strip().split('\t')[1:-1])
            emo = data.strip().split('\t')[-1]
            
            if emo in pos:
                senti = "positive"
            elif emo in neg:
                senti = "negative"
            elif emo in neu:
                senti = "neutral"
            else:
                print('ERROR emotion&sentiment')                
            
            context.append(utt)
            if speaker not in temp_speakerList:
                temp_speakerList.append(speaker)
            speakerCLS = temp_speakerList.index(speaker)
            context_speaker.append(speakerCLS)
            
            self.dialogs.append([context_speaker[:], context[:], emodict[emo], senti])
            self.emoSet.add(emodict[emo])
        
        self.emoList = sorted(self.emoSet)   
        self.sentiList = sorted(self.sentiSet)
        if dataclass == 'emotion':
            self.labelList = self.emoList
        else:
            self.labelList = self.sentiList        
        self.speakerNum.append(len(temp_speakerList))
        
    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return self.dialogs[idx], self.labelList, self.sentidict
