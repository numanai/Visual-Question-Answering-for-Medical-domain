import re
import pandas as pd
import csv
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


class Utility(object):
    @staticmethod
    def clean(df):
        # dictionary of replaceable words
        
        replace_to = {"magnetic resonance imaging":"mri",
        "mri scan":"mri",
        "t1w":"t1",
        "t2w":"t2",
        "weighted":"",
        "shows": "show",
        "reveal":"show",
        "demonstrated":"demonstrate",
        "demonstrate":"show",
        "ct scan":"ct",
        "mri":"mr",
        "mr":"mri",
        "gradientgrempgrswanswi":"",
        "unilteral":"unilateral",
        "appendigitis":"appendicitis",
        "congital":"congenital",
        "cytsic":"cystic",
        "boipsy":"biopsy",
        "dislocaton":"dislocation",
        "julgular":"jugular",
        "chrohn": "crohn",
        "o’donahue":"odonahue" #handing apostrophe
        }

        
        df = df.replace(replace_to, regex=True)

        df = df.apply(lambda x: x.str.replace(','," "))
        df = df.apply(lambda x: x.str.replace('!'," "))
        df = df.apply(lambda x: x.str.replace('-'," "))
        df = df.apply(lambda x: x.str.replace('"'," "))
        df = df.apply(lambda x: x.str.replace("'"," "))
        df = df.apply(lambda x: x.str.replace(':'," "))
        df = df.apply(lambda x: x.str.replace("&", "and"))
        df = df.apply(lambda x: x.str.replace("("," "))
        df = df.apply(lambda x: x.str.replace(")"," "))
        df = df.apply(lambda x: x.str.replace('\\'," "))
        df = df.apply(lambda x: x.str.replace('/'," "))
        df = df.apply(lambda x: x.str.replace("'s "," "))
        df = df.apply(lambda x: x.str.replace('?',' ?'))
        df = df.apply(lambda x: x.str.replace("\s+", " ",regex=True))
        df = df.apply(lambda x: x.str.lower())
        df = df.apply(lambda x: x.str.strip())
        #df = df.apply(lambda x: x.str.split(' '))


        return df['question'], df['answer']

    @staticmethod
    def read_dataset(dataset,year):
        path="Data/"+dataset+"_"+year+"/All_QA_Pairs_"+dataset+"_"+year+".txt"
        df =pd.read_csv(path, sep='|', header=None, quoting=csv.QUOTE_NONE)
        if "val" in dataset:
            df = df.rename(columns={0: 'image_name', 1:"question", 2:"answer"})
        else:
            df = df.rename(columns={0: 'image_name', 1:"question", 2:"answer"})
        print(dataset+" data size=",len(df))
        images = []
        for i in df["image_name"]:

            fname = "Data/"+dataset+"_"+year+"/"+dataset+"_images/"+i+".jpg"
            images.append(fname)
        if "val" in dataset:
            return images, df
        else:
            return images, df

    @staticmethod
    def show_image(id, images, questions, answers):
        fname = images[id]
        img=mpimg.imread(fname, format="jpg")
        print ("Image name :", fname)
        print ("Question   :", questions[id])
        print ("Answer     :", answers[id] )
        plt.imshow(img)
        plt.show()