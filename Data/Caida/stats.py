import numpy as np
import arff
import matplotlib.pyplot as plt

def stats(filename):

        barray = []
        for row in arff.load(filename):
             barray.append(list(row))
        labels = []
        for row in barray:
            labels.append(row.pop())
        labels = np.array(labels)
    
        dict_labels = {}
        for label in labels:
            if not label in dict_labels:
                dict_labels[label] = 1
            else:
                dict_labels[label] += 1

        
        total = 0
        for key, value in dict_labels.items():
            total += value
        print(total)

        pct_labels = {}
        
        for key, value in dict_labels.items():
            pct_labels[key] = (float(value)/total)*100
            if pct_labels[key] < 1:
                dict_labels.pop(key, None)
        
        for l in sorted(pct_labels, key=pct_labels.get, reverse = True):
            print(l, " : %.4f" % pct_labels[l], "%")

        return dict_labels

    

def pie_chart(filename):

        labels = stats(filename)

        plt.pie([v for k,v in labels.items()],
                labels = [k for k,v in labels.items()],
                autopct = '%1.1f%%',
                shadow = True)
        plt.axis('equal')
        plt.show()

        
        


foo = pie_chart('data_caida_original.arff')
#print(foo)
