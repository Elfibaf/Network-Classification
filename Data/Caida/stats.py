import numpy as np
import arff


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

        for key, value in dict_labels.items():
            dict_labels[key] = (value/total)*100
        
        for l in sorted(dict_labels, key=dict_labels.get, reverse = True):
            print(l, " : %.4f" % dict_labels[l], "%")

        return dict_labels

    

    


foo = stats('Caida/data_caida_original.arff')
#print(foo)
