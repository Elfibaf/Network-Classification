import numpy as np
import arff
import matplotlib.pyplot as plt
from matplotlib import cm

def stats(filename):

        #Loading data
        barray = []
        for row in arff.load(filename):
             barray.append(list(row))
        labels = []

        #Retrieving labels
        for row in barray:
            labels.append(row.pop())
        labels = np.array(labels)

        #Counting the number of each label and pitting it in a dictionary
        dict_labels = {}
        for label in labels:
            if not label in dict_labels:
                dict_labels[label] = 1
            else:
                dict_labels[label] += 1

        #Getting total number of samples
        total = 0
        for key, value in dict_labels.items():
            total += value
        print(total)

        #Calculus of statistics for each label then adding it into pct_labels
        pct_labels = {}
        for key, value in dict_labels.items():
            pct_labels[key] = (float(value)/total)*100
            #We only keep the most represented labels in the data
            if pct_labels[key] < 1:
                dict_labels.pop(key, None)

        #Printing stats
        for l in sorted(pct_labels, key=pct_labels.get, reverse = True):
            print(l, " : %.4f" % pct_labels[l], "%")

        return dict_labels

    
def pie_chart(filename):

        #Loading dictionary with the most represented labels
        labels = stats(filename)

        #Creating a color set
        cs = cm.Set1(np.arange(10)/10.)

        #Creating pie chart
	
	plt.title(filename) 
        plt.pie([v for k,v in labels.items()],
                labels = [k for k,v in labels.items()],
                colors = cs,
                autopct = '%1.1f%%',
                shadow = True)
        plt.axis('equal')
        plt.show()

        
       
filename = raw_input("Entry filename:")
foo = pie_chart(filename)

