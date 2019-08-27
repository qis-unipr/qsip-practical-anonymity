import numpy as np
import matplotlib.pyplot as plt
from math import floor

def main():
    bins = 32
    counter = np.zeros(bins)
    total = np.zeros(bins)

    with open('prova.txt','r') as f:
        for line in f:
            line = line.split(',')
            print(line[-1].split(']')[0])
            position = floor(int(line[-1].split(']')[0])*bins/128)
            total[position] += 1
            if int(line[2]) == 102:
                counter[position] += 1
    

    for n in range(bins):
        if total[n] != 0:
            counter[n] = counter[n]/total[n]

    counter[31]=1
    print(counter)
    print(total)

    plt.bar(x=[x for x in range(bins)], height=counter, facecolor='#0504aa', alpha=0.75)#, align='edge')
    plt.xlabel('Success')
    plt.ylabel('Angles')
    plt.axis([-1, bins, 0, 1.0]) 
    plt.xticks(np.arange(bins), ('0', '','','','','','','','','','','','','','', '',u'\u03C0'+'/2', '','','','','','','','','','','','','','', u'\u03C0'))
    plt.grid(axis='y', alpha=0.5)
    plt.show()

if __name__ == '__main__':
    main()