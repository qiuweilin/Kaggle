import numpy as np
import pandas as pd
import re
import csv

def weight(item):
    if item == 'horse':
        weight =  max(0, np.random.normal(5,2,1)[0])
    elif item == 'ball':
        weight = max(0, 1 + np.random.normal(1,0.3,1)[0])
    elif item == 'bike':
        weight = max(0, np.random.normal(20,10,1)[0])
    elif item == 'train':
        weight = max(0, np.random.normal(10,5,1)[0])
    elif item == 'coal':
        weight = 47 * np.random.beta(0.5,0.5,1)[0]
    elif item == 'book':
        weight = np.random.chisquare(2,1)[0]
    elif item == 'doll':
        weight = np.random.gamma(5,1,1)[0]
    elif item == 'blocks':
        weight = np.random.triangular(5,10,20,1)[0]
    elif item == 'gloves':
        weight = 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 \
                                            else np.random.rand(1)[0]
    return weight

if __name__ == '__main__':

    # load gift list as dataframe
    unsorted_gifts = pd.read_csv('santa_gifts.csv')

    # simulate weight of each gift. Add weights to new column in dataframe
    gift_regex = re.compile(r'([a-zA-Z])+')
    weights = []
    for i, row in unsorted_gifts.iterrows():
        gift = gift_regex.search(row['GiftId']).group()
        gift_weight = weight(gift)
        weights.append(gift_weight)

    unsorted_gifts = unsorted_gifts.assign(weights = weights)

    # TODO: Sort gifts into 1000 bags
    gift_id = 0
    sorted_gifts = []
    sorted_gifts.append(['Gifts'])

    for i in range(0,1000):
        sorted_gift = []
        str = ''
        bag_weight = 0
        while bag_weight < 50:
            if gift_id < len(unsorted_gifts):
                str = str + unsorted_gifts.iloc[gift_id, 0] + ' '
                bag_weight += unsorted_gifts.iloc[gift_id, 1]
                gift_id += 1
            else:
                break

        sorted_gift.append(str)
        sorted_gifts.append(sorted_gift)

    resultFile = open("output.csv",'wb')
    wr = csv.writer(resultFile, delimiter = ',')
    wr.writerows(sorted_gifts)


    # TODO: Check each bag must have 3 or more gifts

    # TODO: Check no gift used more than once
