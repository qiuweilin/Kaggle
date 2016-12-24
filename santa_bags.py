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
    unsorted_gifts = unsorted_gifts.sort_values('weights', axis=0, ascending=0)
    all_gifts = unsorted_gifts
    all_gifts = all_gifts.assign(bag_number = np.nan)

    # Sort gifts into 1000 bags
    sorted_gifts = []  # list of each bag
    sorted_gifts.append(['Gifts'])  # list of list of each bag

    for i in range(0,1000):
        str1 = ''
        bag_weight = 0
        gift_count = 0
        # keep adding gift to bag if total weight is less than 50
        while (bag_weight + unsorted_gifts.iloc[0, 1]) < 50:
            # script breaks out of loop if run out of gifts to sort
            if not unsorted_gifts.empty:
                str1 = str1 + unsorted_gifts.iloc[0, 0] + ' '
                bag_weight += unsorted_gifts.iloc[0, 1]
                # once in bag, remove gift from the available gift list
                gift_id = unsorted_gifts.index[0]
                all_gifts.set_value(gift_id, 'bag_number', i)
                unsorted_gifts = unsorted_gifts.iloc[1:,]
                gift_count += 1

            else:
                break

        # TODO: Check each bag must have 3 or more gifts

        while gift_count < 3 :
            df = unsorted_gifts
            gift_to_drop = []
            # find gifts that would not make bag weight heavier than 50
            print (df.iloc[len(df)-10:,])
            df = df.ix[df['weights'] < (50-bag_weight)]
            if not df.empty:
                # sort so that you add the heaviest gift first to the bag
                # this ensures smaller gifts are kept in the list to fill in
                # bags that can only take smaller gifts
                df = df.sort_values('weights', axis=0, ascending=0)
                str1 = str1 + df.iloc[0, 0] + ' '
                bag_weight += df.iloc[0, 1]
                gift_to_drop.append(df.index[0])
                unsorted_gifts = unsorted_gifts.drop(gift_to_drop)
                all_gifts.set_value(gift_to_drop[0], 'bag_number', int(i+1))

            else:
                print ("Bag number: " + str(i) + " only has " + str(gift_count) + "gifts")
                print ("Bag weight: " + str(bag_weight))

                # TODO: have to add code to take smaller gifts out of other bags
                # and add it to this bag. then add another gift to other bag

            gift_count += 1

        sorted_gifts.append([str1])

        # TODO: have to sort unsorted gifts into bags that still have space
        #while len(unsorted_gifts) != 0:

    resultFile = open("output.csv",'wb')
    wr = csv.writer(resultFile, delimiter = ',')
    wr.writerows(sorted_gifts)
