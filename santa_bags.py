import numpy as np
import pandas as pd
import re
import csv

def weight(item):
    # to simulate weight of item
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


def simulate_weights(unsorted_gifts):
    # find the name of the item and simulate the weight through
    # function weight.
    # adds weights as a column in the dataframe unsorted_gifts.
    gift_regex = re.compile(r'([a-zA-Z])+')
    weights = []
    for i, row in unsorted_gifts.iterrows():
        gift = gift_regex.search(row['GiftId']).group()
        # simulate the weight of the item
        gift_weight = weight(gift)
        weights.append(gift_weight)

    # add weights of each gift to a new column
    unsorted_gifts = unsorted_gifts.assign(weights = weights)

    return unsorted_gifts


def sort_into_bags(sorted_gifts_df, unsorted_gifts, index, no_bag):
    # sorts the first 3000 lightest gifts into 1000 bags
    gifts = unsorted_gifts.iloc[no_bag,0].tolist()
    weights = unsorted_gifts.iloc[no_bag,1].tolist()
    sorted_gifts_df = sorted_gifts_df.assign(gift=gifts)
    col = len(sorted_gifts_df.columns)
    sorted_gifts_df = sorted_gifts_df.rename(columns =
                                            {sorted_gifts_df.columns[col-1]:
                                            'gift_%s' % (index+1)})

    if index == 0:
        sorted_gifts_df['total_weight'] = weights
    else:
        sorted_gifts_df = sorted_gifts_df.assign(total_weight=
                                             sorted_gifts_df['total_weight']
                                             + weights)
    unsorted_gifts = unsorted_gifts.drop(unsorted_gifts.index[no_bag])

    return sorted_gifts_df, unsorted_gifts

if __name__ == '__main__':

    # load gift list as dataframe
    unsorted_gifts = pd.read_csv('santa_gifts.csv')

    # simulate weight of each gift. Add weights to new column in dataframe
    unsorted_gifts = simulate_weights(unsorted_gifts)

    # only use gifts that are less than 50lbs
    unsorted_gifts = unsorted_gifts.sort_values('weights', axis=0, ascending=1)
    unsorted_gifts = unsorted_gifts.dropna(how='any')
    unsorted_gifts = unsorted_gifts.ix[unsorted_gifts['weights'] < 50]

    # setting up DataFrame to store bag information
    sorted_gifts_df = pd.DataFrame(np.nan, \
                                   columns=['total_weight'],
                                   index=range(0,1000))
    sorted_gifts_df.index.names = ['Bag']

    # sort the lightest 3000 gifts into the bags
    no_bag = list(range(0,1000))
    index = 0
    while True:
        if sorted_gifts_df.ix[sorted_gifts_df['total_weight']>50].empty:
            sorted_gifts_df, unsorted_gifts = sort_into_bags(sorted_gifts_df,
                                                             unsorted_gifts,
                                                             index, no_bag)
            index += 1
        else:
            no_bag
            break

    print("reached here")
    print sorted_gifts_df
    # converting data in DataFrame into list of list of strings
    sorted_gifts = [['Gifts']]
    sorted_gifts_df = sorted_gifts_df.drop(['total_weight'],1)

    for i in range(0, len(sorted_gifts_df)):
        sorted_list = sorted_gifts_df.iloc[i,:]
        sorted_list = ' '.join(sorted_list)
        sorted_gifts.append([sorted_list])

    print sorted_gifts
    # write result to csv file
    resultFile = open("output.csv",'wb')
    wr = csv.writer(resultFile, delimiter = ',')
    wr.writerows(sorted_gifts)
