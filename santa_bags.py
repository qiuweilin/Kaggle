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
    # sorts gifts into no_bag and
    # (1) updates DataFrame with new weights of bags in 'test_weight'
    # (2) returns list of gift sorted,
    # (3) weights of the gift sorted, as a list
    # (4) and the gifts that are still unsorted after this

    unsorted_gifts.index = range(0, len(unsorted_gifts))

    # if less than 1000 gifts, sort only whatever gifts is left into bags
    if len(no_bag) > len(unsorted_gifts) :
        no_bag = list(range(0,len(unsorted_gifts)))
    gift_list = unsorted_gifts.iloc[no_bag,0].tolist()
    weights = unsorted_gifts.iloc[no_bag,1].tolist()

    if index == 0:
        sorted_gifts_df['test_weight'] = weights
    else:
        # update test_weights for bags that have gifts added.
        updated_bags =\
        sorted_gifts_df.loc[sorted_gifts_df.index[0:len(no_bag)],'test_weight']
        updated_bags = updated_bags + weights
        sorted_gifts_df.loc[sorted_gifts_df.index[0:len(no_bag)],'test_weight']=\
        updated_bags

    # drop gifts that have been sorted into bags, from unsorted_gifts
    unsorted_gifts = unsorted_gifts.drop(unsorted_gifts.index[no_bag])

    return gift_list, weights, unsorted_gifts

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
                                   columns=['total_weight', 'test_weight'],
                                   index=range(0,1000))
    sorted_gifts_df.index.names = ['Bag']

    # keep adding light items into bag as long as bags are <50lbs
    no_bag = list(range(0,1000))
    index = 0  # track number of gifts in bags
    while True:
        # sort into bags and find weight after adding gifts
        sorted_gifts_df = sorted_gifts_df.assign(test_weight=
                                                 sorted_gifts_df['total_weight'])
        gift_list, weights, unsorted_gifts = \
        sort_into_bags(sorted_gifts_df, unsorted_gifts, index, no_bag)

        # if all bags weigh less than 50 after adding gifts,
        # add all gifts to bag into new column.
        # Update total weight as the test_weight
        if max(sorted_gifts_df['test_weight']) < 50:
            sorted_gifts_df['total_weight'] = sorted_gifts_df['test_weight']
            if len(gift_list) < len(sorted_gifts_df):
                missing_len = len(sorted_gifts_df) - len(gift_list)
                a = np.empty(missing_len) * np.nan
                gift_list = gift_list + a.tolist()

            sorted_gifts_df = sorted_gifts_df.assign(gift=gift_list)
            col = len(sorted_gifts_df.columns)
            sorted_gifts_df = sorted_gifts_df.rename(columns =
                                                    {sorted_gifts_df.columns[col-1]:
                                                    'gift_%s' % (index+1)})
            index += 1

        # find bags that exceed 50lbs, return those gifts to unsorted_gifts
        # update the bags that do not exceed 50lbs, with the new gifts

        elif max(sorted_gifts_df['test_weight']) > 50:
            temp = sorted_gifts_df.ix[sorted_gifts_df['test_weight']<50]
            unused_df = sorted_gifts_df.ix[sorted_gifts_df['test_weight']>50]

            if len(unused_df) == len(gift_list):
                # breaks if no bags can take gifts without exceeding 50lbs
                break

            # put back un-used gifts into unsorted_gifts
            unused_list = [gift_list[x] for x in unused_df.index.tolist()]
            unused_weights = [weights[x] for x in unused_df.index.tolist()]
            unused = pd.DataFrame({'GiftId': unused_list,
                                   'weights': unused_weights})
            unsorted_gifts = pd.concat([unsorted_gifts, unused])
            unsorted_gifts = unsorted_gifts.sort_values('weights',
                                                        axis=0,
                                                        ascending=1)
            # for bags that do not add new gifts
            # un-do test weight and set test_weight as the current weight without
            # additional gifts
            unused_df = unused_df.assign(test_weight=
                                         unused_df['total_weight'])
            sorted_gifts_df.update(unused_df)

            # update bags that can add gifts without exceeding 50lbs
            if len(gift_list) > len(temp):
                gift_list = [gift_list[x] for x in temp.index.tolist()]
            elif len(gift_list) < len(temp):
                missing_len = len(temp) - len(gift_list)
                empty_list = np.empty(missing_len) * np.nan
                gift_list = gift_list + empty_list

            temp = temp.assign(gift=gift_list)
            sorted_gifts_df = sorted_gifts_df.assign(gift=np.nan)
            sorted_gifts_df.update(temp)
            col = len(sorted_gifts_df.columns)
            sorted_gifts_df = sorted_gifts_df.rename\
            (columns={sorted_gifts_df.columns[col-1]: 'gift_%s' % (index+1)})
            # re-sort sorted_gifts_df by current weights
            sorted_gifts_df['total_weight'] = sorted_gifts_df['test_weight']
            sorted_gifts_df = sorted_gifts_df.sort_values('total_weight',
                                                          axis=0,
                                                          ascending=1)
            index += 1

        if unsorted_gifts.empty:
            # if no gifts left to sort, break out of loop.
            break

    # converting data in DataFrame into list of list of strings
    sorted_gifts = [['Gifts']]
    sorted_gifts_df = sorted_gifts_df.drop(['total_weight'],1)
    sorted_gifts_df = sorted_gifts_df.drop(['test_weight'],1)

    for i in range(0, len(sorted_gifts_df)):
        sorted_list = sorted_gifts_df.iloc[i,:]
        sorted_list = sorted_list.dropna(how='any')
        sorted_list = ' '.join(sorted_list)
        sorted_gifts.append([sorted_list])

    # write result to csv file
    resultFile = open("output.csv",'wb')
    wr = csv.writer(resultFile, delimiter = ',')
    wr.writerows(sorted_gifts)
