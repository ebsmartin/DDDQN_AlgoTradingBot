import tweepy
import csv
import time
import pandas as pd
from datetime import datetime as dt

def get_num_tweets(client, query, start_time, end_time, starting_row=0):
    try: 
        num_tweets_df = pd.read_csv('num_tweets_v2.csv')
        num_tweets_df = num_tweets_df.iloc[starting_row:]
        total_tweets = num_tweets_df["tweet_count"].sum()
    except:
        paginator = tweepy.Paginator(client.get_all_tweets_count,
                                query=query,
                                start_time=start_time,
                                end_time=end_time,
                                granularity="hour"
                                )
        total_tweets = 0
        df_list = []
        for page in paginator:
            df_list.append(pd.DataFrame(page[0]))
            print("fetched page")

        num_tweets_df = pd.concat(df_list)
        num_tweets_df.to_csv('num_tweets_v2.csv')
        num_tweets_df = num_tweets_df.iloc[starting_row:]
        total_tweets = num_tweets_df["tweet_count"].sum()

    return num_tweets_df, total_tweets

def timestamp_to_datetime(timestamp):
    return dt.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%f')

def write_tweets_csv(tweetlist):
    with open(r'eth_tweets.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            for tweet in tweetlist:
                try:
                    writer.writerow(tweet)
                except UnicodeEncodeError:
                    print("UnicodeEncodeError, skipping row")
                    continue

def get_tweets(client, query, num_tweets_df, ratio):
    request_counter = 0
    running_total = 0
    paginator = None
    row_index = 0
    exception_counter = 0
    
    while row_index < len(num_tweets_df):
        row = num_tweets_df.iloc[row_index]
        tweetlist = []
        try:
            paginator = tweepy.Paginator(client.search_all_tweets,
                                    query=query,
                                    tweet_fields=['context_annotations', 'created_at', 'public_metrics', 'entities', 'geo', 'source', 'referenced_tweets', 'conversation_id'],
                                    start_time=row["start"],
                                    end_time=row["end"],
                                    expansions='author_id',
                                    max_results=100
                                    )

            num_tweets_to_fetch = max(100, int(row["tweet_count"]/ratio))
            done = False
            tweet_counter = 0
            t0 = time.time()

            for page in paginator:
                if not page[0]:
                    break  # if the page is empty, move on (no tweets on this page of the paginator)

                for tweet in page[0]:
                    if tweet_counter >= num_tweets_to_fetch:
                        done = True
                        break

                    tweet = (
                        int(tweet.id),
                        tweet.created_at,
                        int(tweet.author_id),
                        int(tweet.conversation_id),
                        tweet.source.encode('utf-8'),
                        tweet.geo["coordinates"]["coordinates"] if tweet.geo and "coordinates" in tweet.geo else None,
                        tweet.geo["place_id"] if tweet.geo and "place_id" in tweet.geo else None,
                        [int(x["id"]) for x in tweet.entities["mentions"]] if tweet.entities and "mentions" in tweet.entities else None,
                        [x["tag"].encode('utf-8') for x in tweet.entities["hashtags"]] if tweet.entities and "hashtags" in tweet.entities else None,
                        [x["unwound"]["url"].encode('utf-8') if "unwound" in x else x["expanded_url"].encode('utf-8') for x in tweet.entities["urls"]] if tweet.entities and "urls" in tweet.entities else None,
                        tweet.text.encode('utf-8')
                        )

                    tweetlist.append(tweet)
                    tweet_counter += 1
                    running_total += 1

                request_counter += 1

                t1 = time.time() - t0
                t0 = time.time()
                time.sleep(max(0, 1.2-t1))  # we need to wait at least 1.2 second between requests (API documentations says 1 second but I've found it a bit innacurate)

                if done:
                    break

            print(f"done request {request_counter}. row {row.name} ({row['start']} to {row['end']}). Fetched {running_total} total tweets")
            write_tweets_csv(tweetlist)
            row_index += 1  # this is only reached if we didn't get any errors until now.
            exception_counter = 0
        except Exception as e:
            exception_counter += 1
            if exception_counter % 10 == 0:
                print(f"Error fetching data for row {row.name} ({row['start']} to {row['end']}). Trying again in 10s.")
                print(e)
                time.sleep(10)
            else:
                time.sleep(1)
            continue

def read_tweets():
    tweetlist = pd.read_csv('tweetlist.csv', header=None)
    return tweetlist


BEARER_TOKEN = "<REPLACE WITH BEARER TOKEN>"
NUM_TWEETS_TO_SEARCH = 1000000
START_ROW = 0 # if there's an error and we need to pick up from a certain point, we set the value here for the line that had an error

def main():
    client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=False)
    query = '(#eth OR ethereum OR $ETH) lang:en'
    start_time = '2017-01-01T00:00:00.000Z'
    end_time = '2022-07-21T00:00:00.000Z'

    num_tweets_df, total_tweets = get_num_tweets(client, query, start_time, end_time, starting_row=START_ROW)
    ratio = total_tweets / NUM_TWEETS_TO_SEARCH

    get_tweets(client, query, num_tweets_df, ratio)

if __name__ == "__main__":
    main()