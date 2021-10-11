# Individual emotion prediction using self and neighbor emotion time series combined with T-LSTM and memory attentionmechanism (SAEP)
This project implements the surrounding-aware time series prediction method, which is an prediction algorithm for time series data in the social network.
## ***Usage
To try the code, we provide a twitter dataset “Twitter” as an example, which includes expected to be two parts:
### twitter_data.npy 
This file contains users' data shaped N * L * D, where N denotes the number of users, L denotes the sequence length and D denotes the number of emotion categories at each time stamp. The missing data should be marked as -1 (or manually marked).
### twitter_network.pkl 
This file contains the social network information formated as the adjacent list:
~~~
[[node0 's neighbors], [node1's neighbors],..., nodeN's neighbors]
~~~
*Note:* each user index is corresponding to the index of the row in the data array in twitter_data.npy.
### Test
~~~
python impute.py -f data/data.npy -n data/network.pkl -o data/imputed_data.npy
~~~
### Output Format
The program outputs to a file named predict_twitter.npy which contains the data after prediction.
