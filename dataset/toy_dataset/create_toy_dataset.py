import random
import numpy as np

# item_set from 0 to 99
item_set = list(range(100))
# user_set from 0 to 99
user_set = list(range(100))
seq_len = 100

# write first line
# user_id:token	item_id:token	timestamp:float
with open("toy_data.txt", "w") as f:
    f.write("user_id:token\titem_id:token\ttimestamp:float\n")

for u in user_set:
    stamps = []
    while len(stamps) < seq_len:
        timestamp = random.randint(0, 10000)
        if timestamp in stamps:
            continue
        item = random.choice(item_set) # random item
        # item = timestamp % 99 
        # write
        with open("toy_data.txt", "a") as f:
            f.write(f"{u}\t{item}\t{timestamp}\n")
        stamps.append(timestamp)
            
# sort by user and timestamp
data = np.loadtxt("toy_data.txt", delimiter="\t", skiprows=1)
data = data[np.lexsort((data[:, 2], data[:, 0]))]
np.savetxt("toy_data.txt", data, delimiter="\t", fmt="%d\t%d\t%.2f")

