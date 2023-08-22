import numpy as np

arr = np.arange(0, 100, 1)


with open("testout.txt", 'w') as f:
    for i in range(len(arr)):
        f.write(str(arr[i]) + "\n")
    f.close()

print("File created")