from tqdm import tqdm
import time

start_epoch = 1
num_of_epochs = 10
pbar = tqdm(total=num_of_epochs, initial=start_epoch-1)
for epoch in range(start_epoch, start_epoch + num_of_epochs):
    time.sleep(1)
    pbar.update(1)
    print(epoch)
pbar.close()