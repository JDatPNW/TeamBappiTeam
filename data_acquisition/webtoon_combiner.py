import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

# parsing
parser = argparse.ArgumentParser()
parser.add_argument('--filepath', default='./Webtoon-Downloader/src/Mage_Again', help='Directory where imgs are saved by using Webtoon-Downloader')
parser.add_argument('--outputpath', default='./Webtoon-Downloader/src/Mage_Again', help='Directory where to save the combine imgs')
parser.add_argument('--start', default=1, help='Episode to start combine')
parser.add_argument('--end', default=73, help='Episode to end combine')
parser.add_argument('--all', default=True, help='Whether to combine all')
parser.add_argument('--combine', default=5, help='Unit to combine imgs with')
parser.add_argument('--remove', default=True, help='Whether to remove the original imgs')
args = parser.parse_args()


print(f'File path : {args.filepath}')
print(f'Start combine from {args.start} to {args.end}', end='\n')

for idx in range(args.start, args.end):
    # make final path
    try:
        if idx < 10:
            path = f'{args.filepath}/0{idx}/'
            outputpath = f'{args.outputpath}/0{idx}/'
        else:
            path = f'{args.filepath}/{idx}/'
            outputpath = f'{args.outputpath}/{idx}/'

        # make img path list
        img_lst = os.listdir(path)
    except:
        path = f'{args.filepath}/{idx}/'
        outputpath = f'{args.outputpath}/{idx}/'

        # make img path list
        img_lst = os.listdir(path)

    # sort
    img_lst.sort(key= lambda x: int(x.split('_')[1].split('.')[0]))
    
    print('='*20, f'Episode : {idx}', '='*20)
    if args.all:
        # make combine imgs(webtoon)
        for i, img in tqdm(enumerate(img_lst)):
            try:
                if i == 0:
                    final = np.concatenate((np.array(Image.open(path + img)), np.array(Image.open(path + img_lst[i+1]))))
                else:
                    final = np.concatenate((final, np.array(Image.open(path + img_lst[i+1]))))
            except IndexError as e:
                print(e)

        # save combine img
        Image.fromarray(final).save(outputpath + f'webtoon_{idx}.png', 'png')

    else:
        # make combine imgs(webtoon)
        for i, img in tqdm(enumerate(img_lst)):
            try:
                if i == 0:
                    final = np.concatenate((np.array(Image.open(path + img)), np.array(Image.open(path + img_lst[i+1]))))
                # save combine img per args.combine
                elif (i+1) % args.combine ==0:
                    Image.fromarray(final).save(outputpath + f'webtoon_{i}.png', 'png')
                    final = np.array(Image.open(path + img_lst[i+1]))
                else:
                    final = np.concatenate((final, np.array(Image.open(path + img_lst[i+1]))))
            except IndexError as e:
                print(e)

    # remove original img
    if args.remove:
        for img in tqdm(img_lst):
            os.remove(path + img)

print('Done!')
