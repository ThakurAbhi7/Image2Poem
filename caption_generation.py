import torch
import os
import argparse
import json

def caption_fn(path):
    print(path)
    cmd = "python3 predict.py --path " + path + " > caption.txt"
    os.system(cmd)
    f = open("caption.txt", "r")
    caption = f.read()
    cmd = "rm caption.txt"
    os.system(cmd)
    return caption

def main():
    parser = argparse.ArgumentParser(description='Caption Generator')
    parser.add_argument('--path', type=str, help='path to image folder', required=True)
    parser.add_argument('--caption_path', type=str, default='data/multi_cap.json', help='path to gpt2 checkpoint for peom')
    args = parser.parse_args()
    multi = {}
    for image in os.listdir(args.path):
        if ".jpg" not in image:
            continue
        caption = caption_fn(args.path+image)
        multi[image.split('.')[0]] = caption

    out_file = open(args.caption_path, "w")
    json.dump(multi, out_file)
    out_file.close()
    

if __name__ == '__main__':
    main()