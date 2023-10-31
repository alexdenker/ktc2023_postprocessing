


import argparse 


parser = argparse.ArgumentParser(description='reconstruction using conditional diffusion')
parser.add_argument('input_folder')
parser.add_argument('output_folder')
parser.add_argument('level')

def coordinator(args):
    
    print("Input folder: ", args.input_folder)
    print("Output folder: ", args.output_folder)
    print("Level: ", args.level)

    ### read files from args.input_folder 
    # there will be ref.mat in the input_folder, dont process this 

    ### do cool stuff

    ### save reconstructions to args.output_folder 
    # as a .mat file containing a 256x256 pixel array with the name {file_idx}.mat 
    # the pixel array must be named "reconstruction" and is only allowed to have 0, 1 or 2 as values.


if __name__ == '__main__':
    args = parser.parse_args()
    coordinator(args)
