import argparse

def main():
    parser = argparse.ArgumentParser()
    # the last three args are required to set the default value
    parser.add_argument("masked_data", help = "path to data with <PERSON> token present",
                        type=str, nargs = "?", const=1, 
                        default = "../normalized/GLUE-SST-2/GLUE-SST-2-entity_only_high-3.01/train.tsv")
    parser.add_argument("unmasked_data", help = "path to data without <PERSON> token present",
                        type=str, nargs = "?", const=1, 
                        default = "../original/GLUE-SST-2/train.tsv")
    parser.add_argument("output_dir", help = "directory to output final data",
                        type=str, nargs = "?", const=1, 
                        default = "./original_names/GLUE-SST-2/GLUE-SST-2-entity_only_high-3.01")
    args = parser.parse_args()

    with open(args.masked_data, 'r') as inp:
        lines = inp.readlines()
    
    with open(args.unmasked_data, 'r') as inp:
        og_data = inp.readlines()
    
    idx_to_keep = []
    for i,l in enumerate(lines):
        if "<PERSON>" in l:
            idx_to_keep.append(i)
    idx_to_keep.insert(0,0)
    
    data = [og_data[i] for i in idx_to_keep]
    masked_data = [lines[i] for i in idx_to_keep]

    with open(args.output_dir + '/train.tsv', 'w') as f:
        for d in data:
            f.write(d)
    
    with open(args.output_dir + '/masked_data_subset.tsv', 'w') as f:
        for d in masked_data:
            f.write(d)




if __name__ == "__main__":
    main()