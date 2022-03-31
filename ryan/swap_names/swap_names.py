# must pip install names
import names
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_data", help = "path to data whose names should be swapped at random," \
                        "must contain the <PERSON> token and must be a tsv")
    parser.add_argument("output_path", help = "path to output the new data")
    args = parser.parse_args()

    with open(args.input_data) as inp:
        lines = inp.readlines()

    # lines 419 and 407
    sents = []
    for l in lines:
        sents.append(l.split(" "))
    
    for s in sents:
        for i in range(len(s)):
            if s[i] == "<PERSON>" and s[i+1] == "<PERSON>":
                s[i] = names.get_first_name().lower()
                s[i+1] = names.get_last_name().lower()
                i+=1
            elif s[i] == "<PERSON>":
                s[i] = names.get_first_name().lower()
    
    data = []
    for s in sents:
        data.append(" ".join(s))

    with open(args.output_path, 'w') as f:
        for d in data:
            f.write(d)

if __name__ == "__main__":
    main()


