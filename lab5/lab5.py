import random

"""
Adapted from https://github.com/CalciferZh/AMCParser/blob/master/amc_parser.py
"""
def read_line(stream, idx):
    if idx >= len(stream):
        return None, idx
    line = stream[idx].strip().split()
    idx += 1
    return line, idx

def parse_amc(file_path):
    with open(file_path) as f:
        content = f.read().splitlines()

    for idx, line in enumerate(content):
        if line == ':DEGREES':
            content = content[idx+1:]
            break

    frames = []
    idx = 0
    line, idx = read_line(content, idx)
    assert line[0].isnumeric(), line
    EOF = False
    while not EOF:
        joint_degree = {}
        while True:
            line, idx = read_line(content, idx)
            if line is None:
                EOF = True
                break
            if line[0].isnumeric():
                break
            joint_degree[line[0]] = [float(deg) for deg in line[1:]]
        frames.append(joint_degree)
    return frames

def concatMoCap(input_filenames, n_transition_frames, out_filename):
    '''
    concatenate the input MoCap files in random order, 
    generate n_transition_frames transition frames using interpolation between every two MoCap files, 
    save the result as out_filename.
      parameter:
        input_filenames: [str], a list of all input filename strings
        n_transition_frames: int, number of transition frames
        out_filename: output file name
      return:
        None
    '''
    #pass

    # change the order of the list into random order
    random.shuffle(input_filenames)
    # read the first three lines (common line) of the amc file
    with open(input_filenames[0]) as f:
        content = f.read().splitlines()
    for idx, line in enumerate(content):
        if line == ':DEGREES':
            content = content[:idx+1]
            break

    # write the first three lines (common line) to the output file
    output_file = open(out_filename, 'w')
    for i in range(len(content)):
        output_file.write(content[i] + '\n')

    # initial the motion number
    number = 0

    # calculate the number of input files
    number_of_input_files = len(input_filenames)

    # iterate over all input files
    for z in range(number_of_input_files):
        # get all datas of the corresponding input file
        input_file = parse_amc('./' + input_filenames[z])
        current_file_length = len(input_file)

        # for every motion of the corresponding input file
        for i in range(current_file_length):
            # write motion number to the output file
            number = number + 1
            output_file.write(str(number)+'\n')

            # write all keys and values of a motion to the output file
            for keys,values in input_file[i].items():
                output_file.write(keys + ' ')
                for j in range(len(values)):
                    output_file.write(str(values[j]) + ' ')
                output_file.write('\n')
        
        # calculate transition frames
        if (z + 1) != number_of_input_files:
            # get the last motion of current file
            last_motion_of_current_file = input_file[current_file_length - 1]
            next_input_file = parse_amc('./' + input_filenames[z + 1])
            # get the first motion of next file
            first_motion_of_next_file = next_input_file[0]
            # initial temp_dict
            temp_dict = {}

            # add keys and corresponding values to temp_dict
            for keys,values in first_motion_of_next_file.items():
                temp_dict[keys] = values

            for keys,values in last_motion_of_current_file.items():
                # https://blog.csdn.net/JoeBlackzqq/article/details/7385333?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522164532339616780357297524%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=164532339616780357297524&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-8-7385333.first_rank_v2_pc_rank_v29&utm_term=python+两个list各个元素相减&spm=1018.2226.3001.4187
                # get the difference between the data for last motion of current file and the data for first motion of next file
                temp_dict[keys] = list(map(lambda x: x[0]-x[1], zip(temp_dict[keys], values)))
                # divide the difference into the number of n_transition_frames
                for num in range(len(temp_dict[keys])):
                    temp_dict[keys][num] = (1 / (n_transition_frames + 1)) * temp_dict[keys][num]
            
            # create n transition_frames
            for i in range(n_transition_frames):
                number = number + 1
                output_file.write(str(number)+'\n')
                # write all datas of a transition frame to the output file
                for keys,values in last_motion_of_current_file.items():
                    output_file.write(keys + ' ')
                    temp_value = temp_dict[keys]
                    for j in range(len(values)):
                        # every data of transition_frames is equal to the data in last motion + (i + 1) * corresponding data in temp_value
                        output_file.write(str(values[j] + (i + 1) * temp_value[j]) + ' ')
                    output_file.write('\n')
    


def main():
    # initialize three amc files (list ['18_01.amc', '18_03.amc', '18_13.amc']) as input_filenames
    input_filenames = ['18_01.amc', '18_03.amc', '18_13.amc']
    # initialize 500 as n_transition_frames
    n_transition_frames = 500
    # initialize '18_concatenate.amc' as out_filename
    out_filename = 'combined.amc'
    # run concatMoCap
    concatMoCap(input_filenames, n_transition_frames, out_filename)


if __name__ == "__main__":
    main()