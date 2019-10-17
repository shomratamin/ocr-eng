from glob import glob
from random import randint, shuffle
import time



bengali_digits = ['০','১','২','৩','৪','৫','৬','৭','৮','৯']

def generate_random_numbers(no_of_lines):
    number_strings = []
    for _ in range(no_of_lines):
        random_number_length = randint(1,35)
        _number_string = ''
        _number_string_e = ''
        for __ in range(random_number_length):
            rand_index = randint(0,9)
            _number_string += bengali_digits[rand_index]
            _number_string_e += str(rand_index)
        _number_string += '\n'
        _number_string_e += '\n'
        number_strings.append(_number_string)
        number_strings.append(_number_string_e)

    save_text('train_data\\numbers.txt', number_strings)
    

def save_text(file_name, text_lines_list, add_newline = False):
    with open(file_name,'w', encoding='utf-8') as f:
        for line in text_lines_list:
            f.write(line)
            if add_newline:
                f.write('\n')
        f.close()
    print('wrote file : ', file_name)




def concatenate_text_files(folder):
    text_files = glob(folder+'*.txt')

    character_set = set()
    file_content = []
    line_counter = 0
    for text_file in text_files:
        print('reading file : ', text_file)
        file_object = open(text_file,'r', encoding='utf-8')
        lines = file_object.readlines()
        for line in lines:
            line = line.replace('\n','')
            line_segments = line.split('।')
            if len(line_segments) == 1 and len(line_segments[-1]) < 100:
                line_segments[-1] += '\n'
                file_content.append(line_segments[-1])
            else:
                for l in line_segments:
                    if len(l) > 0 and l.count(' ') != len(l) and len(l) < 100:

                        if line_counter % 5 == 0 and l[-1] not in bengali_digits:
                            if l[-1] != ' ':
                                l = l + ' '
                            l = l + '।\n'
                        else:
                            l = l + '\n'
                        if l[0] == ' ':
                            l = l[1:]
                        line_counter += 1
                        file_content.append(l)
    shuffle(file_content)
    for i in range(len(file_content)):
        line = file_content[i]
        space_count = 0
        non_space_fount = False
        for c in line:
            if c not in character_set:
                character_set.add(c)
            if c == ' ' and not non_space_fount:
                space_count += 1
            else:
                non_space_fount = True
        line = line[space_count:]
        file_content[i] = line

    save_text('train_data\\out.txt', file_content)
    save_text('train_data\\characters.txt',character_set,True)
    





if __name__ == '__main__':
    start = time.time()
    concatenate_text_files('raw_data\\')
    # generate_random_numbers(10000)
    end = time.time()
    total = end - start
    print('total time taken: ', total, ' seconds')