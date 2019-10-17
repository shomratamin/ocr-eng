max_line_length = 0
with open('train_data/out.txt','r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        _len = len(line)
        if _len > max_line_length:
            max_line_length = _len

print('max length of line : ', max_line_length)