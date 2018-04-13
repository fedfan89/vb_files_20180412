with open('new_names.csv', 'r') as f:
    print(f.tell())
    size_to_read = 5
    f_contents = f.read(size_to_read)
    print(f.tell())
    print(len(f_contents))
    print(type(f_contents))
    print(type(f))
    while len(f_contents) > 0:
        print(f_contents, end='*')
        f_contents = f.read(size_to_read)

    print(f.tell())
my_string = 'Hi, my name is Paul'
my_string_2 = "Aaron"
print(my_string, end='')
print(my_string_2)
