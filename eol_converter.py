def convert_line_endings(input_file, output_file):
    with open(input_file, 'r', newline='\r\n') as infile:
        content = infile.read()
    
    with open(output_file, 'w', newline='\n') as outfile:
        outfile.write(content)

input_files = ['playlist_mapper.py', 'playlist_reducer.py', 'enrich_mapper.py', 'enrich_reducer.py']

for input_file in input_files:
    output_file = input_file.replace('.py', '_unix.py')
    convert_line_endings(input_file, output_file)