#!/usr/bin/env python
import sys
import csv
import os
import glob

assert len(sys.argv) == 2
dir_name = sys.argv[1]

def generate_line(row_name):
    # put tsplib name
    content = [ row_name ]
    os.chdir("./"+row_name)
    csv_files = glob.glob("*.csv")
    # sort solvers first mult weight, fekete, hp, cc_lp
    csv_files = sort_csv_files(csv_files)
    # line size
    line_no = get_line_size(csv_files[0])
    content.append(line_no)

    if glob.glob("*_all.csv"):
        all_lines = True
    elif glob.glob("*_random.csv"):
        all_lines = False
    else:
        raise StandardError('not supported line type')
    # put lines type
    if all_lines:
        content.append("all")
    else:
        content.append("random")

    # put solver data into content
    for csv_solver in csv_files:
       solver_data = solver_content(csv_solver)
       content.extend(solver_data)
    return content

def sort_csv_files(csv_files):
    sorted_files = range(4)
    for csv_file in csv_files:
        if csv_file.startswith("mult_weight"):
            sorted_files[0] = csv_file
        elif csv_file.startswith("fekete_lp"):
            sorted_files[1] = csv_file
        elif csv_file.startswith("hp_lp"):
            sorted_files[2] = csv_file
        elif csv_file.startswith("cc_lp"):
            sorted_files[3] = csv_file
    return sorted_files

def solver_content(csv_file):
    with open(csv_file, 'rb') as csvfile:
        csv_reader = csv.reader(csvfile)
        csv_reader.next()
        data_row = csv_reader.next()
        # return crossing number, avg crossing number and CPU time
        return (data_row[4], data_row[5], data_row[2])

def get_line_size(csv_filename):
    with open(csv_filename, 'rb') as csvfile:
        csv_reader = csv.reader(csvfile)
        csv_reader.next()
        data_row = csv_reader.next()
        return data_row[1]


file_name = '../tsplib_results.csv'
def write_data(line):
    with open(file_name, 'ab') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(line)

def main():
    data_row = generate_line(dir_name)
    write_data(data_row)

if __name__ == '__main__':
    main()
