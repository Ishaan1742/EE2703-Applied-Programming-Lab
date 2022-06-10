"""
        EE2703 Applied Programming Lab - 2022
        Assignment 1: Sample solution
        Done by: Ishaan Agarwal
        Roll Number: EE20B046
        Date: 25th January, 2022
"""

from sys import argv, exit

"""
It's recommended to use constant variables than hard-coding them everywhere.
For example, if you decide to change the command from '.circuit' to '.start' later,
    you only need to change the constant
"""
CIRCUIT = '.circuit'
END = '.end'
"""
It's a good practice to check if the user has given required and only the required inputs
Otherwise, show them the expected usage.
"""
if len(argv) != 2:
    print(f"\nUsage: {argv[0]} <inputfile>")
    exit()

"""
The user might input a wrong file name by mistake.
In this case, the open function will throw an IOError.
Make sure you have taken care of it using try-catch
"""

class circuit_elements:
    def __init__(self, imp_part):
        self.type = imp_part[0][0]
        self.node1 = imp_part[1]
        self.node2 = imp_part[2]
        self.value = imp_part[-1]
        if(len(imp_part) == 5):
            self.V = imp_part[3]
        elif(len(imp_part) == 6):
            self.node3 = imp_part[3]
            self.node4 = imp_part[4]
    def print(self):
        print(f"Type: {self.type}")
        print(f"Node1: {self.node1}")
        print(f"Node2: {self.node2}")
        print(f"Value: {self.value}")


try:
    with open(argv[1]) as f:
        lines = f.readlines() #read file line by line
        start = -1
        end = -2
        for line in lines:  # extracting circuit definition start and end lines
            if CIRCUIT == line[:len(CIRCUIT)]:
                start = lines.index(line)
            elif END == line[:len(END)]:
                end = lines.index(line)
                break
        if start >= end or start==-1:  # validating circuit block #did not consider start=-1 case
            print('Invalid circuit definition')
            exit(0)

        circuit_def = lines[start+1:end] #part where circuit data is stored

        tokens = [] #list of elements used to store tokens of each line

        for line in reversed(circuit_def):
            imp_part = line.split('#')[0].split() #removes the comments
            tokens.append(circuit_elements(imp_part)) 
            print(' '.join(reversed(imp_part))) #converts reversed list to string separating each token by spaces
        
        print("\nCircuit Elements: ")
        for item in tokens:
            print('')
            item.print()

except IOError:
    print('Invalid file')
    exit() 

