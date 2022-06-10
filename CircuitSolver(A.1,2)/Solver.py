"""
        EE2703 Applied Programming Lab - 2022
        Assignment 2: Modified Nodal Analysis
        Done by: Ishaan Agarwal
        Roll Number: EE20B046
        Date: 7th February, 2022
"""

from sys import argv, exit
import numpy as np
import math
import cmath

#np.set_printoptions(precision =2)

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
So we use try-catch clause to handle it.
"""


class circuit_elements:

    def __init__(self, imp_part):
        self.name = imp_part[0]
        self.type = imp_part[0][0]
        self.node1 = int(imp_part[1][1:]) if imp_part[
            1] != "GND" else 0  # GND is a special case, all other nodes are named ni, i= 0,1,2,3...
        self.node2 = int(imp_part[2][1:]) if imp_part[
            2] != "GND" else 0  # GND is a special case, all other nodes are named ni, i= 0,1,2,3...
        self.value = float(imp_part[-1])
        if (self.type == 'F' or self.type == 'H'):
            self.V = imp_part[3]
        elif (self.type == 'E' or self.type == 'G'):
            self.node3 = int(imp_part[3][1:]) if imp_part[
                3] != "GND" else 0  # GND is a special case, all other nodes are named ni, i= 0,1,2,3...
            self.node4 = int(imp_part[4][1:]) if imp_part[
                4] != "GND" else 0  # GND is a special case, all other nodes are named ni, i= 0,1,2,3...

    def print(self):
        print(f"Type: {self.type}")
        print(f"Node1: {self.node1}")
        print(f"Node2: {self.node2}")
        print(f"Value: {self.value}")


class circuit_elements_ac:

    def __init__(self, imp_part, frequency):
        self.name = imp_part[0]
        self.type = imp_part[0][0]
        self.node1 = int(imp_part[1][1:]) if imp_part[
            1] != "GND" else 0  # GND is a special case, all other nodes are named ni, i= 0,1,2,3...
        self.node2 = int(imp_part[2][1:]) if imp_part[
            2] != "GND" else 0  # GND is a special case, all other nodes are named ni, i= 0,1,2,3...
        if self.type == 'R' or self.type == 'E' or self.type == 'G' or self.type == 'H' or self.type == 'F':
            self.value = float(imp_part[-1])
        elif self.type == 'L':
            self.value = complex(0,
                                 float(imp_part[-1]) * frequency * 2 *
                                 math.pi)  # sL = jwL
        elif self.type == 'C':
            self.value = complex(0, -1 / (float(imp_part[-1]) * frequency * 2 *
                                          math.pi))  # -1/sC = 1/jwC
        else:
            self.amplitude = float(
                imp_part[-2]
            ) / 2  # Amplitude of the sine wave since given is Vp-p
            self.phase = float(imp_part[-1])
            self.value = complex(math.cos(np.radians(self.phase)),
                                 math.sin(np.radians(
                                     self.phase))) * self.amplitude
        if (self.type == 'F' or self.type == 'H'):
            self.V = imp_part[3]
        elif (self.type == 'E' or self.type == 'G'):
            self.node3 = int(imp_part[3][1:]) if imp_part[3] != "GND" else 0
            self.node4 = int(imp_part[4][1:]) if imp_part[4] != "GND" else 0

    def print(self):
        print(f"Type: {self.type}")
        print(f"Node1: {self.node1}")
        print(f"Node2: {self.node2}")
        print(f"Value: {self.value}")
        print(f"frequency: {frequency}")


try:
    with open(argv[1]) as f:
        lines = f.readlines()  #read file line by line
        start = -1
        end = -2
        for line in lines:  # extracting circuit definition start and end lines
            if CIRCUIT == line[:len(CIRCUIT)]:
                start = lines.index(line)
            elif END == line[:len(END)]:
                end = lines.index(line)
                break
        if start >= end or start == -1:  # validating circuit block #did not consider start=-1 case
            print('Invalid circuit definition')
            exit(0)

        circuit_def = lines[start + 1:end]  #part where circuit data is stored

        tokens = []  #list of elements used to store tokens of each line

        dc = False  #flag to check if dc analysis is required

        if (end + 1 <= len(lines) - 1):
            if (lines[end + 1].startswith('.ac')):
                frequency = float(lines[end + 1].split()[-1])
                for line in reversed(circuit_def):
                    imp_part = line.split(
                        '#')[0].split()  #removes the comments
                    tokens.append(circuit_elements_ac(imp_part, frequency))
            else:
                dc = True
                for line in reversed(circuit_def):
                    imp_part = line.split(
                        '#')[0].split()  #removes the comments
                    tokens.append(circuit_elements(imp_part))

        else:
            dc = True
            for line in reversed(circuit_def):
                imp_part = line.split('#')[0].split()  #removes the comments
                tokens.append(circuit_elements(imp_part))
                #print(' '.join(reversed(imp_part))) #converts reversed list to string separating each token by spaces
        """print("\nCircuit Elements: ")
        for item in tokens:
            print('')
            item.print()"""

        #to create MNA matrices G (conductance matrix), V(variable vector) and I(vector of independent sources)

        tokens.reverse(
        )  #reversing the list to get the correct order of elements

        number_of_nodes = 0

        for element in tokens:
            number_of_nodes = max(
                number_of_nodes, element.node1,
                element.node2)  #to find the number of nodes in the circuit
        number_of_nodes += 1  #to account for the ground node

        dimension = number_of_nodes

        for element in tokens:
            if (element.type == 'V' or element.type == 'E'
                    or element.type == 'H'):
                dimension += 1
        dimension += 1  #to account v0 = 0

        if (dc == False):
            G = np.zeros((dimension, dimension),
                         dtype='complex_')  #conductance matrix
            V = np.zeros(dimension, dtype='complex_')  #variable vector
            I = np.zeros(dimension,
                         dtype='complex_')  #independent sources vector
        else:
            G = np.zeros((dimension, dimension), dtype='float_')
            V = np.zeros(dimension, dtype='float_')
            I = np.zeros(dimension, dtype='float_')

        #using all tokens to create G matrix and I matrix

        i = 0

        G[-1][0] = 1  #v0 = 0
        G[0][-1] = 1

        dict = {}

        for element in tokens:  #writing stamps of all the tokens as per MNA concepts
            if element.type == 'R' or element.type == 'L' or element.type == 'C':
                G[element.node1][element.node1] += 1 / (element.value)
                G[element.node1][element.node2] += -1 / (element.value)
                G[element.node2][element.node1] += -1 / (element.value)
                G[element.node2][element.node2] += 1 / (element.value)

            elif element.type == 'V':
                if element.name not in dict:
                    G[element.node1][number_of_nodes + i] += 1
                    G[element.node2][number_of_nodes + i] += -1
                    G[number_of_nodes + i][element.node1] += 1
                    G[number_of_nodes + i][element.node2] += -1
                    I[number_of_nodes + i] += (element.value)
                    dict[element.name] = number_of_nodes + i
                    i += 1
                else:
                    G[element.node1][dict[element.name]] += 1
                    G[element.node2][dict[element.name]] += -1
                    G[dict[element.name]][element.node1] += 1
                    G[dict[element.name]][element.node2] += -1
                    I[dict[element.name]] += (element.value)

            elif element.type == 'I':
                I[element.node1] += -(element.value)
                I[element.node2] += +(element.value)

            elif element.type == 'E':
                if element.name not in dict:
                    G[element.node1][number_of_nodes + i] += 1
                    G[element.node2][number_of_nodes + i] += -1
                    G[number_of_nodes + i][element.node1] += 1
                    G[number_of_nodes + i][element.node2] += -1
                    G[number_of_nodes + i][element.node3] += -(element.value)
                    G[number_of_nodes + i][element.node4] += +(element.value)
                    dict[element.name] = number_of_nodes + i
                    i += 1
                else:
                    G[element.node1][dict[element.name]] += 1
                    G[element.node2][dict[element.name]] += -1
                    G[dict[element.name]][element.node1] += 1
                    G[dict[element.name]][element.node2] += -1
                    G[dict[element.name]][element.node3] += -(element.value)
                    G[dict[element.name]][element.node4] += +(element.value)

            elif element.type == 'G':
                G[element.node1][element.node3] += (element.value)
                G[element.node1][element.node4] += -(element.value)
                G[element.node2][element.node3] += -(element.value)
                G[element.node2][element.node4] += (element.value)

            elif element.type == 'F':
                for object in tokens:
                    if (object.name == element.V):
                        break
                if object.name not in dict:
                    G[element.node1][number_of_nodes + i] += (element.value)
                    G[element.node2][number_of_nodes + i] += -(element.value)
                    dict[object.name] = number_of_nodes + i
                    i += 1
                else:
                    G[element.node1][dict[object.name] + 1] += (element.value)
                    G[element.node2][dict[object.name] + 1] += -(element.value)

            elif element.type == 'H':
                for object in tokens:
                    if (object.name == element.V):
                        break
                if object.name not in dict:
                    G[element.node1][number_of_nodes + i + 1] += 1
                    G[element.node2][number_of_nodes + i + 1] += -1
                    G[number_of_nodes + i + 1][element.node1] += 1
                    G[number_of_nodes + i + 1][element.node2] += -1
                    G[number_of_nodes + i + 1][number_of_nodes +
                                               i] += -(element.value)
                    dict[object.name] = number_of_nodes + i
                    dict[element.name] = number_of_nodes + i + 1
                    i += 2
                else:
                    G[element.node1][number_of_nodes + i] += 1
                    G[element.node2][number_of_nodes + i] += -1
                    G[number_of_nodes + i][element.node1] += 1
                    G[number_of_nodes + i][element.node2] += -1
                    G[number_of_nodes + i][dict[element.V]] += -(element.value)
                    dict[element.name] = number_of_nodes + i
                    i += 1
            '''element.print()
            print("\n")
            print(G)
            print("\n")
            print(I)'''

        try:
            V = np.matmul(np.linalg.inv(G), I)  #solving for V
            #V=np.linalg.solve(G,I) #alternative method
        except np.linalg.LinAlgError:
            print(
                "The conduction matrix cannot be inverted because it is singular. Enter a valid circuit definition"
            )
            exit()

        print("\nVariable Matrix: ")
        V[1:number_of_nodes] = V[1:number_of_nodes] - V[
            0]  # subtracting the ground node potential
        V[0] = 0
        print(V[:-1])

        print("\n")

        if dc == False:
            for i in range(number_of_nodes):
                print(
                    f"Voltage at node {i} is Amplitude: {np.round( cmath.polar(V[i])[0],4 ):15.4f} \t\t Phase in degrees: {np.round( np.degrees(cmath.polar(V[i])[1]),4 ):15.4f}"
                )
            for i in range(len(dict)):
                print(
                    f"Current through {list(dict.keys())[i]} is Amplitude: {np.round( cmath.polar(V[dict[list(dict.keys())[i]]])[0], 4 ):14.4f} \t\t Phase in degrees: {np.round( np.degrees(cmath.polar(V[dict[list(dict.keys())[i]]])[1]), 4 ):15.4f}"
                )
        else:
            for i in range(number_of_nodes):
                print(f"Voltage at node {i} is: {np.round( (V[i]),4 ):15.4f}")
            for i in range(len(dict)):
                print(
                    f"Current through {list(dict.keys())[i]} is: {np.round( (V[dict[list(dict.keys())[i]]]), 4 ):14.4f}"
                )

except IOError:
    print(
        'Invalid file. Please make sure that the circuit definition block is well defined and all component value are in scientific notation.'
    )
    exit()
