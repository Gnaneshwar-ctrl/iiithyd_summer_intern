from operator import le
import os
from stat import filemode
import numpy as np

doubleLine = "\n\n"

def read_lammps_data_positions(number):
    middle = str(number)
    filename = "./configurations_ljsmooth_10k/confdump."+middle+".data"
    cmd = 'tail -1000 '+filename+' > ./p1'
    #a1 = os.system('tail -1000 ${filename}')
    a1=os.system(cmd)
    #int(number)
    ret = np.loadtxt('./p1')[:,2:5]
    return ret


def read_lammps_data_velocity(number):
    middle = str(number)
    filename = "./configurations_ljsmooth_10k/confdump."+middle+".data"
    cmd = 'tail -1000 '+filename+' > ./p1'
    #a1 = os.system('tail -1000 ${filename}')
    a1=os.system(cmd)
    #int(number)
    ret = np.loadtxt('./p1')[:,5:8]
    return ret

increment = 0
while(True):

    data_positions = read_lammps_data_positions(increment)
    data_velocities = read_lammps_data_velocity(increment)

    #print(len(data_positions))
    #print(len(data_velocities))

    prereq = "#bin. KALJ data file T=0.5" + doubleLine + "1000 atoms" + '\n' + "2 atom types" + doubleLine + "0 9.4 xlo xhi" + '\n' + "0 9.4 ylo yhi" + '\n' + "0 9.4 zlo zhi" + doubleLine + "Masses" + doubleLine + "1 1.0" + '\n' + "2 1.0" + doubleLine + "Atoms" + doubleLine

    filename = "./modfied/modfied."+str(increment)+".data"
    os.system('touch ${filename}')
    filemod = open("./modfied/modfied."+str(increment)+".data", "a")
    filemod.write(prereq)
    filemod.close()


    t=1
    while(t<1001):
        newline = ""
        if(t<801):
            newline = str(t) + " " + "1" + " " + str(data_positions[t-1][0]) + " " + str(data_positions[t-1][1]) + " " + str(data_positions[t-1][2]) + '\n'
            filemod = open("./modfied/modfied."+str(increment)+".data", "a")
            filemod.write(newline)
            filemod.close()

        else:
            newline = str(t) + " " + "2" + " " + str(data_positions[t-1][0]) + " " + str(data_positions[t-1][1]) + " " + str(data_positions[t-1][2]) + '\n'
            filemod = open("./modfied/modfied."+str(increment)+".data", "a")
            filemod.write(newline)
            filemod.close()
        t = t + 1


    midreq = '\n' + 'Velocities' + doubleLine

    filemod = open("./modfied/modfied."+str(increment)+".data", "a")
    filemod.write(midreq)
    filemod.close()

    t=1
    while(t<1001):
        newline = ""
        if(t<801):
            newline = str(t) + " " + str(data_velocities[t-1][0]) + " " + str(data_velocities[t-1][1]) + " " + str(data_velocities[t-1][2]) + '\n'
            filemod = open("./modfied/modfied."+str(increment)+".data", "a")
            filemod.write(newline)
            filemod.close()

        else:
            newline = str(t) +  " " + str(data_velocities[t-1][0]) + " " + str(data_velocities[t-1][1]) + " " + str(data_velocities[t-1][2]) + '\n'
            filemod = open("./modfied/modfied."+str(increment)+".data", "a")
            filemod.write(newline)
            filemod.close()
        t = t + 1
    #os.system('mv ${filename} ./newdatafiles')
    if(increment>10000):
        break
    increment = increment + 1
