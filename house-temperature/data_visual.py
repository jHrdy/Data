# artiffical used to datasets test data visualization script 
# that may be used in further IoT projects (with RPi or ESP32)

import matplotlib.pyplot as plt
import numpy as np
from create_data import data

# load artiffical data
temp_kitchen, temp_room, temp_hall = data()

# pushing data into list
def create_continous_data(temperature):
    out_data = []
    for day in temperature:
        for data in day:
            out_data.append(data)
    return out_data        

# data by place
kitchen_continous = create_continous_data(temp_kitchen)
room_continous = create_continous_data(temp_room)
hall_continous = create_continous_data(temp_hall)
hours = [i for i in range(168)]

# main function using fixed step length to display "realtime" data into column graph
def animate_temp_changes():
    def calc_step(container, i):
            return -(container[i] - container[i+1])/10
    cnt = 0
    step = 0.003

    values = kitchen_continous[cnt], room_continous[cnt], hall_continous[cnt] 
    k_speed = -(kitchen_continous[0]-kitchen_continous[1])
    r_speed = -(room_continous[0]-room_continous[1])
    h_speed = -(hall_continous[0]-hall_continous[1])

    r_cnt = 0
    h_cnt = 0
    while cnt != len(kitchen_continous)-1:
        cols = ('Kitchen', 'Room', 'Hall')
        plt.bar(cols,values)
        plt.pause(0.001)
        plt.clf()
        
        if abs(values[0]-kitchen_continous[cnt+1]) < step:
            cnt += 1
            k_speed = calc_step(kitchen_continous, cnt)

        if abs(values[1]-room_continous[r_cnt+1]) < step:
            r_cnt += 1
            r_speed = calc_step(room_continous, r_cnt)

        if abs(values[2]-hall_continous[h_cnt+1]) < step:
            h_cnt += 1
            h_speed = calc_step(hall_continous, h_cnt)

        values = (values[0]+k_speed, values[1]+r_speed, values[2]+h_speed)

if __name__ == "__main__":
    # time series plot
    plt.plot(kitchen_continous, label='Kitchen temp')
    plt.plot(room_continous, label='Room temp')
    plt.plot(hall_continous, label='Hall temp')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time [hrs]')
    plt.ylabel('Temperature [C]')
    plt.show()
    
    # animated columns 
    animate_temp_changes()
    plt.show()