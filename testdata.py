import csv
import numpy as np

filename = 'actuator_data_dummy.csv'
num_rows = 1000  # nData

Kp = 2.0
Kd = 0.3
tau_ff = 0.05

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['time', 'pos_ref', 'pos_actual', 'velocity', 'Kp', 'Kd', 'tau_ff', 'error', 'torque'])

    pos_actual = 0.0
    velocity = 0.0

    for i in range(num_rows):
        time = i * 0.01  # sampling rate 100Hz
        pos_ref = 1.0 + 0.1 * np.sin(0.1 * i)  
        error = pos_ref - pos_actual
        
        velocity = 0.8 * velocity + 0.1 * error 
        torque = Kp * error + Kd * velocity + tau_ff + np.random.normal(0, 0.01) 

        pos_actual += velocity * 0.01  # delta pos = v * delta t
        
        writer.writerow([f"{time:.2f}", f"{pos_ref:.3f}", f"{pos_actual:.3f}", f"{velocity:.3f}", f"{Kp}", f"{Kd}", f"{tau_ff}", f"{error:.3f}", f"{torque:.3f}"])

print(f"Dummy actuator data generated in '{filename}' with {num_rows} rows.")
