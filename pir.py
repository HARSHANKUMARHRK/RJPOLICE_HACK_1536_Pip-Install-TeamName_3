from pyfirmata import Arduino, util
import time

# Define the Arduino board and port
board = Arduino("/dev/cu.usbmodem101")  # Replace 'COM3' with the appropriate port for your Arduino

# Define the digital pin for the PIR sensor
pir_pin = board.get_pin('d:2:i')  # Use the appropriate pin number based on your setup

# Create an iterator to continuously read data from the board
it = util.Iterator(board)
it.start()

try:
    while True:
        # Read the state of the PIR sensor
        pir_state = pir_pin.read()

        # Print the state of the PIR sensor
        print("PIR Sensor State:", pir_state)

        # Wait for a short duration before reading again
        time.sleep(0.1)

except KeyboardInterrupt:
    # Close the connection when the script is interrupted
    board.exit()
