import re
import serial
import time
import numpy as np
from . import motor_cmd as lssc


class RealMotor:
    """Motor control class for physical Lynx Smart Servo motors."""
    
    def __init__(self, id=0, bus=None, bus_lock=None):
        self._id = id
        self._serial_bus = bus
        self._bus_lock = bus_lock  # Placeholder for bus lock if needed in future
        self.imu = np.zeros(6)  # Placeholder for IMU data storage

    def genericWrite(self, cmd, param=None):
        if self._serial_bus is None:
            return False
        if param is None:
            self._serial_bus.write((lssc.LSS_CommandStart + str(self._id) + cmd + lssc.LSS_CommandEnd).encode())
        else:
            self._serial_bus.write((lssc.LSS_CommandStart + str(self._id) + cmd + str(param) + lssc.LSS_CommandEnd).encode())
        return True

    def _read_and_parse_packet(self, cmd, numChars=None):
        """
        Helper function to read and parse a packet from the serial bus,
        discarding packets not intended for this motor or command.
        Returns (readID, readIdent, readValue) or None if no valid packet is found within timeout.
        """
        start_time = time.time()
        while True:
            # Read until the start of a reply packet or timeout
            c = self._serial_bus.read()
            if (c.decode("utf-8") != lssc.LSS_CommandReplyStart):
                if (time.time() - start_time) > self._serial_bus.timeout:
                    # Timeout before even finding a start character
                    return None
                continue # Keep reading until start character

            # Read until the end of the packet
            data = self._serial_bus.read_until(lssc.LSS_CommandEnd.encode('utf-8'))

            # If numChars is provided, remove the last character (LSS_CommandEnd)
            if numChars is not None:
                data = data[:-1]

            if not data:
                # No data received after start character, possibly a timeout
                return None

            try:
                decoded_data = data.decode("utf-8")
            except UnicodeDecodeError:
                # Failed to decode, discard and try again
                continue

            # Determine regex based on whether numChars is provided
            if numChars is None:
                # For integer values, use the original regex
                matches = re.match("(\d{1,3})([A-Z]{1,4})(-?\d{1,18})", decoded_data, re.I)
            else:
                # For string values with a specific number of characters
                matches = re.match("(\d{1,3})([A-Z]{1,4})(.{" + str(numChars) + "})", decoded_data, re.I)

            if (matches is None) or \
               (matches.group(1) is None) or \
               (matches.group(2) is None) or \
               (matches.group(3) is None):
                # Malformed packet, discard and try again
                continue

            readID = matches.group(1)
            readIdent = matches.group(2)
            readValue = matches.group(3)

            # Check if the packet is for the expected motor ID and command
            if (readID == str(self._id) and readIdent == cmd):
                return readID, readIdent, readValue
            else:
                # Wrong ID or identifier, discard and try again
                continue

    def genericRead_Blocking_int(self, cmd):
        if self._serial_bus is None:
            print(f"[Lynx] Error: Motor {self._id}: Serial bus is not initialized for read command '{cmd}'.")
            return None
        try:
            result = self._read_and_parse_packet(cmd)
            if result is None:
                print(f"[Lynx] Error: Motor {self._id}: No valid response received for command '{cmd}' within timeout.")
                return None

            readID, readIdent, readValue = result

            # The _read_and_parse_packet already ensures ID and command match,
            # so these checks are now redundant but kept for clarity or future changes.
            if (readID != str(self._id)):
                print(f"[Lynx] Error: Motor {self._id}: Received packet for wrong motor ID (expected {self._id}, got {readID}) for command '{cmd}'.")
                return None
            if (readIdent != cmd):
                print(f"[Lynx] Error: Motor {self._id}: Received packet for wrong command identifier (expected '{cmd}', got '{readIdent}') in '{readValue}'.")
                return None

        except serial.SerialTimeoutException:
            print(f"[Lynx] Error: Motor {self._id}: Serial read timed out for command '{cmd}'. No response from motor.")
            return None
        except serial.SerialException as se:
            print(f"[Lynx] Error: Motor {self._id}: Serial communication error during read for command '{cmd}': {se}")
            return None
        except Exception as e:
            print(f"[Lynx] Error: Motor {self._id}: An unexpected error occurred while processing read for command '{cmd}': {e}")
            return None

        try:
            return int(readValue)
        except ValueError:
            print(f"[Lynx] Error: Motor {self._id}: Received value '{readValue}' for command '{cmd}' is not a valid integer.")
            return None

    def genericRead_Blocking_int_primitive_returns(self, cmd):
        if self._serial_bus is None:
            return None
        try:
            result = self._read_and_parse_packet(cmd)
            if result is None:
                return None
            
            readID, readIdent, readValue = result
            # No print statements for primitive returns, as per original function's intent
            if(readID != str(self._id)):
                return(None)
            if(readIdent != cmd):
                return(None)
        except:
            return(None)
        return int(readValue)
    
    def genericRead_Blocking_str(self, cmd, numChars):
        if self._serial_bus is None:
            return None
        try:
            result = self._read_and_parse_packet(cmd, numChars)
            if result is None:
                return None
            
            readID, readIdent, readValue = result
            # No print statements for primitive returns, as per original function's intent
            if(readID != str(self._id)):
                return(None)
            if(readIdent != cmd):
                return(None)
        except:
            return(None)
        return(readValue)
    
    def genericRead_Blocking_str_query_limit(self, cmd, numChars):
        if self._serial_bus is None:
            return None
        try:
            result = self._read_and_parse_packet(cmd, numChars)
            if result is None:
                print(f"[Lynx] Error: Motor {self._id}: No valid response received for command '{cmd}' within timeout.")
                return None

            readID, readIdent, readValue = result

            # The _read_and_parse_packet already ensures ID and command match,
            # so this check is now redundant but kept for clarity or future changes.
            if (readID != str(self._id)):
                print(f"[Lynx] Error: Motor {self._id}: Received packet for wrong motor ID (expected {self._id}, got {readID}) for command '{cmd}'.")
                return None
        except:
            return(None)
        return(readValue)
    
    # Soft reset - revert all commands stored in EEPROM
    # If used, motor will be busy for a short moment while resetting
    def reset(self):
        return (self.genericWrite(lssc.LSS_ActionReset))
    

    ### ACTIONS ###

    # This action causes the servo to go "limp". The microcontroller will still be powered, 
    # but the motor will not. As an emergency safety feature, 
    # should the robot not be doing what it is supposed to or risks damage, use the 
    # broadcast ID to set all servos limp #254L<cr>.
    def limp(self):
        with self._bus_lock:
            return (self.genericWrite(lssc.LSS_ActionLimp))
    
    # def emergency_stop(self):
    #     return (self.genericWrite(254, lssc.LSS_ActionLimp))
    
    # This command causes the servo to stop immediately and hold that angular position.
    def hold(self):
        with self._bus_lock:
            return (self.genericWrite(lssc.LSS_ActionHold))
    
    # Move to position in degrees!
    def move_abs(self, pos):
        with self._bus_lock:
            return (self.genericWrite(lssc.LSS_ActionMove, pos))
    
    def move_abs_with_speed(self, pos, speed):
        if self._serial_bus is None:
            return False
        with self._bus_lock:
            self._serial_bus.write((lssc.LSS_CommandStart + 
                                    str(self._id) + 
                                    lssc.LSS_ActionMove + 
                                    str(pos) + 
                                    lssc.LSS_ActionMaxSpeed + 
                                    str(speed) +
                                    lssc.LSS_CommandEnd).encode())
            return True
    
    # Move relative position in degrees
    def moveRelative(self, delta):
        with self._bus_lock:
            return (self.genericWrite(lssc.LSS_ActionMoveRelative, delta))
        
    # Move relative position in degrees with speed
    def moveRelativeWithSpeed(self, delta, speed):
        if self._serial_bus is None:
            return False
        with self._bus_lock:
            self._serial_bus.write((lssc.LSS_CommandStart + 
                                    str(self._id) + 
                                    lssc.LSS_ActionMoveRelative + 
                                    str(delta) + 
                                    lssc.LSS_ActionMaxSpeed + 
                                    str(speed) +
                                    lssc.LSS_CommandEnd).encode())
            return True
    

    ### Status Commands ###
    """
    
    """
    def getPosition(self):
        with self._bus_lock:
            self.genericWrite(lssc.LSS_QueryPosition)
            return (self.genericRead_Blocking_int(lssc.LSS_QueryPosition))
        
    def getVelocity(self):
        with self._bus_lock:
            self.genericWrite(lssc.LSS_QuerySpeed)
            return (self.genericRead_Blocking_int(lssc.LSS_QuerySpeed))
        
    def getPosition_Without_Lock(self):
        self.genericWrite(lssc.LSS_QueryPosition)
        return (self.genericRead_Blocking_int(lssc.LSS_QueryPosition))
    
    def get_IMU(self, key):
        self.genericWrite(lssc.POR_QueryIMULinearAccelPrefix + key)
        return (self.genericRead_Blocking_int(lssc.POR_QueryIMULinearAccelPrefix + key))
    
    def get_IMU_all(self):
        for i, key in enumerate(["X", "Y", "Z", "A", "B", "G"]):
            self.genericWrite(lssc.POR_QueryIMULinearAccelPrefix + key)
            self.imu[i] = self.genericRead_Blocking_int(lssc.POR_QueryIMULinearAccelPrefix + key)
        return self.imu
    
    def get_IMU_partial(self, keys):
        imu_data = []
        for key in keys:
            self.genericWrite(lssc.POR_QueryIMULinearAccelPrefix + key)
            imu_value = self.genericRead_Blocking_int(lssc.POR_QueryIMULinearAccelPrefix + key)
            imu_data.append(imu_value)
        return np.array(imu_data)
    
    def getStatus(self):
        self.genericWrite(lssc.LSS_QueryLimitStatus)
        return (self.genericRead_Blocking_int(lssc.LSS_QueryStatus))