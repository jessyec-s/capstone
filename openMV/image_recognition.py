# Single Color Code Tracking Example
#
# This example shows off single color code tracking using the OpenMV Cam.
#
# A color code is a blob composed of two or more colors. The example below will
# only track colored objects which have both the colors below in them.

import sensor, image, time, math
# from pyb import UART
from pyb import LED

# Distance Constants
lens_mm = 2.8 # standard lens
height_obj_mm = 30.0
image_height_pixels = 240
image_width_pixels = 340
sensor_w_mm = 3.984
sensor_h_mm = 2.952
offset_mm = 100.0 # offset fix

h_fov = 2 * math.atan((sensor_w_mm/2) / lens_mm)
v_fov = 2 * math.atan((sensor_h_mm/2) / lens_mm)
print("h_fov: ", h_fov, ", v_fov: ", v_fov)

#Distance func
def distance_to_obj(h): # h = blob.h()
    return ((lens_mm * height_obj_mm * image_height_pixels) / (h * sensor_h_mm))

blue_led  = LED(3)
green_led  = LED(2)
red_led  = LED(1)

# uart = UART(3, 115200, timeout_char = 1000)
blue_led.on()
# Color Tracking Thresholds (L Min, L Max, A Min, A Max, B Min, B Max)
# The below thresholds track in general red/green things. You may wish to tune them...
#thresholds = [(30, 100, 15, 127, 15, 127), # generic_red_thresholds -> index is 0 so code == (1 << 0)
#              (30, 100, -64, -8, -32, 32)] # generic_green_thresholds -> index is 1 so code == (1 << 1)

thresholds = [(55, 100,-24, 11, 32, 86),     #1#yellow
              (0, 39, -82, 127, 14, 127),     #2#red
              (39, 100,-51,-12, 10, 57)]      #4#green
# Codes are or'ed together when "merge=True" for "find_blobs".

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time = 2000)
sensor.set_auto_gain(False) # must be turned off for color tracking
sensor.set_auto_whitebal(False) # must be turned off for color tracking
clock = time.clock()

blue_led.off()
green_led.off()
red_led.off()

object_x_old = 0
object_y_old = 0

code = 2 ## 1:yellow   2:red    4:green
buf = "00"
# Only blobs that with more pixels than "pixel_threshold" and more area than "area_threshold" are
# returned by "find_blobs" below. Change "pixels_threshold" and "area_threshold" if you change the
# camera resolution. "merge=True" must be set to merge overlapping color blobs for color codes.

while(True):
    clock.tick()

    blue_led.off()
    green_led.off()
    red_led.off()

    img = sensor.snapshot()
    for blob in img.find_blobs(thresholds, pixels_threshold=100, area_threshold=100, merge=True):
#check which color should be detected
        if uart.any()>0 :
            buf=uart.read()
            print (buf[0])
            if buf[0]==ord('y') :
                code = 1
            if buf[0]==ord('r') :
                code = 2
            if buf[0]==ord('g') :
                code = 4

#check if there is object with right color
        if blob.code() == code:

            img.draw_rectangle(blob.rect())
            img.draw_cross(blob.cx(), blob.cy())

#make sure the detected object is stable and print the coordinates
#first it detect if the coordinates of blob is available
#second compared with the last position to make sure if the object is not moving
#third reduce the affect of anbience
            if blob.cx()!=None and (
                abs(object_x_old - int(blob.cx())) < 8 and
                abs(object_y_old - int(blob.cy())) < 8) and (
                blob.w()>35 and
                blob.h()>35):
               #just detect the objects. turn on the blue only
               blue_led.on()
               red_led.off()
               green_led.off()
               print("stable!")
               print("blob.cx, blob.cy, blob.w, blob.h are: ")
               print(blob.cx(), blob.cy(),blob.w(), blob.h())
               print("distance: ", distance_to_obj(blob.h()))
               #print (buf)

            object_x_old = int(blob.cx())
            object_y_old = int(blob.cy())
