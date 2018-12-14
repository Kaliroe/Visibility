

"""Calculations using the Planet_info.csv arrays which were created in
Planets_astropy.py"""


import numpy as np
import matplotlib.pyplot as plt
import datetime
import astropy
import astropy.units as u
import pandas as pd
from astropy.time import Time
from astropy.coordinates import *
from scipy.integrate import quad

##########
#constants
##########

#radi
R_jup = 71492000*u.meter
R_sat = 60268000*u.meter
R_ur = 25559000*u.meter
R_nep = 24764000*u.meter
R_ear = 6371000*u.meter
R_ven = 6052000*u.meter
R_mars = 3396000*u.meter
R_mer = 2439500*u.meter
R_sun = 695508000*u.meter


R_jup = R_jup.to(u.AU)
R_sat = R_sat.to(u.AU)
R_ur = R_ur.to(u.AU)
R_nep = R_nep.to(u.AU)
R_ear = R_ear.to(u.AU)
R_ven = R_ven.to(u.AU)
R_mars = R_mars.to(u.AU)
R_mer = R_mer.to(u.AU)
R_sun = R_sun.to(u.AU)


#effective temp of each planet in Kelvin

T_jup = 124
T_sat = 95
T_ur = 59
T_nep = 59
T_ear = 279
T_ven = 230
T_mars = 227
T_mer = 448
T_sun = 5772

temp_cmb = 2.725 #K

radius = {"Mercury": R_mer, "Mars": R_mars, "Neptune": R_nep, "Jupiter": R_jup,\
          "Uranus": R_ur, "Venus": R_ven, "Saturn": R_sat, "Sun": R_sun}
temp = {"Mercury": T_mer, "Mars": T_mars, "Neptune": T_nep, "Jupiter": T_jup,\
          "Uranus": T_ur, "Venus": T_ven, "Saturn": T_sat, "Sun": T_sun}
name = ["Mercury", "Mars", "Neptune", "Jupiter", "Uranus", "Venus", "Saturn","Sun"]


h = astropy.constants.h.value
k_B = astropy.constants.k_B.value
c = astropy.constants.c.value

########################################
#loading the arrays from Planet_info.csv
########################################

data = pd.read_csv("Planet_info.csv")

t_array = data["time"]


Mer_dist = data["Mercury Distance"]
Mars_dist = data["Mars Distance"]
Nep_dist = data["Neptune Distance"]
Jup_dist = data["Jupiter Distance"]
Ur_dist = data["Uranus Distance"]
Ven_dist = data["Venus Distance"]
Sat_dist = data["Saturn Distance"]
Sun_dist = data["Sun Distance"]

Mer_dec = data["Mercury Declination"]
Mars_dec = data["Mars Declination"]
Nep_dec = data["Neptune Declination"]
Jup_dec = data["Jupiter Declination"]
Ur_dec = data["Uranus Declination"]
Ven_dec = data["Venus Declination"]
Sat_dec = data["Saturn Declination"]
Sun_dec = data["Sun Declination"]

Mer_glat = data["Mercury Latitude"]
Mars_glat = data["Mars Latitude"]
Nep_glat = data["Neptune Latitude"]
Jup_glat = data["Jupiter Latitude"]
Ur_glat = data["Uranus Latitude"]
Ven_glat = data["Venus Latitude"]
Sat_glat = data["Saturn Latitude"]
Sun_glat = data["Sun Latitude"]

Mer_glon = data["Mercury Longitude"]
Mars_glon = data["Mars Longitude"]
Nep_glon = data["Neptune Longitude"]
Jup_glon = data["Jupiter Longitude"]
Ur_glon = data["Uranus Longitude"]
Ven_glon = data["Venus Longitude"]
Sat_glon = data["Saturn Longitude"]
Sun_glon = data["Sun Longitude"]

Mer_alt = data["Mercury Altitude"]
Mars_alt = data["Mars Altitude"]
Nep_alt = data["Neptune Altitude"]
Jup_alt = data["Jupiter Altitude"]
Ur_alt = data["Uranus Altitude"]
Ven_alt = data["Venus Altitude"]
Sat_alt = data["Saturn Altitude"]
Sun_alt = data["Sun Altitude"]
            

Wanaka = EarthLocation(lat=-44.722500*u.deg, lon=169.245833*u.deg, height=348*u.m)
utcoffset = 13*u.hour
time = Time('2019-4-1 00:00:00') - utcoffset
flight_length = int(30*24) #in hours

D_spider = .27 #m
Planck_like = 2.6

################
#loop parameters
################

lat=-44.722500*u.deg

dec_sky_min = lat - 90*u.deg
dec_sky_max = lat + 90*u.deg

dec_min = dec_sky_min.value + 20
dec_max = dec_sky_min.value + 60

########################
#time array redefinition 
########################

time_array = []

for hour in range(0,len(t_array)):
    time_i = Time(t_array[hour])
    time_array.append(time_i.datetime)
    
    
#############
#Calculations
#############


#output fequencies in GHz
f_out = np.array([150,217,280,353])

#convert to Hz
f_out = f_out*1e9

#bandpass widths
b_width = np.array([38,60,50,90])

b_width = b_width*1e9

#wave length from the output frequencies in AU
w_len = c/f_out

def solid(radius, distance):
    """Calculates the solid angle from two values of the same units
    Arguments:
        radius: radius of object
        distance: distance of the object from reference point
    Returns the solid angle
    """
    angle = (np.pi/4)*((2*radius)**2)/(distance**2)
    return angle.value

##########################
#Idividual power functions
##########################
    
def get_power_ideal(freq, bandwidth, temp):
    """Power of an object if the object filled the entier lens of the 
    telescope. Using the approximation aperture*solid angle ~ wavelength**2.
    Arguments:
        freq: frequencie at which to measure the power
        bandwidth: bandwidth corresponding to the frequency
        temp: temperature of the object being measured
    Returns: calculation of power 
    """  
    def integrand(x):
        func = h*x/(np.e**(h*x/(k_B*temp))-1)
        return func
    min = freq-bandwidth
    max = freq+bandwidth
    power = quad(integrand,min,max)[0]
    return power


def get_power_cmb(freq,bandwidth,diameter):
    """Calculates the power of the CMB for a given freqency, bandwidth, and aperture
    Arguments:
        freq: frequency at which to measure the power
        bandwidth: bandwidth corresponding to the frequency
        diameter: diameter of the aperture of the telescope in meters
    Returns: calculation of power of the CMB
    """
    theta = c/freq/diameter*1.22
    
    aperture = np.pi*(diameter/2)**2
    
    solid_angle = 4*np.pi*(np.sin(theta/4))**2 
    
    def integrand(x):
        func = h*(x**3)*solid_angle*(aperture)/((c**2)*(np.exp(h*x/(k_B*temp_cmb))-1))
        return func
        
    min = freq-bandwidth
    max = freq+bandwidth
    power = quad(integrand,min,max)[0]
    #print(f"The power of the CMB for a frequency range of {min} Hz to {max} Hz is {power} Watts")
    return power


def get_power_efficiency(tau, print_values=False):
    """Power of the planets + sun at a certain efficiency of the telescope
    Arguments:
        tau: the efficiency of the telescope
        print_values: will print the names of the planets and the power
            calculations
    Returns: calculation of power of the planets + sun in a dictionary with
        capital planet names as keys
    """  
    Power_mer = []
    Power_mars = []
    Power_nep = []
    Power_jup = []
    Power_ur = []
    Power_ven = []
    Power_sat = []
    Power_sun = []
    
    Power = {"Mercury": Power_mer, "Mars": Power_mars, "Neptune": Power_nep, "Jupiter": Power_jup,\
          "Uranus": Power_ur, "Venus": Power_ven, "Saturn": Power_sat, "Sun": Power_sun}
    
    for planet in name:
        def integrand(x):
            func = h*x/(np.e**(h*x/(k_B*temp[planet]))-1)
            return func
        if print_values:
            print(planet)
        for i in range (0,len(f_out)):
            power = quad(integrand,f_out[i]-b_width[i],f_out[i]+b_width[i])[0]
            power = power*tau
            Power[planet].append(power)
            if print_values:
                print(power)
    return Power

######## Most accurate ########

def get_power_real(freq,bandwidth,diameter, hour, print_values=False):
    """Power of the planets + sun at a certain time, frequency, and bandwidth
    Arguments:
        freq: frequencie at which to measure the power
        bandwidth: bandwidth corresponding to the frequency
        diameter: diameter of the aperture of the telescope in meters
        hour: index of the time at which the calculation will be done
        print_values: will print the names of the planets and the power
            calculations
    Returns: calculation of power of the planets + sun in a dictionary with
        capital planet names as keys
    """  
    Power_mer = []
    Power_mars = []
    Power_nep = []
    Power_jup = []
    Power_ur = []
    Power_ven = []
    Power_sat = []
    Power_sun = []
    
    aperture = np.pi*(diameter/2)**2
    
    Power = {"Mercury": Power_mer, "Mars": Power_mars, "Neptune": Power_nep, "Jupiter": Power_jup,\
          "Uranus": Power_ur, "Venus": Power_ven, "Saturn": Power_sat, "Sun": Power_sun}
    
    P_distance = {"Mercury": Mer_dist[hour], "Mars": Mars_dist[hour], \
                  "Neptune": Nep_dist[hour], "Jupiter": Jup_dist[hour],\
          "Uranus": Ur_dist[hour], "Venus": Ven_dist[hour], "Saturn": \
          Sat_dist[hour], "Sun": Sun_dist[hour]}

    for planet in name:
        solid_angle = solid((radius[planet]),P_distance[planet])
        def integrand(x):
            func = h*(x**3)*solid_angle*(aperture)/((c**2)*(np.exp(h*x/(k_B*temp[planet]))-1))
            return func
        if print_values:
            print(planet)
        power = quad(integrand,freq-bandwidth,freq+bandwidth)[0]
        Power[planet].append(power)
        if print_values:
            print(power)
    return Power


###################
#Big power function
###################

def get_power_gen(function,start = 0, end = 24):
    """Calculates power for a given power function over a specified time range
    for all the planets plus the sun.
    Arguments:
        function: power function with any arguments entered
        start: index of the time array at which to start
        end: index of the time array at which to end
    Returns: List powers arrays in the following order: P_mer, P_mars, P_nep, 
        P_jup, P_ur, P_ven, P_sat, P_sun.
    """
    P_mer = []
    P_mars = []
    P_nep = []
    P_jup = []
    P_ur = []
    P_ven = []
    P_sat = []
    P_sun = []
    for t in range(start,end):
        power = function
        P_mer.append(power[0])
        P_mars.append(power[1])
        P_nep.append(power[2])
        P_jup.append(power[3])
        P_ur.append(power[4])
        P_ven.append(power[5])
        P_sat.append(power[6])
        P_sun.append(power[7])
    return P_mer, P_mars, P_nep, P_jup, P_ur, P_ven, P_sat, P_sun 

def get_power(freq,bandwidth,diameter,start = 0, end = 24):
    """Calculates power for a given frequency, bandwidth, and aperture over a 
    specified time range for all the planets plus the sun.
    Arguments:
        freq: frequencie at which to measure the power
        bandwidth: bandwidth corresponding to the frequency
        diameter: diameter of the aperture of the telescope in meters
        start: index of the time array at which to start
        end: index of the time array at which to end
    Returns: lists of calculations of power of the planets + sun in a dictionary 
        with capital planet names as keys
    """
    aperture = np.pi*(diameter/2)**2
    
    Mer_power = []
    Mars_power = []
    Nep_power = []
    Jup_power = []
    Ur_power = []
    Ven_power = []
    Sat_power = []
    Sun_power = []
    for t in range(start,end):
        power = get_power_real(freq,bandwidth,diameter, t)
        Mer_power.append(power["Mercury"])
        Mars_power.append(power["Mars"])
        Nep_power.append(power["Neptune"])
        Jup_power.append(power["Jupiter"])
        Ur_power.append(power["Uranus"])
        Ven_power.append(power["Venus"])
        Sat_power.append(power["Saturn"])
        Sun_power.append(power["Sun"])
    
    Power = {"Mercury": Mer_power, "Mars": Mars_power, "Neptune": Nep_power, "Jupiter": Jup_power,\
          "Uranus": Ur_power, "Venus": Ven_power, "Saturn": Sat_power, "Sun": Sun_power}
    
    return Power
    
p_power = get_power(f_out[0], b_width[0],Planck_like,start = 0, end = len(time_array))



def avg_power(dict):
    """Accepts a dictionary containing lists of values and returns a dictionary
        with the same keys and averages of the lists. Can be used for lists of 
        any type of value, not just power"""
    Mer_p_avg = 0
    Mars_p_avg = 0
    Nep_p_avg = 0
    Jup_p_avg = 0
    Ur_p_avg = 0
    Ven_p_avg = 0
    Sat_p_avg = 0
    Sun_p_avg = 0
            
    averages = {"Mercury": Mer_p_avg, "Mars": Mars_p_avg, "Neptune": Nep_p_avg, "Jupiter": Jup_p_avg,\
          "Uranus": Ur_p_avg, "Venus": Ven_p_avg, "Saturn": Sat_p_avg, "Sun": Sun_p_avg}
    for key in dict.keys():
        averages[key] = np.average(dict[key])
    return averages

A = avg_power(p_power)

cmb = get_power_cmb(f_out[0], b_width[0],Planck_like)

cmb_ratios = {}

for key in A.keys():
    cmb_ratios[key] = A[key]/cmb
    

##########################################
#Plotting all the values against date/time
##########################################


def power_plot(Mercury=True, Mars=True, Neptune=True, Jupiter=True, Uranus=\
               True, Venus=True, Saturn=True, Sun=False, save_plot = False,\
               start=0,end=len(time_array)):
    """Plots the power of the planets + sun over time plotted in semilogy
    Arguments:
        Mercury: plot Mercury data, default True
        Mars: plot mars data, default True
        Neptune: plot Neptune data, default True
        Jupiter: plot Jupiter data, default True
        Uranus: plot Uranus data, default True
        Venus: plot Venus data, default True
        Saturn: plot Saturn data, default True
        Sun: plot Sun data, default False
        save_plot: will save the figure as "power_plot.png"
        start: index of the time array at which to start
        end: index of the time array at which to end
    Returns nothing
    """
    
    if Mercury:
        plt.semilogy(time_array[start:end],\
                     Mer_power[start:end], label = "Mercury")
    
    if Mars:
        plt.semilogy(time_array[start:end],\
                     Mars_power[start:end], label = "Mars")
    
    if Neptune:
        plt.semilogy(time_array[start:end],\
                     Nep_power[start:end], label = "Neptune")
    
    if Jupiter:
        plt.semilogy(time_array[start:end],\
                     Jup_power[start:end], label = "Jupiter")
    
    if Uranus:
        plt.semilogy(time_array[start:end],\
                     Ur_power[start:end], label = "Uranus")    
    
    if Venus:
        plt.semilogy(time_array[start:end],\
                     Ven_power[start:end], label = "Venus")    
    
    if Saturn:
        plt.semilogy(time_array[start:end],\
                     Sat_power[start:end], label = "Saturn")    
    
    if Sun:
        plt.semilogy(time_array[start:end],\
                     Sun_power[start:end], label = "Sun")
    
    plt.gcf().autofmt_xdate()
    
    plt.xlabel("Date in Wanaka")
    plt.ylabel("Power")
    plt.title("Planet power over time")
    plt.legend()
    if save_plot:
        plt.savefig("power_plot.png")
    plt.show()
    


def dec_plot(Mercury=True, Mars=True, Neptune=True, Jupiter=True, Uranus=\
               True, Venus=True, Saturn=True, Sun=True, save_plot = False,\
               start=0,end=len(time_array)):
    """Plots the declination of the planets + sun over time
    Arguments:
        Mercury: plot Mercury data, default True
        Mars: plot mars data, default True
        Neptune: plot Neptune data, default True
        Jupiter: plot Jupiter data, default True
        Uranus: plot Uranus data, default True
        Venus: plot Venus data, default True
        Saturn: plot Saturn data, default True
        Sun: plot Sun data, default True
        save_plot: will save the figure as "declination_plot.png"
        start: index of the time array at which to start
        end: index of the time array at which to end
    Returns nothing
    """
    plt.figure()
    
    if Mercury:
        plt.plot(time_array[start:end],Mer_dec[start:end], label = "Mercury")
    
    if Mars:
        plt.plot(time_array[start:end],Mars_dec[start:end], label = "Mars")
    
    if Neptune:
        plt.plot(time_array[start:end],Nep_dec[start:end], label = "Neptune")
    
    if Jupiter:
        plt.plot(time_array[start:end],Jup_dec[start:end], label = "Jupiter")
    
    if Uranus:
        plt.plot(time_array[start:end],Ur_dec[start:end], label = "Uranus")    
    
    if Venus:
        plt.plot(time_array[start:end],Ven_dec[start:end], label = "Venus")    
    
    if Saturn:
        plt.plot(time_array[start:end],Sat_dec[start:end], label = "Saturn")    
    
    if Sun:
        plt.plot(time_array[start:end],Sun_dec[start:end], label = "Sun")
    
    plt.gcf().autofmt_xdate()
    
    plt.xlabel("Date in Wanaka")
    plt.ylabel("Declination from Earth in degrees")
    plt.title("Planet declination over time")
    plt.legend()
    if save_plot:
        plt.savefig("declination_plot.png")
    plt.show()

def glat_plot(Mercury=True, Mars=True, Neptune=True, Jupiter=True, Uranus=\
               True, Venus=True, Saturn=True, Sun=True, save_plot = False,\
               start=0,end=len(time_array)):
    """Plots the galactic latitude of the planets + sun over time
    Arguments:
        Mercury: plot Mercury data, default True
        Mars: plot mars data, default True
        Neptune: plot Neptune data, default True
        Jupiter: plot Jupiter data, default True
        Uranus: plot Uranus data, default True
        Venus: plot Venus data, default True
        Saturn: plot Saturn data, default True
        Sun: plot Sun data, default True
        save_plot: will save the figure as "galactic_latitude_plot.png"
        start: index of the time array at which to start
        end: index of the time array at which to end
    Returns nothing
    """
    plt.figure()
    
    if Mercury:
        plt.plot(time_array[start:end],Mer_glat[start:end], label = "Mercury")
    
    if Mars:
        plt.plot(time_array[start:end],Mars_glat[start:end], label = "Mars")
    
    if Neptune:
        plt.plot(time_array[start:end],Nep_glat[start:end], label = "Neptune")
            
    if Jupiter:
        plt.plot(time_array[start:end],Jup_glat[start:end], label = "Jupiter")
            
    if Uranus:
        plt.plot(time_array[start:end],Ur_glat[start:end], label = "Uranus")
        
    if Venus:
        plt.plot(time_array[start:end],Ven_glat[start:end], label = "Venus")
        
    if Saturn:
        plt.plot(time_array[start:end],Sat_glat[start:end], label = "Saturn")
        
    if Sun:
        plt.plot(time_array[start:end],Sun_glat[start:end], label = "Sun")
    plt.gcf().autofmt_xdate()
    
    plt.xlabel("Date in Wanaka")
    plt.ylabel("Galactic Latitude")
    plt.title("Galactic Latitude")
    plt.legend()
    if save_plot:
        plt.savefig("galactic_latitude_plot.png")
    plt.show()

def dist_plot(Mercury=True, Mars=True, Neptune=True, Jupiter=True, Uranus=\
               True, Venus=True, Saturn=True, Sun=True, save_plot = False,\
               start=0,end=len(time_array)):
    """Plots the distance of the planets + sun over time
    Arguments:
        Mercury: plot Mercury data, default True
        Mars: plot mars data, default True
        Neptune: plot Neptune data, default True
        Jupiter: plot Jupiter data, default True
        Uranus: plot Uranus data, default True
        Venus: plot Venus data, default True
        Saturn: plot Saturn data, default True
        Sun: plot Sun data, default True
        save_plot: will save the figure as "distance_plot.png"
        start: index of the time array at which to start
        end: index of the time array at which to end
    Returns nothing
    """
    plt.figure()
    
    if Mercury:
        plt.semilogy(time_array[start:end],Mer_dist[start:end], label = "Mercury")
            
    if Mars:
        plt.semilogy(time_array[start:end],Mars_dist[start:end], label = "Mars")
            
    if Venus:
        plt.semilogy(time_array[start:end],Nep_dist[start:end], label = "Neptune")
            
    if Jupiter:
        plt.semilogy(time_array[start:end],Jup_dist[start:end], label = "Jupiter")
            
    if Uranus:
        plt.semilogy(time_array[start:end],Ur_dist[start:end], label = "Uranus")
            
    if Venus:
        plt.semilogy(time_array[start:end],Ven_dist[start:end], label = "Venus")
        
    if Saturn:
        plt.semilogy(time_array[start:end],Sat_dist[start:end], label = "Saturn")
        
    if Sun:
        plt.semilogy(time_array[start:end],Sun_dist[start:end], label = "Sun")
    
    plt.gcf().autofmt_xdate()
    
    plt.xlabel("Date in Wanaka")
    plt.ylabel("Distance from Earth (AU)")
    plt.title("Planet distance over time")
    plt.legend()
    if save_plot:
        plt.savefig("distance_plot.png")
    plt.show()

def glon_plot(Mercury=True, Mars=True, Neptune=True, Jupiter=True, Uranus=\
               True, Venus=True, Saturn=True, Sun=True, save_plot = False,\
               start=0,end=len(time_array)):
    """Plots the galactic longitude of the planets + sun over time
    Arguments:
        Mercury: plot Mercury data, default True
        Mars: plot mars data, default True
        Neptune: plot Neptune data, default True
        Jupiter: plot Jupiter data, default True
        Uranus: plot Uranus data, default True
        Venus: plot Venus data, default True
        Saturn: plot Saturn data, default True
        Sun: plot Sun data, default False
        save_plot: will save the figure as "galactic_longitude_plot.png"
        start: index of the time array at which to start
        end: index of the time array at which to end
    Returns nothing
    """
    plt.figure()
    
    if Mercury:
        plt.plot(time_array[start:end],Mer_glon[start:end], label = "Mercury")
    
    if Mars:
        plt.plot(time_array[start:end],Mars_glon[start:end], label = "Mars")
    
    if Neptune:
        plt.plot(time_array[start:end],Nep_glon[start:end], label = "Neptune")
            
    if Jupiter:
        plt.plot(time_array[start:end],Jup_glon[start:end], label = "Jupiter")
            
    if Uranus:
        plt.plot(time_array[start:end],Ur_glon[start:end], label = "Uranus")
        
    if Venus:
        plt.plot(time_array[start:end],Ven_glon[start:end], label = "Venus")
        
    if Saturn:
        plt.plot(time_array[start:end],Sat_glon[start:end], label = "Saturn")
        
    if Sun:
        plt.plot(time_array[start:end],Sun_glon[start:end], label = "Sun")
    plt.gcf().autofmt_xdate()
    
    plt.xlabel("Date in Wanaka")
    plt.ylabel("Galactic Longitude")
    plt.title("Galactic Longitude")
    plt.legend()
    if save_plot:
        plt.savefig("galactic_longitude_plot.png")
    plt.show()


def alt_plot(Mercury=True, Mars=True, Neptune=True, Jupiter=True, Uranus=\
               True, Venus=True, Saturn=True, Sun=True, save_plot = False,\
               start=0,end=len(time_array)):
    """Plots the altitude of the planets + sun over time
    Arguments:
        Mercury: plot Mercury data, default True
        Mars: plot mars data, default True
        Neptune: plot Neptune data, default True
        Jupiter: plot Jupiter data, default True
        Uranus: plot Uranus data, default True
        Venus: plot Venus data, default True
        Saturn: plot Saturn data, default True
        Sun: plot Sun data, default True
        save_plot: will save the figure as "altitude_plot.png"
        start: index of the time array at which to start
        end: index of the time array at which to end
    Returns nothing
    """
    plt.figure()
    
    if Mercury:
        plt.plot(time_array[start:end],Mer_alt[start:end], label = "Mercury")
            
    if Mars:
        plt.plot(time_array[start:end],Mars_alt[start:end], label = "Mars")
            
    if Neptune:
        plt.plot(time_array[start:end],Nep_alt[start:end], label = "Neptune")
            
    if Jupiter:
        plt.plot(time_array[start:end],Jup_alt[start:end], label = "Jupiter")
            
    if Uranus:
        plt.plot(time_array[start:end],Ur_alt[start:end], label = "Uranus")
            
    if Venus:
        plt.plot(time_array[start:end],Ven_alt[start:end], label = "Venus")
        
    if Saturn:
        plt.plot(time_array[start:end],Sat_alt[start:end], label = "Saturn")
        
    if Sun:
        plt.plot(time_array[start:end],Sun_alt[start:end], label = "Sun")
    
    plt.gcf().autofmt_xdate()
    
    plt.xlabel("Date in Wanaka")
    plt.ylabel("Altitude from Earth (AU)")
    plt.title("Planet altitude over time")
    plt.legend()
    if save_plot:
        plt.savefig("altitude_plot.png")
    plt.show()