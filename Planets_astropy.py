"""
Code for finding the planets over Wanaka New Zealand. Data is taken from 4/1/2019
to 4/20/2019 in hour intervals using astropy
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime
import astropy
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import *
import pandas as pd


############################
#creating Wanaka New Zealand
############################

Wanaka = EarthLocation(lat=-44.722500*u.deg, lon=169.245833*u.deg, height=348*u.m)
utcoffset = 13*u.hour
time = Time('2019-4-1 00:00:00') - utcoffset

################
#loop parameters
################

flight_length = int(30*24) #in hours

def get_planets(location, time_i, flight_length, distance = True, declination\
                =True, g_latitude=True, g_longitude=True, altitude=True, 
                save_file=True, return_values=False,debug=False):
    """Calculates the distance, declination, galactic latitude, and galactic
    longitude of the planets plus the sun over some Earth location in hour 
    intervals for the duration of a flight. Data is saved in arrays to a csv 
    file or printed as a dictionary.
    Arguments:
        location: astropy EarthLocation calculation
        time_i: start time of the flight in astropy units of hours
        flight_length: total time the telescope will be taking data, integer 
        distance: True or False taking distance data in A.U., default is True
        declination: True or False taking declination data in degeea, default 
            is True
        g_latitude: True or False taking galactic lataitude data in degrees, 
            default is True
        g_longitude: True or False taking galactic longitude data in degrees, 
            default is True
        altitude: True or False taking altitude data in degrees, default is 
            True
        save_file: save the data collected to a csv file named 
            "Planet_info.csv", default is True
        return_values: returns the arrays without saving them to a file, 
            default is False
        debug: prints out the line that is being calculated
    Returns: Nothing if return_values is False, dictionary containing data if 
        return_values is True, saves a cvs file with the data if save_file is 
        true, no data saved if save_file is false.
    """ 
    
    ##############
    #planet arrays
    ##############
    
    time_array = []
    
    Mer_dist = []
    Mars_dist = []
    Nep_dist = []
    Jup_dist = []
    Ur_dist = []
    Ven_dist = []
    Sat_dist = []
    Sun_dist = []

    Mer_dec = []
    Mars_dec = []
    Nep_dec = []
    Jup_dec = []
    Ur_dec = []
    Ven_dec = []
    Sat_dec = []
    Sun_dec = []

    Mer_glat = []
    Mars_glat = []
    Nep_glat = []
    Jup_glat = []
    Ur_glat = []
    Ven_glat = []
    Sat_glat = []
    Sun_glat = []

    Mer_glon = []
    Mars_glon = []
    Nep_glon = []
    Jup_glon = []
    Ur_glon = []
    Ven_glon = []
    Sat_glon = []
    Sun_glon = []
    
    Mer_alt = []
    Mars_alt = []
    Nep_alt = []
    Jup_alt = []
    Ur_alt = []
    Ven_alt = []
    Sat_alt = []
    Sun_alt = []

    #####
    #loop
    #####
    
    for hour in range(0,flight_length):
        now = time_i + hour*u.hour
        time_array.append(now.datetime)
        
        mer = get_body('mercury', now, Wanaka)
        mars = get_body('mars', now, Wanaka)
        nep = get_body('neptune', now, Wanaka)
        jup = get_body('jupiter', now, Wanaka)
        ur = get_body('uranus', now, Wanaka)
        ven = get_body('venus', now, Wanaka)
        sat = get_body('saturn', now, Wanaka)
        sun = get_body('sun', now, Wanaka)
        
        if distance:
            Mer_dist.append(mer.distance.AU)
            Mars_dist.append(mars.distance.AU)
            Nep_dist.append(nep.distance.AU)
            Jup_dist.append(jup.distance.AU)
            Ur_dist.append(ur.distance.AU)
            Ven_dist.append(ven.distance.AU)
            Sat_dist.append(sat.distance.AU)
            Sun_dist.append(sun.distance.AU)
            
        if declination:
            Mer_dec.append(mer.dec.deg)
            Mars_dec.append(mars.dec.deg)
            Nep_dec.append(nep.dec.deg)
            Jup_dec.append(jup.dec.deg)
            Ur_dec.append(ur.dec.deg)
            Ven_dec.append(ven.dec.deg)
            Sat_dec.append(sat.dec.deg)
            Sun_dec.append(sun.dec.deg)
            
        if g_latitude:
            Mer_glat.append(mer.galactic.data.lat.deg)
            Mars_glat.append(mars.galactic.data.lat.deg)
            Nep_glat.append(nep.galactic.data.lat.deg)
            Jup_glat.append(nep.galactic.data.lat.deg)
            Ur_glat.append(ur.galactic.data.lat.deg)
            Ven_glat.append(ven.galactic.data.lat.deg)
            Sat_glat.append(sat.galactic.data.lat.deg)
            Sun_glat.append(sun.galactic.data.lat.deg)
            
        if g_longitude:
            Mer_glon.append(mer.galactic.data.lon.deg)
            Mars_glon.append(mars.galactic.data.lon.deg)
            Nep_glon.append(nep.galactic.data.lon.deg)
            Jup_glon.append(nep.galactic.data.lon.deg)
            Ur_glon.append(ur.galactic.data.lon.deg)
            Ven_glon.append(ven.galactic.data.lon.deg)
            Sat_glon.append(sat.galactic.data.lon.deg)
            Sun_glon.append(sun.galactic.data.lon.deg)
            
        if altitude:
           mercuryaltaz = mer.transform_to(AltAz(obstime=now, location = Wanaka)) 
           Mer_alt.append(mercuryaltaz.alt.degree)
            
           marsaltaz = mars.transform_to(AltAz(obstime=now, location = Wanaka)) 
           Mars_alt.append(marsaltaz.alt.degree)
           
           neptunealtaz = nep.transform_to(AltAz(obstime=now, location = Wanaka)) 
           Nep_alt.append(neptunealtaz.alt.degree)
           
           jupiteraltaz = jup.transform_to(AltAz(obstime=now, location = Wanaka)) 
           Jup_alt.append(jupiteraltaz.alt.degree)
           
           uranusaltaz = ur.transform_to(AltAz(obstime=now, location = Wanaka)) 
           Ur_alt.append(uranusaltaz.alt.degree)
           
           venusaltaz = ven.transform_to(AltAz(obstime=now, location = Wanaka)) 
           Ven_alt.append(venusaltaz.alt.degree)
           
           saturnaltaz = sat.transform_to(AltAz(obstime=now, location = Wanaka)) 
           Sat_alt.append(saturnaltaz.alt.degree)
           
           sunaltaz = sun.transform_to(AltAz(obstime=now, location = Wanaka)) 
           Sun_alt.append(sunaltaz.alt.degree)
        if debug:
            print("done with line", now)
        
    
    #####################
    #Saving data as a csv
    #####################
    
    if save_file:
            
        df = pd.DataFrame({"time" : time_array, "Mercury Distance" : Mer_dist, \
                           "Mars Distance": Mars_dist, "Neptune Distance" : Nep_dist, \
                           "Jupiter Distance" : Jup_dist, "Uranus Distance": Ur_dist,\
                           "Venus Distance" : Ven_dist, "Saturn Distance" : Sat_dist,\
                           "Sun Distance" : Sun_dist,
                           "Mercury Declination": Mer_dec,"Mars Declination":Mars_dec,\
                           "Neptune Declination":Nep_dec,"Jupiter Declination":Jup_dec,\
                           "Uranus Declination" : Ur_dec,"Venus Declination" : Ven_dec,\
                           "Saturn Declination" : Sat_dec,
                           "Sun Declination" : Sun_dec, "Mercury Latitude" : Mer_glat,\
                           "Mars Latitude":Mars_glat, "Neptune Latitude":Nep_glat,\
                           "Jupiter Latitude":Jup_glat,"Uranus Latitude" : Ur_glat,\
                           "Venus Latitude" : Ven_glat, "Saturn Latitude" : Sat_glat,\
                           "Sun Latitude" : Sun_glat,
                           "Mercury Longitude": Mer_glon,"Mars Longitude":Mars_glon,\
                           "Neptune Longitude":Nep_glon,"Jupiter Longitude":Jup_glon,\
                           "Uranus Longitude" : Ur_glon,"Venus Longitude" : Ven_glon,\
                           "Saturn Longitude" : Sat_glon, "Sun Longitude" : Sun_glon,
                           "Mercury Altitude": Mer_alt,"Mars Altitude":Mars_alt,\
                           "Neptune Altitude":Nep_alt,"Jupiter Altitude":Jup_alt,\
                           "Uranus Altitude" : Ur_alt,"Venus Altitude" : Ven_alt,\
                           "Saturn Altitude" : Sat_alt, "Sun Altitude" : Sun_alt})
            
        df.to_csv("Planet_info.csv", index=False)
        
        return "file saved"
        
    #######################
    #returning a dictionary
    #######################
    if return_values:
        return {"time" : time_array, "Mercury Distance" : Mer_dist, \
                           "Mars Distance": Mars_dist, "Neptune Distance" : Nep_dist, \
                           "Jupiter Distance" : Jup_dist, "Uranus Distance": Ur_dist,\
                           "Venus Distance" : Ven_dist, "Saturn Distance" : Sat_dist,\
                           "Sun Distance" : Sun_dist,
                           "Mercury Declination": Mer_dec,"Mars Declination":Mars_dec,\
                           "Neptune Declination":Nep_dec,"Jupiter Declination":Jup_dec,\
                           "Uranus Declination" : Ur_dec,"Venus Declination" : Ven_dec,\
                           "Saturn Declination" : Sat_dec,
                           "Sun Declination" : Sun_dec, "Mercury Latitude" : Mer_glat,\
                           "Mars Latitude":Mars_glat, "Neptune Latitude":Nep_glat,\
                           "Jupiter Latitude":Jup_glat,"Uranus Latitude" : Ur_glat,\
                           "Venus Latitude" : Ven_glat, "Saturn Latitude" : Sat_glat,\
                           "Sun Latitude" : Sun_glat,
                           "Mercury Longitude": Mer_glon,"Mars Longitude":Mars_glon,\
                           "Neptune Longitude":Nep_glon,"Jupiter Longitude":Jup_glon,\
                           "Uranus Longitude" : Ur_glon,"Venus Longitude" : Ven_glon,\
                           "Saturn Longitude" : Sat_glon, "Sun Longitude" : Sun_glon,
                           "Mercury Altitude": Mer_alt,"Mars Altitude":Mars_alt,\
                           "Neptune Altitude":Nep_alt,"Jupiter Altitude":Jup_alt,\
                           "Uranus Altitude" : Ur_alt,"Venus Altitude" : Ven_alt,\
                           "Saturn Altitude" : Sat_alt, "Sun Altitude" : Sun_alt}




