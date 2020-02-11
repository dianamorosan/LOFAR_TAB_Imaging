# Python code that plots triangulated intensity maps with
# dynamic spectra for 3 frequencies
#
# Functions for background subtraction of data included
#
# Functionality added for saving images fits files
#
#
# Author: Diana Morosan (diana.morosan@helsinki.fi)
# Latest version: October 2019
#
# Please acknowledge the use of this code


import h5py
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as pl
import time
import datetime
import numpy as np
from pylab import figure,imshow,xlabel,ylabel,title,close,colorbar
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import date2num
from matplotlib import dates
from matplotlib.colors import LogNorm
import pyfits
import sunpy.coordinates
import astropy.coordinates
from astropy.time import Time 
import sunpy.sun
import glob

############################################
# function that writes an image to fits file
############################################
def make_fits(filename, image_data, x, y, x_delta, y_delta, time ):
	
	header = pyfits.Header( [ ('CRVAL1', x),('CRPIX1',0),('CTYPE1','X (arcmin)'),('CDELT1',x_delta),('CRVAL2', y),('CRPIX2',0),('CTYPE2','Y (arcmin)'),('CDELT2',y_delta), ('T_OBS',time) ]  )
	pyfits.writeto(filename, image_data, header)


#####################################
# create quiet time normalising array
#####################################
def quiet_time_array(start_min, end_min):
    start_min_quiet = start_min
    end_min_quiet = end_min

    start_line_quiet = int( (start_min_quiet/(total_time/60.))*t_lines )
    end_line_quiet = int( (end_min_quiet/(total_time/60.))*t_lines )

    # beam far away from sources to create normalising array
    file = obsid + '_SAP000_B070_S0_P000_bf.h5'
    f = h5py.File( file, 'r' )
    data = f['SUB_ARRAY_POINTING_000/BEAM_070/STOKES_0'][start_line_quiet:end_line_quiet,:]
    print( 'Quiet Data: ', data.shape )
    data = np.mean(data, axis = 0)
    print( data.shape )
    array_quiet = data
    print( array_quiet )

    np.savetxt('Quiet_time_array_1541.txt', array_quiet, fmt = '%10.5f')

    return array_quiet

#######################################
# create median array for normalization
#######################################
def median_array(beam_file):

    # input: beam far away from sources to create normalising array
    file = beam_file
    f = h5py.File( file, 'r' )
    data = f['SUB_ARRAY_POINTING_000/BEAM_070/STOKES_0'][start_line_norm:end_line_norm,:]

    median_arr = []
    for sb in range(data.shape[1]):
        median_arr.append(np.median(data[:,sb]))

    return median_arr
    #print len(median_arr), median_arr

###########################
# coordinate transformation
###########################`
def coordinate_transformation(ra, dec, time):

    time_obs = time

    # polar north angle
    p = sunpy.coordinates.sun.P(time_obs).rad # quantities needed in radians
    # sunpy.coordinates.get_sun_P - deprecated after sunpy 1.0

    # solar ra and dec on the day
    ra0 = astropy.coordinates.get_sun(Time(time_obs)).ra.rad #sunpy.sun.true_rightascension(time_obs).rad
    dec0 = astropy.coordinates.get_sun(Time(time_obs)).dec.rad #sunpy.sun.true_declination(time_obs).rad

    ra = ra*np.pi/180
    dec = dec*np.pi/180

    '''
    # method 1 - approximation
    solar_x =  ( -(ra - ra0)*np.cos( dec0 )*np.cos( p ) + (dec - dec0)*np.sin( p ) )*180/np.pi*60
    solar_y =  ( (ra - ra0)*np.cos( dec0 )*np.sin( p ) + (dec - dec0)*np.cos( p ) )*180/np.pi*60
    '''

    # method 2 - exact method - based on ALMA transformation reference
    rho = np.arccos( np.cos( dec )*np.cos( dec0 )*np.cos(ra - ra0) + np.sin(dec)*np.sin(dec0) )
    phi = np.arctan2( ( np.sin(ra-ra0) ), ( np.tan(dec)*np.cos(dec0) - np.sin(dec0)*np.cos(ra-ra0) ) )

    solar_x = np.arctan( -np.tan(rho)*np.sin(phi-p))*180/np.pi*60
    solar_y = np.arctan(np.tan(rho)*np.cos(phi-p))*180/np.pi*60

    return solar_x, solar_y



# extracting file header
files = glob.glob('*.h5')
files.sort()
f = h5py.File( files[0], 'r' )

beam = f.attrs['FILENAME'].decode("utf-8")[16:19]
stokes = f.attrs['FILENAME'].decode("utf-8")[21:22]
sap = f.attrs['FILENAME'].decode("utf-8")[11:14]

data = f['SUB_ARRAY_POINTING_000/BEAM_000/STOKES_0'][:,:]

freq_arr =  f['SUB_ARRAY_POINTING_000/BEAM_000/COORDINATES/COORDINATE_1'].attrs['AXIS_VALUES_WORLD']
freq_arr = freq_arr/1e6 #in Mhz

print( 'Data array shape: ', data.shape )

t_lines = data.shape[0]
f_lines = data.shape[1]

total_time = list(f.attrs.values())[22] #in seconds
lines_per_sec = int(t_lines/total_time)
print( 'Lines per second: ', lines_per_sec )

# extracting time informattion and write it in python format
time = f.attrs['OBSERVATION_START_UTC'].decode("utf-8")
start_time_obs = datetime.datetime.strptime( time, '%Y-%m-%dT%H:%M:%S.%f000Z' )

print( 'Start time observation:', start_time_obs.time() )


'''
    Start user input parameters
'''

#########################
# useful input parameters
# can be included as command line arguments instead
###################################################
nfiles = len(files)# number of beams + 1
obsid = 'L99008'

##############################################################################
# parameters to play with depending on which time/freq/chunk of data is needed
##############################################################################

# image cadence
averaging_time = 1 #in seconds

# beam to use in dynamic spectrum plot
dynamic_spectrum_beam = 9

#clipping parameters for image
zmin = 0.9
zmax = 3

# start/end minute of movie
start_min = 0.
end_min = 2.

# start time
start_time =  (start_time_obs + datetime.timedelta(minutes = start_min) )
print( 'Start time movie:', start_time.time() )

start_min_norm = 0.
end_min_norm = 15.#total_time/60.

# specify 3 input subbands
# use for simultaneous TAB and interferometric modes
sb = 25
sb1 = 37
sb2 = 48

# or
# 3 input frequencies -- specify frequency value for equal frequency spacing only
# not suitable for simultaneous TAB and interferometric modes -- need to specify subband number
start_freq = freq_arr[sb*16] #46 #in MHz
end_freq = freq_arr[sb*16+16] #46.5 #in MHz

start_freq1 = freq_arr[sb1*16] #60 #in MHz
end_freq1 = freq_arr[sb1*16+16] #60.5 #in MHz

start_freq2 = freq_arr[sb2*16] #70 #in MHz
end_freq2 = freq_arr[sb2*16+16] #70.5 #in MHz


'''
    End user input parameters
'''


################################
# calculating imaging parameters
################################

start_line = int( (start_min/(total_time/60.))*t_lines ) 
end_line = int( (end_min/(total_time/60.))*t_lines ) 
print( 'Start line/End line: ', start_line, end_line)

start_line_norm = start_line #int( (start_min/(total_time/60.))*t_lines )
end_line_norm = int( (end_min/(total_time/60.))*t_lines )

# no of images based on start/end time
no_images = int( (end_line - start_line)/(lines_per_sec*averaging_time) ) 

print( 'Number of images: ' + str(no_images) )

start_freq_ds = list(f.attrs.values())[30] #in MHz
end_freq_ds = list(f.attrs.values())[8]

start_freq_ds_plot = start_freq_ds
end_freq_ds_plot = end_freq_ds

t_resolution = (total_time/t_lines)*1000 #in milliseconds
f_resolution = (end_freq_ds - start_freq_ds)/f_lines

start_freq_line_ds = int( (start_freq_ds_plot - start_freq_ds)/f_resolution )
end_freq_line_ds = int( (end_freq_ds_plot - start_freq_ds)/f_resolution )


# constructing a time array from the start time
time_new = np.zeros( end_line - start_line )
for j in range(len(time_new)):
    z = (start_time + datetime.timedelta(milliseconds = t_resolution*j))
    time_new[j] = matplotlib.dates.date2num(z)

# creating frequency array
freq = np.zeros( int((end_freq_ds_plot - start_freq_ds_plot)/f_resolution) )
for k in range(len(freq)):
    freq[k] = start_freq_ds_plot + k*f_resolution

# estimating location of frequency slices
# here avergaing over one subband with 16 channels per subband
# for simultaneous TAB and interferometric observations
start_freq_line = sb*16 #int( (start_freq - start_freq_ds)/f_resolution )
end_freq_line = sb*16 + 16 #int( (end_freq - start_freq_ds)/f_resolution)

start_freq_line1 = sb1*16 #int( (start_freq1 - start_freq_ds)/f_resolution ) # 10 MHz corresponds to 800 frequency lines
end_freq_line1 = sb1*16+16 #int( (end_freq1 - start_freq_ds)/f_resolution)

start_freq_line2 = sb2*16 #int( (start_freq2 - start_freq_ds)/f_resolution )
end_freq_line2 = sb2*16+16 #int( (end_freq2 - start_freq_ds)/f_resolution)


intensity = np.zeros(( nfiles, no_images ))
intensity1 = np.zeros(( nfiles, no_images ))
intensity2 = np.zeros(( nfiles, no_images ))

ra = np.zeros( nfiles )
dec = np.zeros( nfiles )

INT = np.zeros( nfiles )
INT1 = np.zeros( nfiles )
INT2 = np.zeros( nfiles )

# Background subtraction arrays:
#array_quiet = quiet_time_array(455, 455.5)
#array_quiet = np.loadtxt( 'Quiet_time_array_1541.txt' )
median_arr = median_array(obsid + '_SAP000_B070_S0_P000_bf.h5')

###################################################
#extracting intensity information for 3 frequencies
###################################################

count = 0

# looping through beams
for i in range(nfiles):

    filename = obsid+'_SAP000_B'+str(i).rjust(3,'0')+'_S0_P000_bf.h5'
    print( filename )
    f = h5py.File(filename,'r')
    # extracting coordinates of individual beam from h5py file
    ra[count] = list(f['SUB_ARRAY_POINTING_000/BEAM_'+str(i).rjust(3,'0')].attrs.values())[17]
    dec[count] = list(f['SUB_ARRAY_POINTING_000/BEAM_'+str(i).rjust(3,'0')].attrs.values())[20]

    # extracting data at specific frequency in MHz bins, eg 50-55 MHz
    # extract entire/big chunk of data set first if normalisation needed later
    data = f['SUB_ARRAY_POINTING_000/BEAM_'+str(i).rjust(3,'0')+'/STOKES_0'][start_line_norm:end_line, :]

    '''
        normalization technique can be changed below
    '''

    # normalising to take out the frequency response
    for sb in range(data.shape[1]):
        
        # very important data normalising step: either by median in each beam
        # or by median_array/quiet_time calculated from far-away quiet beam
        # can also be calculated at a later stage
        data[:,sb] = data[:,sb]/np.median(data[:end_line_norm,sb])#median_arr[sb]#array_quiet[sb]#

    data0 = data[start_line-start_line_norm:, start_freq_line:end_freq_line]
    data1 = data[start_line-start_line_norm:, start_freq_line1:end_freq_line1]
    data2 = data[start_line-start_line_norm:, start_freq_line2:end_freq_line2]

    # average over frequency range
    data0 = np.mean(data0, axis = 1)
    data1 = np.mean(data1, axis = 1)
    data2 = np.mean(data2, axis = 1)

    for j in range(no_images):
        # average intensity over N sec
        intensity[ count ][ j ] = np.mean(data0[int(j*lines_per_sec*averaging_time):int((j+1)*lines_per_sec*averaging_time)]) # 1 second = lines_per_sec lines of time
        intensity1[ count ][ j ] = np.mean(data1[int(j*lines_per_sec*averaging_time):int((j+1)*lines_per_sec*averaging_time)])
        intensity2[ count ][ j ] = np.mean(data2[int(j*lines_per_sec*averaging_time):int((j+1)*lines_per_sec*averaging_time)])

    count = count + 1
    f.close()


    #######################################
    #extracting dynamic spectra information
    #######################################

    for i in range(dynamic_spectrum_beam,dynamic_spectrum_beam+1):
    filename = obsid+'_SAP000_B'+str(i).rjust(3,'0')+'_S0_P000_bf.h5'
    print( filename )
    f = h5py.File(filename,'r')
    data = f['SUB_ARRAY_POINTING_000/BEAM_'+str(i).rjust(3,'0')+'/STOKES_0'][start_line:end_line,start_freq_line_ds:end_freq_line_ds]
    for sb in range(data.shape[1]):
        data[:,sb] = data[:,sb]/np.median(data[:,sb])
    data = np.transpose(data)

    f.close()


##################################################
#plotting the data to check background subtraction
##################################################
   
#print( intensity.shape )

# time array
time_int = []#np.zeros(intensity.shape[1])
for i in range(intensity.shape[1]):
	time_int.append( start_time + datetime.timedelta(seconds = i*averaging_time))

pl.figure(0,figsize=(22,14))

pl.plot( time_int, intensity[dynamic_spectrum_beam, :], color = 'r', label = '40 MHz' )
pl.plot( time_int, intensity1[dynamic_spectrum_beam, :], color = 'b', label = '50 MHz' )
pl.plot( time_int, intensity2[dynamic_spectrum_beam, :], color = 'g', label = '60 MHz' )
pl.legend()
ax = pl.gca()
#ax.set_yscale('log')
ax.xaxis_date()
ax.xaxis.set_major_locator(dates.MinuteLocator())
ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S'))
ax.xaxis.set_major_locator( MaxNLocator(nbins = 7) )
pl.savefig('Normalized_intensity.png')
#pl.show()

# dynamic spectrum extent
xmin1 = np.min(time_new)
xmax1 = np.max(time_new)
ymin1 = np.min(freq)
ymax1 = np.max(freq)

# looping over time
for i in range(no_images):

    pl.figure(1,figsize=(22,14))

    '''
    plotting dynamic spectrum and time bar
    '''
    ax1 = pl.subplot2grid((2,3), (0,0), colspan=3)
    imshow(data, vmin = zmin , vmax = zmax, aspect='auto', norm = LogNorm(), extent=(xmin1,xmax1,end_freq_ds_plot,start_freq_ds_plot))
    xlabel('Start Time: ' + str( start_time.time() ))
    ylabel('Frequency (MHz)')
    title(str(dynamic_spectrum_beam))

    time_im = start_time + datetime.timedelta(seconds = i*averaging_time)
    x = np.zeros( len(freq) ) + matplotlib.dates.date2num( time_im )
    pl.plot( x, freq, '-w' )
    pl.xlim(xmin1, xmax1)
    pl.ylim(ymin1, ymax1)
    pl.colorbar()
    # reversing y axis
    ax1.set_ylim(ax1.get_ylim()[::-1])

    #setting x axis as a time axis
    ax1.xaxis_date()
    ax1.xaxis.set_major_locator(dates.MinuteLocator())
    ax1.xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S'))
    ax1.xaxis.set_major_locator( MaxNLocator(nbins = 7) )


    # coordinate transformation

    solar_x, solar_y = coordinate_transformation(ra, dec, time_im)

    # finding min/max for plotting

    xmin = np.min(solar_x)
    xmax = np.max(solar_x)
    ymin = np.min(solar_y)
    ymax = np.max(solar_y)


    '''
    plotting images for each frequency slice
    '''
    ################################ 1
    INT = intensity2[:,i:i+1].flatten()

    # Size of regular grid
    ny, nx = 200, 200

    # Generate a regular grid to interpolate the data.
    xi = np.linspace(xmin, xmax, nx)
    deltax = xi[1] - xi[0]
    yi = np.linspace(ymin, ymax, ny)
    deltay = yi[1] - yi[0]
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate using delaunay triangularization
    zi = matplotlib.mlab.griddata(solar_x,solar_y,INT,xi,yi,interp='linear')#interp='nn'

    ax2 = pl.subplot2grid((2,3), (1,0))
    pl.pcolormesh(xi,yi,zi, vmin = zmin, vmax = zmax, norm = LogNorm() )
    pl.xlabel('X (arcmin)')
    pl.ylabel('Y (arcmin)')
    pl.title( str(start_freq2) + '-' + str(end_freq2) + ('MHz, Time: ') + str( time_im.time() ) )

    # plots the solar limb
    SUN = pl.Circle((0,0), radius = 16., color = 'y', fc ='none')

    ax2.add_patch( SUN )
    ax2.add_patch( SUN )


    #print zi.shape, xmin, ymin,deltax, deltay, np.min(zi), np.max(zi)
    zi = np.asarray( zi )

    # save image as fits file
    filename = format( start_min, '.0f' ) + 'min' + '_' + str(start_freq2) + '-' +  str(end_freq2) + 'MHz_' + str(i).zfill(4) + '.fits'
    make_fits( filename, zi, xmin, ymin, deltax, deltay, str( time_im.date() ) + ' ' +  str( time_im.time() ) )


    ################################ 2
    INT1 = intensity1[:,i:i+1].flatten()

    # Interpolate using delaunay triangularization
    zi = matplotlib.mlab.griddata(solar_x,solar_y,INT1,xi,yi,interp='linear')

    ax3 = pl.subplot2grid((2,3), (1,1))
    pl.pcolormesh(xi,yi,zi, vmin = zmin, vmax = zmax, norm = LogNorm() )
    pl.xlabel('X (arcmin)')
    pl.ylabel('Y (arcmin)')
    pl.title( str(start_freq1) + '-' + str(end_freq1) + ('MHz, Time: ') + str( time_im.time() ) )

    # plots the solar limb
    SUN = pl.Circle((0,0), radius = 16., color = 'y', fc ='none')
    ax3.add_patch( SUN )

    #print zi.shape, xmin, ymin,deltax, deltay, np.min(zi), np.max(zi)
    zi = np.asarray( zi )

    # save image as fits file
    filename = format( start_min, '.0f' ) + 'min' + '_' + str(start_freq1) + '-' +  str(end_freq1) + 'MHz_' + str(i).zfill(4) + '.fits'
    make_fits( filename, zi, xmin, ymin, deltax, deltay, str( time_im.date() ) + ' ' +  str( time_im.time() ) )


    ################################ 3
    INT2 = intensity[:,i:i+1].flatten()

    # Interpolate using delaunay triangularization
    zi = matplotlib.mlab.griddata(solar_x,solar_y,INT2,xi,yi,interp='linear')

    ax4 = pl.subplot2grid((2,3), (1,2))
    pl.pcolormesh(xi,yi,zi, vmin = zmin, vmax = zmax, norm = LogNorm() )
    pl.xlabel('X (arcmin)')
    pl.ylabel('Y (arcmin)')
    pl.title( str(start_freq) + '-' + str(end_freq) + ('MHz, Time: ') + str( time_im.time() ) )

    # plots the solar limb
    SUN = pl.Circle((0,0), radius = 16., color = 'y', fc ='none')
    ax4.add_patch( SUN )

    #print zi.shape, xmin, ymin,deltax, deltay, np.min(zi), np.max(zi)
    zi = np.asarray( zi )

    # save image as fits file
    filename = format( start_min, '.0f' ) + 'min' + '_' + str(start_freq) + '-' +  str(end_freq) + 'MHz_' + str(i).zfill(4) + '.fits'
    make_fits( filename, zi, xmin, ymin, deltax, deltay, str( time_im.date() ) + ' ' +  str( time_im.time() ) )


    pl.savefig(str(i)+'.png')
    print( i )
    close()



