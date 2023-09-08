import sys
import csv
import time
from matplotlib import pyplot as plt
import numpy as np
import scipy.interpolate as interpol
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
import fnmatch

def line(x,m,b):
	return m*x + b

def gaussianDataFilter(x,y,sigma):
	y_filtered = []
	
	for i in range(0,len(x)):
		Filter = (1./(sigma*np.sqrt(2.*np.pi))) * np.exp(-.5*((x-x[i])/sigma)**2.)
		Filter = Filter/np.sum(Filter)
		y_filtered.append(np.sum(Filter * y)/np.sum(Filter))
	
	return x, np.array(y_filtered)

def energyUniaxial(phi,h,theta):
	gamma = K*np.sin(theta - phi)**2. - 2*h_grid*np.cos(phi)
	return E
	
def energyCubic(phi,h,theta):
	#E = K*np.sin(theta - phi + np.pi/4)**2. * np.cos(theta - phi + np.pi/4)**2. - h*H*M*np.cos(phi)
	gamma = np.sin(theta - phi + np.pi/4)**2. * np.cos(theta - phi + np.pi/4)**2. - 2*h*np.cos(phi)
	return gamma

def gaussianDataFilter(x,y,sigma):
	y_filtered = []
	
	for i in range(0,len(x)):
		Filter = (1./(sigma*np.sqrt(2.*np.pi))) * np.exp(-.5*((x-x[i])/sigma)**2.)
		Filter = Filter/np.sum(Filter)
		y_filtered.append(np.sum(Filter * y)/np.sum(Filter))
	
	return x, np.array(y_filtered)
	
def modelHysteresis(energyFunc,N,Hk,theta,err,H1Data,H2Data,plot):
	phi = np.linspace(0,2*np.pi,N)
	h = np.linspace(-10,10,N)
	phi_grid,h_grid = np.meshgrid(phi,h)
	
	#mu_0 = 4*np.pi*1e-7
	#Hk = 2*K/(mu_0*M)
	h1Data = H1Data/Hk
	h2Data = H2Data/Hk

	E = energyFunc(phi_grid,h_grid,theta)
	
	E_phi_deriv = np.zeros(E.shape)
	E_phi_deriv2 = np.zeros(E.shape)

	for i in range(0,N):
		E_phi_deriv[i] = np.gradient(E[i])/(phi[1] - phi[0])
		E_phi_deriv2[i] = np.gradient(E_phi_deriv[i])/(phi[1] - phi[0])
	
	minMask = np.abs(E_phi_deriv) < err
	stableMinMask = E_phi_deriv2[minMask] > 0

	m1_grid_min = np.cos(phi_grid[minMask][stableMinMask])[np.cos(phi_grid[minMask][stableMinMask]) > 0.]
	m2_grid_min = np.cos(phi_grid[minMask][stableMinMask])[np.cos(phi_grid[minMask][stableMinMask]) < 0.]

	h1_grid_min = h_grid[minMask][stableMinMask][np.cos(phi_grid[minMask][stableMinMask]) > 0.]
	h2_grid_min = h_grid[minMask][stableMinMask][np.cos(phi_grid[minMask][stableMinMask]) < 0.]
	
	h1_grid_min = h1_grid_min[m1_grid_min > .4]
	m1_grid_min = m1_grid_min[m1_grid_min > .4]
	h2_grid_min = h2_grid_min[m2_grid_min < -.4]
	m2_grid_min = m2_grid_min[m2_grid_min < -.4]
	
	sigma = .05
	
	"""
	if theta > 63*np.pi/180:
		sigma = .1
	else:
		sigma = .5
	"""
	h1_filtered,m1_filtered = gaussianDataFilter(h1_grid_min,m1_grid_min,sigma)
	h2_filtered,m2_filtered = gaussianDataFilter(h2_grid_min,m2_grid_min,sigma)

	#plt.plot(h1_filtered,m1_filtered,color = 'black')
	#plt.plot(h2_filtered,m2_filtered,color = 'black')
	#plt.scatter(h1_grid_min,m1_grid_min,color = 'blue')
	#plt.scatter(h2_grid_min,m2_grid_min,color = 'red')
	#plt.show()
	
	#plt.scatter(h1_grid_min[m1_grid_min > m1_filtered],m1_grid_min[m1_grid_min > m1_filtered],color = 'blue')
	#plt.scatter(h2_grid_min[m2_grid_min < m2_filtered],m2_grid_min[m2_grid_min < m2_filtered],color = 'red')
	#plt.show()

	filterMask1 = m1_grid_min > m1_filtered
	filterMask2 = m2_grid_min < m2_filtered

	h1_filtered,m1_filtered = gaussianDataFilter(h1_grid_min[filterMask1],m1_grid_min[filterMask1],sigma)
	h2_filtered,m2_filtered = gaussianDataFilter(h2_grid_min[filterMask2],m2_grid_min[filterMask2],sigma)

	m1_filtered = m1_filtered[h1_filtered.argsort()]
	h1_filtered = h1_filtered[h1_filtered.argsort()]
	
	duplicateMask = np.zeros_like(h1_filtered,dtype = bool)
	duplicateMask[np.unique(h1_filtered,return_index = True)[1]] = True

	m1_filtered = m1_filtered[duplicateMask]
	h1_filtered = h1_filtered[duplicateMask]
	
	duplicateMask = np.zeros_like(h2_filtered,dtype = bool)
	duplicateMask[np.unique(h2_filtered,return_index = True)[1]] = True

	m2_filtered = m2_filtered[duplicateMask]
	h2_filtered = h2_filtered[duplicateMask]
	
	m2_filtered = m2_filtered[h2_filtered.argsort()]
	h2_filtered = h2_filtered[h2_filtered.argsort()]

	tck1 = interpol.splrep(h1_filtered,m1_filtered)
	m1Spline = interpol.splev(h1Data,tck1)
	
	tck2 = interpol.splrep(h2_filtered,m2_filtered)
	m2Spline = interpol.splev(h2Data,tck2)
	
	#plt.plot(h1Data,m1Spline,color = 'black')
	#plt.plot(h2Data,m2Spline,color = 'black')
	#plt.scatter(h1_filtered,m1_filtered,color = 'blue')
	#plt.scatter(h2_filtered,m2_filtered,color = 'red')
	#plt.show() 
	
	if plot:
		return Hk * h1_grid_min[filterMask1], Hk * h2_grid_min[filterMask2], m1_grid_min[filterMask1], m2_grid_min[filterMask2]
	else:
		return Hk * h1Data, Hk * h2Data, m1Spline, m2Spline



def hysteresisResiduals(func,args,N,theta,H,hData,mData,mErrorData,axis,n,sigma,plot):
	#function that calculates the residuals between a thin film magnetic
	#sample's m-h loop and the theoretical one calculated by
	#hysteresis(...), which uses the Stoner-Wohlfarth Model
	
	"""
	arguments:
		func: the function to use for the derivatives of the energy
			functional that hysteresis() will use

		args: an array of all arguments needed by func except for
			phi
		
		N: The number of points to sample along critical curve

		theta: the angle between the easy axis of the sample and the
			sweep field to be used in hysteresis(...)

		H: the field everything is normalized by for the critical 
			curve calculation

		dataFile: the file with the experimental data of the 
				hysteresis loop
	
		n: the number of points you want in the residuals return

		sigma: the sigma value of the gaussian filter the data 
			is to be run through

		plot: boolean that determines whether or not we plot 
			all steps of fitting process

	returns:
		
		residuals: an array of the residuals of length n
	"""
	
	global M_SAT

	mu_0 = 4*np.pi*1e-7

	print('the fit parameters are currently: ')
	print('\tH: ',H)
	print('\ttheta: ',theta)
	
	hData = np.array(hData)*79.57747/1000. #converting from Oe to kA/m
	mData = np.array(mData)
	mErrorData = np.array(mErrorData)
	sort = hData.argsort()
	hData = hData[sort]
	mData = mData[sort]
	mErrorData = mErrorData[sort]
	
	
	#Need to remove a linear background from the data	
	n = int(.1*len(hData))
	params, pcov = curve_fit(line,hData[-n:],mData[-n:])
	slopeFit1 = params[0]
	b1 = params[1]
	params, pcov = curve_fit(line,hData[:n],mData[:n])
	slopeFit2 = params[0]
	b2 = params[1]
	b = (b1 + b2)/2.
	slopeFit = (slopeFit1 + slopeFit2)/2.
	
	mData = mData - line(hData,slopeFit,b)
	
	
	#hysteresis(...) could return either 4 or 2 arrays depending on
	#whether we actually get a loop or just a curve 
		
	if axis == 'easy':
		h1Data = hData[mData > 0.]
		h2Data = hData[mData < 0.]
		m1Data = mData[mData > 0.]
		m2Data = mData[mData < 0.]
		m1ErrorData = mErrorData[mData > 0.]
		m2ErrorData = mErrorData[mData < 0.]
	
		m1Sat = max(np.abs(m1Data))
		m2Sat = max(np.abs(m2Data))
		m1Data = m1Data/m1Sat
		m2Data = m2Data/m2Sat

	elif axis == 'hard':
		#write a gaussian filter for the entire hysteresis loop,
		#anything below it belongs in the lower branch and
		#anything abve it belongs in the upper branch
		print('I haven\'t coded this yet...') #Because I didn't need to...

	#sometimes the derivative method of separating the branches
	#doesn't entirely work and you get outliers in the data
	#this just removes any point that's more than 6-sigma away
	#from other points 
	
	h1DataSymm = np.concatenate((h1Data,-h2Data))
	m1DataSymm = np.concatenate((m1Data,-m2Data))
	h2DataSymm = np.concatenate((-h1Data,h2Data))
	m2DataSymm = np.concatenate((-m1Data,m2Data))
	
	h1Data = h1DataSymm
	m1Data = m1DataSymm
	h2Data = h2DataSymm
	m2Data = m2DataSymm
		
	#plt.scatter(h1DataSymm,m1DataSymm)
	#plt.scatter(-h2Data,-m2Data)
	#plt.scatter(-h1Data,-m1Data)
	#plt.scatter(h2DataSymm,m2DataSymm)
	#plt.show()
		
	h1Diff = np.abs(h1Data[:len(h1Data)-1] - h1Data[1:])
	h2Diff = np.abs(h2Data[:len(h2Data)-1] - h2Data[1:])
	m1_mask = h1Diff < 6.*np.average(h1Diff)
	m2_mask = h2Diff < 6.*np.average(h2Diff)
	m1_maskList = list(m1_mask)
	m2_maskList = list(m2_mask)
	m1_maskList.append(False)
	m2_maskList.append(False)
	m1_mask = np.array(m1_maskList,dtype = bool)
	m2_mask = np.array(m2_maskList,dtype = bool)
	
	m1Data = m1Data[m1_mask]
	h1Data = h1Data[m1_mask]	
	m2Data = m2Data[m2_mask]
	h2Data = h2Data[m2_mask]
	
	#getting rid of points beyond 6-sigma away from the
	#other points in the m-direction
	
	m1Diff = np.abs(m1Data[:len(m1Data)-1] - m1Data[1:])
	m2Diff = np.abs(m2Data[:len(m2Data)-1] - m2Data[1:])
	m1_mask = m1Diff < 4.*np.average(m1Diff)
	m2_mask = m2Diff < 4.*np.average(m2Diff)
	m1_maskList = list(m1_mask)
	m2_maskList = list(m2_mask)
	m1_maskList.append(False)
	m2_maskList.append(False)
	m1_mask = np.array(m1_maskList,dtype = bool)
	m2_mask = np.array(m2_maskList,dtype = bool)

	m1Data = m1Data[m1_mask]
	h1Data = h1Data[m1_mask]
	m2Data = m2Data[m2_mask]
	h2Data = h2Data[m2_mask]
		
	#Running the data through a gaussian filter
	h1Data, m1Data = gaussianDataFilter(h1Data,m1Data,sigma)
	h2Data, m2Data = gaussianDataFilter(h2Data,m2Data,sigma)
	
	#the data needs to be going from right to left for the 
	#dependent variable for the cubic spline to work
	sort1 = h1Data.argsort()
	h1DataSort = h1Data[sort1]
	m1DataSort = m1Data[sort1]
	sort2 = h2Data.argsort()
	h2DataSort = h2Data[sort2]
	m2DataSort = m2Data[sort2]
	"""
	#sort1 = hApplied1.argsort()
	#hApplied1 = hApplied1[sort1]
	#m1 = m1[sort1]
	#sort2 = hApplied2.argsort()
	#hApplied2 = hApplied2[sort2]
	#m2 = m2[sort2]
	"""		
	m1DataGrad = np.abs(np.gradient(m1DataSort))
	m2DataGrad = np.abs(np.gradient(m2DataSort))

	m1GradMask = m1DataGrad[int(.2*len(m1DataGrad)):int(.8*len(m1DataGrad))] > 15.*np.average(m1DataGrad)
	m2GradMask = m2DataGrad[int(.2*len(m2DataGrad)):int(.8*len(m2DataGrad))] > 15.*np.average(m2DataGrad)
		
	#This code is to stop the data from jumping abruptly
	#like when the branches split and one doesn't have
	#as large a sweep field as the other so the 
	#gaussian filtered data jumps
	if len(h1DataSort[int(.2*len(m1DataGrad)):int(.8*len(m1DataGrad))][m1GradMask]):
		m1_mask = h1DataSort > max(h1DataSort[int(.2*len(m1DataGrad)):int(.8*len(m1DataGrad))][m1GradMask])
		h1DataSort = h1DataSort[m1_mask]
		m1DataSort = m1DataSort[m1_mask]

	if len(h2DataSort[int(.2*len(m2DataGrad)):int(.8*len(m2DataGrad))][m2GradMask]):
		m2_mask = h2DataSort < min(h2DataSort[int(.2*len(m2DataGrad)):int(.8*len(m2DataGrad))][m2GradMask])
		h2DataSort = h2DataSort[m2_mask]
		m2DataSort = m2DataSort[m2_mask]
		#This code is to stop the spline and modeled data
		#from going beyond the applied sweep field data
		#on the ends of the hysteresis loop
	"""
	if(hApplied1[0] < h1DataSort[0]):
		m1 = m1[hApplied1 > h1DataSort[0]]
		hApplied1 = hApplied1[hApplied1 > h1DataSort[0]]
	
	if(hApplied2[-1] > h2DataSort[-1]):
		m2 = m2[hApplied2 < h2DataSort[-1]]
		hApplied2 = hApplied2[hApplied2 < h2DataSort[-1]]
	"""	
	#sometimes points are directly over each other which causes 
	#the cubic spline to return nothing but NaNs so this removes
	#them from the data
	duplicateMask1 = np.zeros_like(h1DataSort, dtype = bool)
	duplicateMask1[np.unique(h1DataSort, return_index = True)[1]] = True
	duplicateMask2 = np.zeros_like(h2DataSort, dtype = bool)
	duplicateMask2[np.unique(h2DataSort, return_index = True)[1]] = True
	
	#main(energyCubic,5000,58000,70*np.pi/180,1e-3,np.linspace(0,100000,1000),np.linspace(-100000,0,1000))
	solution = modelHysteresis(func,5000,H,theta,1e-3,h1DataSort[duplicateMask1],h2DataSort[duplicateMask2],False)

	if len(solution) == 4:
		hApplied1 = solution[0]
		hApplied2 = solution[1]
		mApplied1 = solution[2]
		mApplied2 = solution[3]
		
		m1SatNorm = np.abs(np.mean(m1DataSort[duplicateMask1][int(.9*len(m1DataSort[duplicateMask1])):]))
		m1DataSort = m1DataSort/m1SatNorm
		#print(hApplied1)
		#print(h1DataSort[duplicateMask1])
		
		#mask = np.round(hApplied1,4) == np.round(h1DataSort[duplicateMask1],4)
		#if(len(hApplied1[mask]) == len(h1DataSort[duplicateMask1])):
		residuals1 = m1DataSort[duplicateMask1] - mApplied1
		"""
		else:
		#NOTE: this is untested...
			hApplied2Filtered = []
			extraResiduals = []
			for i in range(0,len(hApplied2)):
				mask2 = np.round(hApplied2[i],4) == np.round(h1DataSort[duplicateMask1][~mask],4)
				if(len(h1DataSort[duplicateMask1][~mask][mask2]) > 0):
					hApplied2Filtered.append(hApplied2[i])
					extraResiduals.append(-(mApplied2[i] - m1DataSort[duplicateMask1][~mask][mask2]))
				
			hApplied2Filtered = np.array(hApplied2Filtered)
			extraResiduals = np.array(extraResiduals)
	
			sort1 = hApplied2Filtered.argsort()
			hApplied2Filtered = hApplied2Filtered[sort1]
			extraResiduals = extraResiduals[sort1]
			#print(len(m2DataSort[duplicateMask2]),len(mApplied1))
			residuals = (m1DataSort[duplicateMask1][mask] - mApplied1[mask])

			print(residuals)
			residuals1 = np.concatenate(np.abs(residuals, extraResiduals))
		"""	
		

		if plot:
			#plt.scatter(hApplied1*mu_0*1e3,mApplied1,color = 'blue')
			plt.scatter(h1DataSort[duplicateMask1]*mu_0*1e3,m1DataSort[duplicateMask1],color = 'red',s = 10)

	elif len(solution) == 2:
		print('really hope this doesnt happen... (1)')
		hApplied = solution[0]
		mApplied = solution[1]
		
	solution = modelHysteresis(func,5000,H,theta,1e-3,h1DataSort[duplicateMask1],h2DataSort[duplicateMask2],False)
	
	if len(solution) == 4:
		hApplied1 = solution[0]
		hApplied2 = solution[1]
		mApplied1 = solution[2]
		mApplied2 = solution[3]
		
		m2SatNorm = np.abs(np.mean(m2DataSort[duplicateMask2][:int(.1*len(m2DataSort[duplicateMask2]))]))
		
		m2DataSort = m2DataSort/m2SatNorm
		
		M_SAT = np.mean([m1Sat/np.abs(m1SatNorm),m2Sat/np.abs(m2SatNorm)])
		
		#mask = np.round(hApplied2,4) == np.round(h2DataSort[duplicateMask2],4)
		#if(len(hApplied2[mask]) == len(h2DataSort[duplicateMask2])):
		residuals2 = (m2DataSort[duplicateMask2] - mApplied2)
		"""
		else:
		#NOTE: this is untested...
			hApplied1Filtered = []
			extraResiduals = []
			for i in range(0,len(hApplied1)):
				mask2 = np.round(hApplied1[i],4) == np.round(h2DataSort[duplicateMask2][~mask],4)
				if(len(h2DataSort[duplicateMask2][~mask][mask2]) > 0):
					hApplied1Filtered.append(hApplied1[i])
					extraResiduals.append(-(mApplied1[i] - m2DataSort[duplicateMask2][~mask][mask2]))
				
			hApplied1Filtered = np.array(hApplied1Filtered)
			extraResiduals = np.array(extraResiduals)
	
			sort1 = hApplied1Filtered.argsort()
			hApplied1Filtered = hApplied1Filtered[sort1]
			extraResiduals = extraResiduals[sort1]
			
			residuals = (m2DataSort[duplicateMask2][mask] - mApplied2[mask])

			residuals2 = np.concatenate(np.abs(residuals, extraResiduals))
		"""	
		
		if plot:
			
			#plt.scatter(hApplied2*mu_0*1e3,mApplied2,color = 'blue')
			plt.scatter(h2DataSort[duplicateMask2]*mu_0*1e3,m2DataSort[duplicateMask2],color = 'red',label = 'VSM Data', s = 10)
		
			solution = modelHysteresis(func,5000,H,theta,1e-3,np.linspace(min(h1DataSort[duplicateMask1]),max(h1DataSort[duplicateMask1]),1000),np.linspace(min(h2DataSort[duplicateMask2]),max(h2DataSort[duplicateMask2]),1000),True)
			#solution = modelHysteresis(np.linspace(-5.*H,5.*H,5000),theta,hBias,H,False)
			h1 = solution[0]
			h2 = solution[1]
			m1 = solution[2]
			m2 = solution[3]

			plt.plot(h1*mu_0*1e3,m1,color = 'blue',label = 'Stoner-Wohlfarth Model')
			plt.plot(h2*mu_0*1e3,m2,color = 'blue')
			plt.vlines(-.0357,-.889,.651,colors = 'blue',linestyles = 'dashed')
			plt.vlines(.0357,-.651,.889,colors = 'blue',linestyles = 'dashed')
			plt.xlabel(r'$\mu_0$H (T)',fontsize = 16)
			plt.ylabel(r'$\hat{h} \cdot \vec{M}/M_S$ (unitless)',fontsize = 16)
			plt.legend()
			plt.show()

	elif len(solution) == 2:
		print('really hope this doesnt happen... (2)')
		hApplied = solution[0]
		mApplied = solution[0]

	ret = np.concatenate((residuals1,residuals2))

	if plot:
		error = 0.
		howMany = 0
		for i in range(0,len(hApplied1)):
			mask1 = np.round(hApplied1[i],4) == np.round(h1DataSort[duplicateMask1],4)
			if (len(h1DataSort[duplicateMask1][mask1]) == 1):
				error = error + np.abs(mApplied1[i] - m1DataSort[duplicateMask1][mask1][0]) / mApplied1[i]
				howMany = howMany + 1
		for i in range(0,len(hApplied2)):
			mask2 = np.round(hApplied2[i],4) == np.round(h2DataSort[duplicateMask2],4)
			if (len(h2DataSort[duplicateMask2][mask2]) == 1):
				error = error + np.abs(mApplied2[i] - m2DataSort[duplicateMask2][mask2][0]) / mApplied2[i]
				howMany = howMany + 1
		print('The percent error is: ', error/(howMany))
#		print(h1DataSort[duplicateMask1])
#		print(hApplied1)
#		print(mApplied1)
#		error = np.sum(np.abs(mApplied1 - m1DataSort[duplicateMask1][mask1]) / mApplied1) + 
	
	print('The current sum of the residuals is: ', np.sum(ret))
	print('\n')

	return ret
	
def hysteresisFitting(func,args,N,dataFile,axis,n):
	#just reading in the data from the file, 
	#might need to change this later

	testList = [dataFile]
	testFileType = fnmatch.filter(testList, '*.dat.txt')
	if len(testFileType) == 1:
		with open(dataFile,'r') as myFile:
			read = csv.reader(myFile,delimiter = ',')
			TempData = []
			hData = []
			mData = []
			mErrorData = []
			for row in read:
				try:
					TempData.append(float(row[0]))
					hData.append(float(row[1]))
					mData.append(float(row[2]))
					mErrorData.append(float(row[3]))
				except:
					print('Skipping the file header...')
	
	else:
		with open(dataFile,'r') as myFile:
			read = csv.reader(myFile,delimiter = ',')
			Comment = []
			Time = []
			TempData = []
			hData = []
			mData = []
			mErrorData = []
			for row in read:
				try:
					Comment.append(str(row[0]))
					Time.append(float(row[1]))
					TempData.append(float(row[2]))
					hData.append(float(row[3]))
					mData.append(float(row[4]))
					mErrorData.append(float(row[5]))
				except:
					print('Skipping the file header...')
			
	
	#a functino that uses a least squares algorithm to fit hysteresis
	#loop data to a loop calculated by hysteresis(...) with the 
	#Stoner-Wohlfarth model and the critical curve of the magnetic 
	#energy
	#def hysteresisResiduals(func,args,N,theta,H,hData,mData,mErrorData,axis,n,sigma,plot):
	
	#solution = least_squares(lambda y: hysteresisResiduals(func,args,N,y[0],y[1],hData,mData,mErrorData,axis,n,3,False),[1.35,105], method = 'dogbox', diff_step = .1, x_scale = [.05,5],bounds = [[5*np.pi/18,80.],[8.*np.pi/18.,250]]) #NOTE: this breaks when hBias_guess = 1.
	solution = least_squares(lambda y: hysteresisResiduals(func,args,N,45.*np.pi/180.,y,hData,mData,mErrorData,axis,n,3,False),[75], method = 'dogbox', diff_step = .1, x_scale = [5],bounds = [[50.],[350]]) #NOTE: this breaks when hBias_guess = 1.
	
	hBias = 0
	
	theta = 90.*np.pi/180.
	#theta = solution.x[0]
	#hBias = solution.x[1]
	#H = solution.x[1]
	H = solution.x[0]
	#slope = solution.x[2]

	print('\n')
	print('The final fit parameters for the hysteresis loop are:')
	print('\thBias: ', hBias, ' (unitless)')
	print('\tH: ', H, ' (kA/m)')
	print('\ttheta: ', theta, ' (rads)')
	print('\tK_u: ',4*np.pi*1e-7*H*M_SAT/2. * 1e9,' (nJ)')
	print('\tM_SAT: ', M_SAT/1000., ' (Am^2)')
	
	"""
	hBias = 0.#.3
	H = 24.447#.5
	theta = .64717#.01
	"""
	#theta = 7.*np.pi/18.
	
	hysteresisResiduals(func,args,N,theta,H,hData,mData,mErrorData,axis,n,3.,True)
	hysteresisResiduals(func,args,N,theta,1.2*H,hData,mData,mErrorData,axis,n,3.,True)

hysteresisFitting(energyCubic,[],5000,sys.argv[1],'easy',3500)
