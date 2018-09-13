import os
import numpy


path = os.path.dirname(os.path.realpath(__file__))


def get_mappingMatrix(ao_system):

	if ao_system=='CANARY':
		mm = numpy.load(path+'/canary/mm_canary.npy')
		mmc = numpy.load(path+'/canary/mmc_canary.npy')
		md = numpy.load(path+'/canary/md_canary.npy')
	
	if ao_system=='AOF':
		mm = numpy.load(path+'/aof/mm_aof.npy')
		mmc = numpy.load(path+'/aof/mmc_aof.npy')
		md = numpy.load(path+'/aof/md_aof.npy')
	
	if ao_system=='HARMONI':
		mm = numpy.load(path+'/harmoni/mm_harmoni.npy')
		mmc = numpy.load(path+'/harmoni/mmc_harmoni.npy')
		md = numpy.load(path+'/harmoni/md_harmoni.npy')

	return mm, mmc, md