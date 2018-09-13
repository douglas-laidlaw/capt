
import base.readConfig
this=base.readConfig.init(globals())
wfs_nsubx=7 #Number of subaps
tstep=1/150.#Simulation timestep in seconds (250Hz).
AOExpTime=40.#40 seconds exposure (use --iterations=xxx to modify)
phasesize=16#number of phase pixels per sub-aperture.
npup=wfs_nsubx*phasesize#Number of phase points across the pupil
telDiam=4.2
telSec=1.#Central obscuration
ntel=npup#Telescope diameter in pixels
nAct=wfs_nsubx+1#Number of actuators across the DM
ngsLam=500.#NGS wavelength
lgsLam=589.#LGS wavelength
sciLam=1650.#Science wavelength
lgsAsterismRadius=60.#arcseconds
ngsAsterismRadius=20.#arcseconds
nsci=1
nlgs=0
nngs=2
ndm=this.getVal("ndm",3)
import util.tel
#Create a pupil function
pupil=util.tel.Pupil(npup,ntel/2,ntel/2*telSec/telDiam)

#Create the WFS overview
lgsalt=90000.#90km sodium layer
import util.guideStar
import util.elong
#Create the LGS PSFs (elongated).  There are many ways to do this - this is a simple one.
lgssig=1e6
psf=util.elong.make(spotsize=phasesize*4,nsubx=wfs_nsubx,wfs_n=phasesize,lam=lgsLam,telDiam=telDiam,telSec=telSec,beacon_alt=lgsalt,beacon_depth=10000.,launchDist=0.,launchTheta=0.,pup=pupil,photons=lgssig)[0]

sourceList=[]
wfsDict={}
for i in range(nlgs):#60 arcsec off-axis
    id="%d"%(i+1)
    wfsDict[id]=util.guideStar.LGS(id,wfs_nsubx,lgsAsterismRadius,i*360./nlgs,lgsalt,phasesize=phasesize,minarea=0.5,sig=lgssig,sourcelam=lgsLam,reconList=["recon"],pupil=pupil,launchDist=0,launchTheta=0,lgsPsf=psf)
    sourceList.append(wfsDict[id])
for i in range(nngs):#90 arcsec off-axis
    id="%d"%(i+1+nlgs)
    wfsDict[id]=util.guideStar.NGS(id,wfs_nsubx,ngsAsterismRadius,i*360./nngs,phasesize=phasesize,minarea=0.5,sig=1e6,sourcelam=ngsLam,reconList=["recon"],pupil=pupil)
    sourceList.append(wfsDict[id])
wfsOverview=util.guideStar.wfsOverview(wfsDict)

#Create a Science overview.
import util.sci
sciDict={}
for i in range(nsci):
    id="sci%d"%(i+1)
    sciDict[id]=util.sci.sciInfo(id,i*10.,0.,pupil,sciLam,phslam=sciLam,calcRMS=0)
    sourceList.append(sciDict[id])
sciOverview=util.sci.sciOverview(sciDict)
#Create the atmosphere object and source directions.
from util.atmos import geom,layer,source
atmosDict={}
nlayer=2 #10 atmospheric layer
layerList={"allLayers":["L%d"%x for x in range(nlayer)]}
strList=[0.5]+[0.5/(nlayer-1.)]*(nlayer-1)#relative strength of the layers
hList=(0,12376)#height of the layers
vList=[10., 15.]#velocity of the layers
dirList=[45., 135.]#direction (degrees) of the layers
for i in range(nlayer):
 atmosDict["L%d"%i]=layer(hList[i],dirList[i],vList[i],strList[i],10+i)

l0=25. #outer scale
r0=0.1 #fried's parameter
atmosGeom=geom(atmosDict,sourceList,ntel,npup,telDiam,r0,l0)


#Create the DM object.
from util.dm import dmOverview,dmInfo
import numpy
if ndm>1:
    dmHeight=numpy.arange(ndm)*(hList[-1]/(ndm-1.))
else:
    dmHeight=[0]
dmInfoList=[]
for i in range(ndm):
    dmInfoList.append(dmInfo('dm%dpath'%i,[x.idstr for x in sourceList],dmHeight[i],nAct,minarea=0.1,actuatorsFrom="recon",pokeSpacing=(None if wfs_nsubx<20 else 10),maxActDist=1.5,decayFactor=0.,reconLam=lgsLam,closedLoop=0))
for i in range(nsci):
    #Add the virtual DM projector
    dmInfoList.append(dmInfo('vdmsci%d'%(i+1),["sci%d"%(i+1)],0,nAct,minarea=0.1,actuatorsFrom=["dm%dpath"%x for x in range(ndm)],maxActDist=1.5,decayFactor=0.,reconLam=lgsLam,closedLoop=0,primaryTheta=atmosGeom.sourceTheta("sci%d"%(i+1)),primaryPhi=atmosGeom.sourcePhi("sci%d"%(i+1))))
    #And the physical MOAO DMs
    dmInfoList.append(dmInfo('dmsci%d'%(i+1),["sci%d"%(i+1)],0,nAct,minarea=0.1,actuatorsFrom="vdmsci%d"%(i+1),maxActDist=1.5,decayFactor=0.,reconLam=lgsLam,closedLoop=0))
dmOverview=dmOverview(dmInfoList,atmosGeom)
reconIdStr='recon'

#reconstructor
this.tomoRecon=new()
r=this.tomoRecon
r.rcond=0.05#condtioning value for SVD
r.recontype="pinv"#reconstruction type
r.pokeval=1.#strength of poke
r.gainFactor=1.
r.abortAfterPoke=1#Loop gain
r.computeControl=1#To compute the control matrix after poking
r.reconmxFilename="rmx.fits"#control matrix name (will be created)
r.pmxFilename="pmx.fits"#interation matrix name (will be created)
