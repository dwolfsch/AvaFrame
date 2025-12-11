"""
Functions that are used specifically for modeling debris flows (within DebrisFrame).
"""

# Load modules
import logging
import numpy as np
import copy

import avaframe.com1DFA.DFAfunctionsCython as DFAfunC
import avaframe.com1DFA.particleTools as particleTools
import avaframe.in3Utils.geoTrans as geoTrans
import avaframe.com1DFA.com1DFA as com1DFA
import avaframe.in2Trans.shpConversion as shpConv
from avaframe.in1Data import getInput as gI

# create local logger
# change log level in calling module to DEBUG to see log messages
log = logging.getLogger(__name__)


def initializeTimeDepRelease(cfg, inputSimLines, particles, fields, dem, zPartArray0, t, atol=1e-05):
    """
    Update particles with "new" particles initialised by a time dependent.

    Parameters
    ---------
    cfg: configparser
        configuration settings
    inputSimLines : dict
        dictionary with input data dictionaries (releaseLine,...)
    particles : dict
        particles dictionary at t that are in the flow already
    fields: dict
        fields dictionary at t
    dem: dict
        dictionary with info on DEM data
    zPartArray0: dict
        dictionary containing z - value of particles at timestep 0
    t: float
        timestep of iteration
    atol: float
        look for matching time steps with atol tolerance - default is atol=1.e-5

    Returns
    ---------
    particles: dict
        particles dictionary at t including the new released particles
    fields: dict
        updated fields dictionary at t including the new released particles
    zPartArray0: dict
        dictionary containing z - value of particles at timestep 0
    """
    timeDepRelValues = inputSimLines["releaseLine"]["timeDepRelValues"]
    if np.isclose(t, timeDepRelValues["timeStep"], atol=atol, rtol=0).any():
        iTup = np.where(np.isclose(t, timeDepRelValues["timeStep"], atol=atol, rtol=0))
        # iTup is a tuple containing an array with one value in the first position, so we can extract the index:
        i = iTup[0].item()
        log.info(
            "add release at timestep: %.2f s with thickness %s m and velocity %s m/s"
            % (t, timeDepRelValues["thickness"][i], timeDepRelValues["velocity"][i])
        )
        # similar workflow to secondary release!
        particles, zPartArray0 = addReleaseParticles(
            cfg,
            particles,
            inputSimLines,
            timeDepRelValues["thickness"][i],
            timeDepRelValues["velocity"][i],
            dem,
            zPartArray0,
        )
        particles = DFAfunC.getNeighborsC(particles, dem)
        # update fields (compute grid values)
        if fields["computeTA"]:
            particles = DFAfunC.computeTrajectoryAngleC(particles, zPartArray0)
        particles, fields = DFAfunC.updateFieldsC(cfg["GENERAL"], particles, dem, fields)

    return particles, fields, zPartArray0


def addReleaseParticles(cfg, particles, inputSimLines, thickness, velocityMag, dem, zPartArray0):
    """
    add new particles initialized by a time dependent to particles that are in the flow already

    Parameters
    ---------
    cfg: configparser object
        configuration settings
    particles : dict
        particles dictionary at t that are in the flow already
    inputSimLines : dict
        dictionary with input data dictionaries (releaseLine,...)
    thickness: float
        thickness of current release
    velocityMag: float
        velocity of current release
    dem: dict
        dictionary with info on DEM data
    zPartArray0: numpy array
        z - value of particles at timestep 0

    Returns
    ---------
    particles: dict
        particles dictionary at t including the current released particles
    zPartArray0: dict
        dictionary containing z - value of particles at timestep 0
    """
    relLine = inputSimLines["releaseLine"]
    relLine["header"] = dem["originalHeader"].copy()
    relLine = geoTrans.prepareArea(
        relLine,
        dem,
        np.sqrt(2),
        thList=[thickness],
        combine=True,
        checkOverlap=False,
    )

    # check if already existing particles are within the release polygon
    # it's possible that there are still a few particles in the polygon with low velocities
    # TODO: could think of a threshold of number of particles that are still allowed in the polygons?
    mask = geoTrans.getParticlesInPolygon(particles, relLine, cfg["GENERAL"].getfloat("thresholdPointInRel"))
    if np.sum(mask) > 0:
        # if there is at least one particle within the polygon (including the buffer):
        message = (
                "Already existing particles are within the release polygon, which can cause numerical instabilities (at timestep: %02f s)"
            % (particles["t"] + particles["dt"])
        )
        # timestep in particles is not updated yet
        log.error(message)
        raise ValueError(message)

    particlesRelease = com1DFA.initializeParticles(
        cfg["GENERAL"],
        relLine,
        dem,
    )
    particlesRelease = DFAfunC.updateInitialVelocity(cfg["GENERAL"], particlesRelease, dem, velocityMag)

    particles = particleTools.mergeParticleDict(particles, particlesRelease)
    # save initial z position for travel angle computation
    zPartArray0 = np.append(zPartArray0, copy.deepcopy(particlesRelease["z"]))
    return particles, zPartArray0


def checkTimeDepRelease(timeDepRelValues, timeDepRelCsv, cfgGen):
    """
    check if time dependent (general) release values satisfies the following requirements:
    - release - timesteps are unique
    - provided release - thickness is larger than zero
    - the release - timesteps are not too close (that the particle density becomes too high)

    Parameters
    -----------
    timeDepRelCsv: str
        directory to csv table containing time dependent release values
    timeDepRelValues: dict
        contains time dependent release values: timestep, thickness, velocity
    cfgGen: configparser Object
        GENERAL section of the configuration file
    """
    # check if timesteps are unique
    timeStepUnique = np.unique(timeDepRelValues["timeStep"])
    if timeStepUnique.ndim == 0:
        if timeStepUnique != timeDepRelValues["timeStep"]:
            message = "The provided time dependent release time steps in %s are not unique" % (timeDepRelCsv)
            log.error(message)
            raise ValueError(message)
    elif len(timeStepUnique) != len(timeDepRelValues["timeStep"]):
        message = "The provided time dependent release timesteps in %s are not unique" % (timeDepRelCsv)
        log.error(message)
        raise ValueError(message)

    # check that release thickness > 0
    for th in timeDepRelValues["thickness"]:
        if th <= 0:
            message = "For every release time step a thickness > 0 needs to be provided in %s" % (
                timeDepRelCsv
            )
            log.error(message)
            raise ValueError(message)

    # check if a timestep = 0 is provided, when no REL area is used
    if cfgGen.getboolean("timeDependentRelease"):
        if 0 not in timeStepUnique:
            message = (
                    "If release is time dependent, a thickness needs to be provided for  time step 0 s in %s"
                    % (timeDepRelCsv)
            )
            log.error(message)
            raise ValueError(message)


def prepareTimeDepRelLine(inputSimFiles, releaseLine, cfg):
    """
    read time dependent release values

    Parameters
    ----------
    inputSimFiles : dict
        dictionary containing
        - timeDepRelCsv: str, path to time dependent release values (csv-)file
    releaseLine: dict
        contains informations of release line
    cfg: configparser object
        configuration for simType
    demOri : dict
        dictionary with dem info (header original origin), raster data correct mesh cell size

    Returns
    -------
    releaseLine: dict
        added time dependent release values: timeStep, thickness, velocity
    """

    try:
        releaseLine["values"] = gI.getTimeDepRelCsv(inputSimFiles["timeDepRelCsv"])
        releaseLine["thicknessSource"] = ["csv file"]
    except:
        message = "No time dependent release csv file found"
        log.error(message)
        raise FileNotFoundError(message)

    checkTimeDepRelease(releaseLine["values"], inputSimFiles["timeDepRelCsv"], cfg["GENERAL"])

    return releaseLine


def checkTravelledDistance(cfgGen, timeDepRelValues, timeDepRelCsv):
    """
    not used at the moment (related to timeStepDistance in the configuration file)!
    check if time steps of time dependent release are not to close that the particle density becomes too high
    check that particles moved out of release area before new particles are initialized
    time between release time steps
    first timestep is skipped since this is always ok.

    Parameters
    -----------
    cfgGen: configparser
        configuration settings, section "GENERAL"
    timeDepRelValues: dict
        contains time dependent release values: timestep, thickness, velocity
    timeDepRelCsv: str
        directory to csv table containing time dependent release values
    """
    timeStepUnique = np.unique(timeDepRelValues["timeStep"])
    if timeStepUnique.ndim > 0:
        relDT = np.append(timeDepRelValues["timeStep"], 0) - np.append(0, timeDepRelValues["timeStep"])
        vel = np.where(np.array(timeDepRelValues["velocity"]) > 0, np.array(timeDepRelValues["velocity"]), 1)
        distance = vel[:-1] * relDT[1:-1]

        if np.any(distance < cfgGen.getfloat("timeStepDistance")):
            message = "Please select timesteps with greater spacing in %s." % (timeDepRelCsv)
            # TODO: error or warning?
            log.error(message)
            raise ValueError(message)
