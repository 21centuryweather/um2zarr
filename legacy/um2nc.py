# Copyright 2022 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""
Original script (/g/data/access/projects/access/apps/pythonlib/um2netcdf4/2.1/um2netcdf4.py) 
created by Martin Dix.

Modified by Davide Marchegiani at ACCESS-NRI - davide.marchegiani@anu.edu.au

Convert a UM fieldsfile to netCDF
"""

import datetime
import os
import numpy as np
import cf_units
import cftime
import netCDF4
import iris
import iris.util
import iris.coords
import iris.exceptions
import iris.fileformats
import amami
import amami.um_utils as umutils
from amami.um_utils import Stash
from amami.exceptions import AmamiError, UMError
from amami.loggers import LOGGER
from amami.helpers import get_abspath


def get_nc_format(format_arg: str) -> str:
    """Convert format numbers to format strings"""
    nc_formats = {
        1: 'NETCDF4',
        2: 'NETCDF4_CLASSIC',
        3: 'NETCDF3_CLASSIC',
        4: 'NETCDF3_64BIT',
    }
    try:
        return nc_formats[int(format_arg)]
    except ValueError:
        return format_arg


def check_ncformat(ncformat, use64bit):
    """
    Check whether the --64bit option was chosen along with
    the nc format 'NETCDF3_CLASSIC', as they are incompatible.
    """
    if ncformat == 'NETCDF3_CLASSIC' and use64bit:
        LOGGER.error(
            "Chosen netCDF format 'NETCDF3_CLASSIC' is incompatible with"
            "the '--64bit' option, as it does not support 64-bit data."
        )


def add_global_attrs(infile, fid, nohist) -> None:
    """Add global attributes to converted NetCDF file"""
    if not nohist:
        date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        history = f"File {infile} converted with 'amami um2nc' v{amami.__version__} "\
            f"on {date}"
        fid.update_global_attributes({'history': history})
    fid.update_global_attributes({'Conventions': 'CF-1.6'})


def get_heaviside_uv(cubes):
    """Get heaviside_uv field if UM file has it, otherwise return None"""
    for c in cubes:
        if Stash(c.attributes['STASH']).itemcode == 30301:
            return c


def get_heaviside_t(cubes):
    """Get heaviside_t field if UM file has it, otherwise return None"""
    for c in cubes:
        if Stash(c.attributes['STASH']).itemcode == 30304:
            return c


def apply_mask(cube, heaviside, hcrit):
    """
    Apply heaviside function to cube
    """
    # Function must handle case where the cube is defined only on
    # a subset of the levels of the heaviside function
    LOGGER.debug(
        f"Shape | cube: {cube.shape}, heaviside: {heaviside.shape}"
    )
    if cube.shape == heaviside.shape:
        # If the shapes match it's simple
        # Temporarily turn off warnings from 0/0
        with np.errstate(divide='ignore', invalid='ignore'):
            cube.data = np.ma.masked_array(
                cube.data/heaviside.data,
                heaviside.data <= hcrit
            ).astype(np.float32)
    else:
        # Are the levels of c a subset of the levels of the heaviside variable?
        c_p = cube.coord('pressure').points
        h_p = heaviside.coord('pressure').points
        LOGGER.debug(
            f"Levels for masking | cube: {c_p}, heaviside: {h_p}"
        )
        if set(c_p).issubset(h_p):
            # Match is possible
            constraint = iris.Constraint(pressure=c_p)
            h_tmp = heaviside.extract(constraint)
            # Double check they're actually the same after extraction
            if not np.all(c_p == h_tmp.coord('pressure').points):
                raise UMError(
                    "Unexpected mismatch in levels of extracted heaviside function.")

            with np.errstate(divide='ignore', invalid='ignore'):
                cube.data = np.ma.masked_array(
                    cube.data/h_tmp.data,
                    h_tmp.data <= hcrit
                ).astype(np.float32)
        else:
            long_name = Stash(cube.attributes['STASH']).long_name
            msg = f"Unable to match levels of heaviside function to variable {long_name}."
            raise UMError(msg)


def apply_mask_to_pressure_level_field(
    cube,
    stash,
    heaviside_uv,
    heaviside_t,
    hcrit,
):
    """
    Check whether there are any pressure level fields that should be masked
    using heaviside function and mask them.
    """
    itemcode = stash.itemcode
    # Heaviside_uv
    if (30201 <= itemcode <= 30288) or (30302 <= itemcode <= 30303):
        if heaviside_uv:
            LOGGER.info(
                f"Masking field '{stash.long_name}' using heaviside_uv field "
                f"`{Stash(heaviside_uv.attributes['STASH']).long_name}` and "
                f"critical value {hcrit}"
            )
            apply_mask(cube, heaviside_uv, hcrit)
        else:
            LOGGER.warning(
                "Heaviside_uv field needed for masking pressure level data "
                f"is not present. The field '{stash.long_name} -- ITEMCODE:{itemcode}' "
                f"will be skipped.\n"
                "If you still want convert this field without masking, "
                "use the '--nomask' option."
            )
            return False
    # Heaviside_t
    elif 30293 <= itemcode <= 30298:
        if heaviside_t:
            LOGGER.info(
                f"Masking field '{stash.long_name}' using heaviside_t field "
                f"`{Stash(heaviside_t.attributes['STASH']).long_name}` and "
                f"critical value {hcrit}"
            )
            apply_mask(cube, heaviside_t, hcrit)
        else:
            LOGGER.warning(
                "Heaviside_t field needed for masking pressure level data "
                f"is not present. The field '{stash.long_name} -- ITEMCODE:{itemcode}' "
                f"will be skipped.\n"
                "If you still want convert this field without masking, "
                "use the '--nomask' option."
            )
            return False
    return True


def name_cube(cube, stash, simple):
    """
    Assign different name properties to cube
    """
    # Name cube variable
    if simple:
        cube.var_name = f"fld_s{stash.section}{stash.item}"
    elif stash.unique_name:
        cube.var_name = stash.unique_name
    # Cases with max or min
    if cube.var_name:
        if any(m.method == 'maximum' for m in cube.cell_methods):
            cube.var_name += "_max"
        if any(m.method == 'minimum' for m in cube.cell_methods):
            cube.var_name += "_min"
    # The iris name mapping seems wrong for these - perhaps assuming rotated grids?
    if cube.standard_name == 'x_wind':
        cube.standard_name = 'eastward_wind'
    elif cube.standard_name == 'y_wind':
        cube.standard_name = 'northward_wind'
    # If standard name mismatch use STASH
    if (
        cube.standard_name
        and
        stash.standard_name
        and
        (cube.standard_name != stash.standard_name)
    ):
        LOGGER.warning(
            f"Standard name mismatch for ITEMCODE: {stash.itemcode}.\n"
            f"iris standard name: {cube.standard_name}, STASH standard name: {stash.standard_name}."
        )
        cube.standard_name = stash.standard_name
    # If unit mismatch use STASH
    if (
        cube.units
        and
        stash.units
        and
        (str(cube.units) != stash.units)
    ):
        LOGGER.warning(
            f"Units mismatch for ITEMCODE: {stash.itemcode}.\n"
            f"iris units: {cube.units}, STASH units: {stash.units}."
        )
        cube.units = stash.units
    # If there's no iris standard_name or long_name use one from STASH
    if not cube.standard_name and stash.standard_name:
        cube.standard_name = stash.standard_name
    if not cube.long_name and stash.long_name:
        cube.long_name = stash.long_name


def fix_cell_methods(cube):
    """Remove intervals in cell methods as it's not reliable."""
    # Input is tuple of cell methods
    newm = []
    for m in cube.cell_methods:
        newi = []
        for i in m.intervals:
            # Skip the misleading hour intervals
            if i.find('hour') == -1:
                newi.append(i)
        n = iris.coords.CellMethod(
            m.method, m.coord_names, tuple(newi), m.comments)
        newm.append(n)
    cube.cell_methods = tuple(newm)


def fix_latlon_coord(cube, grid_type):
    """Get proper lat/lon coordinate names based on cube grid_type"""
    def _add_coord_bounds(coord):
        if not coord.has_bounds():
            if len(coord.points) > 1:
                coord.guess_bounds()
            # For length 1, assume it's global. guess_bounds doesn't work in this case
            elif coord.name() == 'latitude':
                coord.bounds = np.array([[-90., 90.]])
            elif coord.name() == 'longitude':
                coord.bounds = np.array([[0., 360.]])
    try:
        lat = cube.coord('latitude')
        # Force to double for consistency with CMOR
        lat.points = lat.points.astype(np.float64)
        _add_coord_bounds(lat)
        lon = cube.coord('longitude')
        lon.points = lon.points.astype(np.float64)
        _add_coord_bounds(lon)

        if len(lat.points) == 180:
            lat.var_name = 'lat_river'
        elif (
            (lat.points[0] == -90 and grid_type == 'EG')
            or
            (
                np.allclose(-90.+np.abs(0.5 *
                            (lat.points[1]-lat.points[0])), lat.points[0])
                and
                grid_type == 'ND'
            )
        ):
            lat.var_name = 'lat_v'
        else:
            lat.var_name = 'lat'

        if len(lon.points) == 360:
            lon.var_name = 'lon_river'
        elif (
            (lon.points[0] == 0 and grid_type == 'EG')
            or
            (
                np.allclose(
                    np.abs(0.5*(lon.points[1]-lon.points[0])), lon.points[0])
                and
                grid_type == 'ND'
            )
        ):
            lon.var_name = 'lon_u'
        else:
            lon.var_name = 'lon'
    except iris.exceptions.CoordinateNotFoundError:
        msg = ("File cannot be processed. UM files with time series currently unsupported. Consider"
               " converting with convsh https://ncas-cms.github.io/xconv-doc/html/example1.html")
        raise UMError(msg)


def fix_level_coord(cube, z_rho, z_theta):
    """Rename model_level_number coordinates to better distinguish rho and theta levels"""
    try:
        c_lev = cube.coord('model_level_number')
        c_height = cube.coord('level_height')
        c_sigma = cube.coord('sigma')
    except iris.exceptions.CoordinateNotFoundError:
        return
    if abs(c_height.points[0]-z_rho).min() < 1e-6:
        c_lev.var_name = 'model_rho_level_number'
        c_height.var_name = 'rho_level_height'
        c_sigma.var_name = 'sigma_rho'
    elif abs(c_height.points[0]-z_theta).min() < 1e-6:
        c_lev.var_name = 'model_theta_level_number'
        c_height.var_name = 'theta_level_height'
        c_sigma.var_name = 'sigma_theta'


def fix_pressure_coord(cube):
    """Fix pressure coords"""
    try:
        plevs = cube.coord('pressure')
        plevs.attributes['positive'] = 'down'
        plevs.convert_units('Pa')
        # Round coord points otherwise
        # they're off by 1e-10 which looks odd in ncdump
        plevs.points = np.round(plevs.points, 5)
        # If needed, flip to get pressure decreasing as in CMIP6 standard
        if plevs.points[0] < plevs.points[-1]:
            cube = iris.util.reverse(cube, 'pressure')
    except iris.exceptions.CoordinateNotFoundError:
        pass
    return cube


def to32bit_data(cube):
    """Change data to 32 bit"""
    if cube.data.dtype == 'float64':
        cube.data = cube.data.astype(np.float32)
    elif cube.data.dtype == 'int64':
        cube.data = cube.data.astype(np.int32)


def set_missing_value(cube):
    """
    Set the missing_value attribute. 
    Use an array to force the type to match the data type
    """
    kind = cube.data.dtype.kind
    if kind == 'f':
        fill_value = 1.e20
    else:
        # Use netCDF defaults
        key = f"{kind}{cube.data.dtype.itemsize:1d}"
        fill_value = netCDF4.default_fillvals[key]
    cube.attributes['missing_value'] = np.array([fill_value], cube.data.dtype)


def convert_proleptic_calendar(cube):
    """
    If reference date is before 1600 use proleptic gregorian
    calendar and change units from hours to days
    """
    def _convert_proleptic(time):
        """Convert units from hours to days and shift origin from 1970 to 0001"""
        newunits = cf_units.Unit(
            "days since 0001-01-01 00:00", calendar='proleptic_gregorian')
        # Need a copy because can't assign to time.points[i]
        tvals = np.array(time.points)
        if time.bounds is not None:
            tbnds = np.array(time.bounds)
            has_bnds = True
        else:
            has_bnds = False
        for i in range(len(time.points)):
            date = time.units.num2date(tvals[i])
            newdate = cftime.DatetimeProlepticGregorian(
                date.year,
                date.month,
                date.day,
                date.hour,
                date.minute,
                date.second
            )
            tvals[i] = newunits.date2num(newdate)
            if has_bnds:  # Fields with instantaneous data don't have bounds
                for j in range(2):
                    date = time.units.num2date(tbnds[i][j])
                    newdate = cftime.DatetimeProlepticGregorian(
                        date.year,
                        date.month,
                        date.day,
                        date.hour,
                        date.minute,
                        date.second
                    )
                    tbnds[i][j] = newunits.date2num(newdate)
        time.points = tvals
        if has_bnds:
            time.bounds = tbnds
        time.units = newunits

    try:
        reftime = cube.coord('forecast_reference_time')
    except iris.exceptions.CoordinateNotFoundError:
        # Dump files don't have forecast_reference_time
        return
    time = cube.coord('time')
    refdate = reftime.units.num2date(reftime.points[0])
    tuom = time.units.origin == 'hours since 1970-01-01 00:00:00'
    LOGGER.debug(
        f"Time units origin match: {tuom}"
    )
    if time.units.calendar == 'proleptic_gregorian' and refdate.year < 1600:
        _convert_proleptic(time)
    else:
        if time.units.calendar == 'gregorian':
            new_calendar = 'proleptic_gregorian'
        else:
            new_calendar = time.units.calendar
        time.units = cf_units.Unit(
            "days since 1970-01-01 00:00", calendar=new_calendar)
        time.points = time.points/24.
        if time.bounds is not None:
            time.bounds = time.bounds/24.
    cube.remove_coord('forecast_period')
    cube.remove_coord('forecast_reference_time')


def cubewrite(cube, sman, compression):
    """Write cube to file"""
    fill_value = cube.attributes['missing_value']
    try:
        # If time is a dimension but not a coordinate dimension,
        # coord_dims('time') returns an empty tuple
        if tdim := cube.coord_dims('time'):
            # For fields with a pseudo-level, time may not be the first dimension
            if tdim != (0,):
                tdim = tdim[0]
                neworder = list(range(cube.ndim))
                neworder.remove(tdim)
                neworder.insert(0, tdim)
                LOGGER.warning(
                    "Incorrect dimension order for ITEMCODE: "
                    f"{Stash(cube.attributes['STASH']).itemcode}.\n"
                    f"Changing dimension order to {neworder}."
                )
                cube.transpose(neworder)
        else:
            cube = iris.util.new_axis(cube, cube.coord('time'))
        sman.write(
            cube,
            zlib=True,
            complevel=compression,
            unlimited_dimensions=['time'],
            fill_value=fill_value)
    except iris.exceptions.CoordinateNotFoundError:
        # No time dimension (probably ancillary file)
        sman.write(
            cube,
            zlib=True,
            complevel=compression,
            fill_value=fill_value
        )


def main(args):
    """
    Main function for `um2nc` command
    """
    # Get input path
    LOGGER.debug(f"{args=}")
    infile = get_abspath(args.infile)
    LOGGER.debug(f"{infile=}")
    # Get output path
    outfile = get_abspath(args.outfile, checkdir=True)
    # Get netCDF format
    nc_format = get_nc_format(args.format)
    check_ncformat(nc_format, args.use64bit)
    # Use mule to get the model levels to help with dimension naming
    LOGGER.info(f"Reading UM file {infile}")
    ff = umutils.read_fieldsfile(infile)
    try:
        cubes = iris.load(infile)
    except iris.exceptions.CannotAddError:
        msg = ("UM file can not be processed. UM files with time series currently not supported."
               "Convert with convsh https://ncas-cms.github.io/xconv-doc/html/example1.html.")
        raise UMError(msg)

    # Get order of fields (from stash codes)
    stash_order = list(dict.fromkeys([f.lbuser4 for f in ff.fields]))
    LOGGER.debug(f"{stash_order=}")
    # Order the cubelist based on input order
    cubes.sort(key=lambda c:
               stash_order.index(c.attributes['STASH'].section*1000 + c.attributes['STASH'].item))

    # Get heaviside fields for pressure level masking
    if not args.nomask:
        heaviside_uv = get_heaviside_uv(cubes)
        heaviside_t = get_heaviside_t(cubes)
    # Get grid type
    grid_type = umutils.get_grid_type(ff)
    # Get sea level on rho levels
    z_rho = umutils.get_sealevel_rho(ff)
    # Get sea level on theta levels
    z_theta = umutils.get_sealevel_theta(ff)
    # Write output file
    LOGGER.info(f"Writing netCDF file {outfile}")

    try:
        with iris.fileformats.netcdf.Saver(outfile, nc_format) as sman:
            # Add global attributes
            add_global_attrs(infile, sman, args.nohist)
            for c in cubes:
                stash = Stash(c.attributes['STASH'])
                itemcode = stash.itemcode
                LOGGER.debug(
                    f"Processing STASH field: {itemcode}"
                )
                # Skip fields not specified with --include-list option
                # or fields specified with --exclude-list option
                if (
                    (args.include_list and itemcode not in args.include_list)
                    or
                    (args.exclude_list and itemcode in args.exclude_list)
                ):
                    LOGGER.debug(
                        f"Field with itemcode '{itemcode}' excluded from the conversion."
                    )
                    continue
                # Name cube
                name_cube(c, stash, args.simple)
                # Remove unreliable intervals in cell methods
                fix_cell_methods(c)
                # Properly name lat/lon coordinates
                fix_latlon_coord(c, grid_type)
                # Properly name model_level_number coordinates
                fix_level_coord(c, z_rho, z_theta)
                # Mask pressure level fields
                if not args.nomask:
                    if not apply_mask_to_pressure_level_field(
                        c,
                        stash,
                        heaviside_uv,
                        heaviside_t,
                        args.hcrit
                    ):
                        continue
                # Fix pressure coordinates
                c = fix_pressure_coord(c)
                # change data to 32bit
                if not args.use64bit:
                    to32bit_data(c)
                # Set missing value
                set_missing_value(c)
                # Convert proleptic calendar
                convert_proleptic_calendar(c)
                LOGGER.info(
                    f"Writing field '{c.var_name}' -- ITEMCODE: {itemcode}"
                )
                cubewrite(c, sman, args.compression)

    # Catch any errors and remove the output file if it exists
    except Exception as ex:
        if os.path.exists(outfile):
            os.remove(outfile)
        raise AmamiError(ex)
