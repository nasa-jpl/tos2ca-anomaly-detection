import geojson
import json
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import cartopy.mpl.ticker as cticker
import cartopy.crs as ccrs
import numpy as np
import random
import xarray as xr
import s3fs
import boto3
import shapefile
import cartopy.feature as cfeature
import netCDF4 as nc

from copy import deepcopy
from datetime import datetime as dt
from tos2ca.database.connection import openDB, closeDB
from tos2ca.database.elasticache import getData
from tos2ca.utils.s3 import s3Upload
from tos2ca.database.queries import updateStatus
from tos2ca.utils import tos2ca_secrets
from glob import glob
from shapely.geometry import Polygon, shape, MultiPolygon
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle

def get_anno_coords(lat, lon, mask):
    series_ids = np.unique(mask[mask > 0])
    mid = np.array([(lon[-1] + lon[0]) / 2, (lat[-1] + lat[0]) / 2])
    x_idx = np.arange(len(lon))
    y_idx = np.arange(len(lat))

    centroids = dict()
    bbox = dict()
    for i in series_ids:  # range(1, num_events + 1):
        event = deepcopy(mask)
        event[event != i] = 0
        event_x = np.sum(event, 0)
        event_x /= event_x.sum()
        event_y = np.sum(event, 1)
        event_y /= event_y.sum()

        x_idx_box = x_idx[event_x > 0]
        y_idx_box = y_idx[event_y > 0]
        corners = []
        for j in [0, -1]:
            for k in [0, -1]:
                corners.append(
                    [
                        lon[x_idx_box[j]],
                        lat[y_idx_box[k]]
                    ]
                )
        corners = np.array(corners)
        mid_dist = np.sqrt(np.sum((corners - mid) ** 2, 1))
        best_corner = corners[mid_dist == mid_dist.min(), :]
        bbox[i] = (
            best_corner[0, 0],
            best_corner[0, 1]
        )
        centroids[i] = (
            np.inner(lon, event_x),
            np.inner(lat, event_y)
        )

    line_coords = dict()
    for i in series_ids:
        x_b, y_b = bbox[i]
        x_c, y_c = centroids[i]

        line_coords[i] = (
            [x_c, x_b],
            [y_c, y_b]
        )

    return line_coords, bbox


def mask_plot(jobID, chunkID):
    """
    Function to draw a plot showing ForTraCC masks plotted on top
    of the inequality variable
    :param jobID: the jobID we want to run the plots for
    :type jobID: int
    :param chunkID: chunkID we want to plot
    :type chunkID: int
    """
    db, cur = openDB()
    updateStatus(db, cur, jobID, 'plotting')
    secret = tos2ca_secrets.get_secret("mysql-tos2causer-tos2ca", "us-west-2")
    bucketName = secret.get("bucket")
    data = {}
    tmp, jobInfo, start_time = getData(jobID, chunkID)
    if len(data) == 0:
        data = tmp
    else:
        data['images'].update(tmp['images'])

    sql = f'SELECT variable, dataset FROM jobs WHERE jobID={jobID}'
    cur.execute(sql)
    results = cur.fetchall()
    dataset = results[0]['dataset']
    variable = results[0]['variable']
    sql = f'SELECT location, type FROM output WHERE jobID={jobID} AND type IN ("masks", "subset")'
    cur.execute(sql)
    results = cur.fetchall()
    dataFiles = []
    for result in results:
        if result['type'] == 'masks':
            maskFile = result['location']
        if result['type'] == 'subset':
            dataFiles.append(result['location'])

    with open('/data/code/data-dictionaries/tos2ca-phdef-dictionary.json') as phdef:
        info = json.load(phdef)
    units = info[dataset]['units'][variable]

    fs = s3fs.S3FileSystem()

    # Read in nc4 data using xarray since read_nc4 doesn't work wirh S3 buckets
    with xr.open_dataset(fs.open(maskFile, 'rb'), group='navigation') as ds:
        lat = np.asarray(ds['lat'][:])
        lon = np.asarray(ds['lon'][:])
    grid_shape = (len(lat), len(lon))
    masks = dict()
    for timestamp in data['images'].keys():
        with xr.open_dataset(fs.open(maskFile, 'rb'), group=f'masks/{timestamp}') as ds:
            mask_indices = np.asarray(ds['mask_indices'][:])
        masks[timestamp] = np.zeros(grid_shape)
        for i, j, event_id in mask_indices:
            masks[timestamp][i, j] = event_id

    colors = list(mcolors.CSS4_COLORS.keys())
    random.shuffle(colors)

    lat_res = np.abs(lat[0] - lat[1])
    lon_res = np.abs(lon[0] - lon[1])
    for timestep, mask in masks.items():
        print('Plotting %s' % timestep)

        data_vals = np.array(data['images'][timestep])
        title = f"{dataset} {dt.strptime(timestep, '%Y%m%d%H%M').strftime('%Y-%m-%d %H:%M:%S')} - {variable} ({units})"

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
        ax.coastlines('10m')
        ax.set_xticks([*range(-180,190,10)], crs=ccrs.PlateCarree())
        ax.set_yticks([*range(-90,100,10)], crs=ccrs.PlateCarree())
        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
        pc = ax.pcolormesh(
            lon,
            lat,
            data_vals,
            shading='auto',
            alpha=0.3,
            cmap='jet'
        )
        plt.colorbar(pc, label=f'{variable} ({units})')

        series_ids = np.unique(mask[mask > 0]).astype(int)
        num_events = len(series_ids)
        features = list()
        if num_events > 0:
            line_coords, bbox = get_anno_coords(lat, lon, mask)
            mask[mask == 0] = np.nan

            data_event = deepcopy(data_vals)
            data_event[np.isnan(mask)] = np.nan

            ax.pcolormesh(
                lon,
                lat,
                data_event,
                shading='auto',
                cmap='jet'
            )
            for i in series_ids:
                event_mask = deepcopy(mask)
                event_mask[event_mask != i] = np.nan

                cs = ax.contourf(
                    lon,
                    lat,
                    event_mask,
                    alpha=0
                )
                for collection in cs.collections:
                    for path in collection.get_paths():
                        patch = matplotlib.patches.PathPatch(
                            path,
                            fill=False,
                            linewidth=2.0,
                            edgecolor=colors[i % len(colors)]
                        )
                        ax.add_patch(patch)

                        # Save geojson -- copied from original code
                        if path.to_polygons():
                            for npoly, polypoints in enumerate(path.to_polygons()):
                                #REMINDER: lat and lon positions need to be the reverse of
                                #how they're passed to the plot
                                poly_lats = polypoints[:, 0]
                                poly_lons = polypoints[:, 1]
                                poly_init = Polygon(
                                    [coords for coords in zip(poly_lats, poly_lons)]
                                )
                                if poly_init.is_valid:
                                    poly_clean = poly_init
                                else:
                                    poly_clean = poly_init.buffer(0.)
                                if npoly == 0:
                                    poly = poly_clean
                                else:
                                    poly = poly.difference(poly_clean)
                            footprint = geojson.Feature(
                                geometry=poly,
                                properties={
                                    'anomaly': str(i),
                                    'dateTime': timestep
                                }
                            )
                            features.append(footprint)
            gj = geojson.FeatureCollection(features)
            geoJsonFile = f'/data/tmp/{jobID}-{timestep}.json'
            with open(geoJsonFile, 'w') as f:
                geojson.dump(gj, f)
            jobInfo = {
                'filename': geoJsonFile,
                'startDateTime': dt.strptime(timestep, '%Y%m%d%H%M').strftime('%Y-%m-%d %H:%M:%S'),
                'type': 'geoJSON'
            }
            s3Upload(jobID, jobInfo, bucketName, db, cur)

            for i in series_ids:
                x, y = line_coords[i]
                ax.plot(x, y, color='white', linewidth=2.0)
                ax.annotate(f'{i}', bbox[i], color='white')

        plt.xlabel(f'longitude ({lon_res} resolution)')
        plt.ylabel(f'latitude ({lat_res} resolution)')
        plt.title(title)

        plotFile = f'/data/tmp/{jobID}-{timestep}.png'
        plt.savefig(plotFile)
        plt.close()

        jobInfo = {
            'filename': plotFile,
            'startDateTime': dt.strptime(timestep, '%Y%m%d%H%M').strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'plot'
        }
        s3Upload(jobID, jobInfo, bucketName, db, cur)
        updateStatus(db, cur, jobID, 'complete')

    closeDB(db)

    return


def interpolated_plot(jobID, timestamp, anomalyID):
    """
    This plotting function will plot a specific anomaly at a specific timestamp
    :param jobID: curated jobID
    :type jobID: int
    :param timestamp: timestamp to plot in YYYYMMMDDHHMM format
    :type: str
    :param anomalyID: ID of the anomaly to plot
    :type: int
    """
    db, cur = openDB()
    sql = f'SELECT j.variable, o.location FROM output o, jobs j WHERE j.jobID={jobID} AND j.jobID=o.jobID AND o.type="interpolated subset" AND o.location LIKE "%/{jobID}-Interp%"'
    cur.execute(sql)
    results = cur.fetchone()
    location = results['location']
    variable = results['variable']
    secret = tos2ca_secrets.get_secret("mysql-tos2causer-tos2ca", "us-west-2")
    bucketName = secret.get("bucket")

    fs = s3fs.S3FileSystem()
    ds = xr.open_dataset(fs.open(location, 'rb'), group=timestamp + '/' + str(anomalyID))

    data = ds[variable].values[...]
    lat_array = data[:, 0]
    lon_array = data[:, 1]

    arr = np.where(data[:, 2]>=0, data[:, 2], np.nan)

    plt.style.use(['seaborn-poster'])

    fig = plt.figure(figsize=(20,20))
    ax = plt.subplot(111)

    plt.plot(lon_array, lat_array,  '.k', markersize = 5)
    plt.scatter(lon_array, lat_array, s = 20, c= arr,cmap = 'jet')

    plt.title('Interpolated ' + variable + ' - MaskTime: '+ timestamp +', Event ID: ' + str(anomalyID)+', File: ' + location )
    plt.colorbar(label = variable + ' (' + ds[variable].Units+')', orientation =  'horizontal', shrink = 0.4, pad = 0.06)
    plt.ylabel('Latitude (deg)')
    plt.xlabel('Longitude (deg)')
    plt.tight_layout()
    plotFile = '/data/tmp/' + str(jobID) + '_' + timestamp + '_' + str(anomalyID) + '_.png'
    plt.savefig(plotFile)
    plt.close()

    jobInfo = {
        'filename': plotFile,
        'startDateTime': dt.strptime(timestamp, '%Y%m%d%H%M').strftime('%Y-%m-%d %H:%M:%S'),
        'type': 'interpolated plot'
    }
    s3Upload(jobID, jobInfo, bucketName, db, cur)

    ds.close()
    db.close()

    return

def interpolated_plot_all(jobID, timestamp):
    """
    This plotting function will plot a specific anomaly at a specific timestamp
    :param jobID: curated jobID
    :type jobID: int
    :param timestamp: timestamp to plot in YYYYMMMDDHHMM format
    :type: str
    """
    db, cur = openDB()
    sql = f'SELECT j.variable, o.location FROM output o, jobs j WHERE j.jobID={jobID} AND j.jobID=o.jobID AND o.type="interpolated subset" AND o.location LIKE "%/{jobID}-Interp%"'
    cur.execute(sql)
    results = cur.fetchone()
    location = results['location']
    variable = results['variable']
    secret = tos2ca_secrets.get_secret("mysql-tos2causer-tos2ca", "us-west-2")
    bucketName = secret.get("bucket")

    sql = f'SELECT location, type FROM output WHERE jobID={jobID} AND type IN ("interpolated hierarchy") AND location LIKE "%/{jobID}/{jobID}-Interpolat%"'
    cur.execute(sql)
    results = cur.fetchall()
    print(results)
    for result in results:
        interpolatedHierarchyFile = result['location']

    s3 = boto3.resource('s3')
    interpolatedHierarchyFileParts = interpolatedHierarchyFile.split('/')
    content_object = s3.Object(interpolatedHierarchyFileParts[2], '/'.join(interpolatedHierarchyFileParts[3:]))
    file_content = content_object.get()['Body'].read().decode('utf-8')
    interpolatedHierarchyInfo = json.loads(file_content)
    anomaly_num = list(interpolatedHierarchyInfo[timestamp].keys())
    
    fs = s3fs.S3FileSystem()

    try:
        lat_array = []
        lon_array = []
        interp_array = []  
        
        units = ''
        #loop over anomaly ids for plotting
        for anomaly_id in anomaly_num:  
                
            print("processing anomaly id: ", anomaly_id)
            ds = xr.open_dataset(fs.open(location, 'rb'), group=timestamp + '/' + anomaly_id)
            units = ds[variable].Units

            data = ds[variable].values[...]
            lat_array = np.append(lat_array, data[:, 0])
            lon_array = np.append(lon_array, data[:, 1])
                
            data_array = np.where(data[:, 2]>=0, data[:, 2], np.nan)
            interp_array = np.append(interp_array, data_array)
        
        plt.style.use(['seaborn-poster'])
        fig = plt.figure(figsize=(20,20))
        ax = plt.subplot(111, projection=ccrs.PlateCarree())
        
        plt.scatter(lon_array, lat_array, s = 2, c= interp_array, transform=ccrs.PlateCarree(), cmap = 'jet')
        plt.colorbar(label = variable + ' ('+ units +')', orientation =  'horizontal', shrink = 0.4, pad = 0.06)

        ax.set_yticks(np.arange(min(lat_array),max(lat_array), 5), crs=ccrs.PlateCarree())
        lat_formatter = cticker.LatitudeFormatter()
        ax.yaxis.set_major_formatter(lat_formatter)

        ax.set_xticks(np.arange(min(lon_array),max(lon_array), 5), crs=ccrs.PlateCarree())
        lon_formatter = cticker.LongitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
            
        ax.coastlines()

        
        plt.title('Interpolated '+variable+' - Time: '+timestamp)
        plt.ylabel('Latitude (deg)')
        plt.xlabel('Longitude (deg)')
        fig.canvas.draw()
        plt.tight_layout()
        plotFile = '/data/tmp/' + str(jobID) + '_' + timestamp + '.png'
        plt.savefig(plotFile)
        plt.close()

        jobInfo = {
            'filename': plotFile,
            'startDateTime': dt.strptime(timestamp, '%Y%m%d%H%M').strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'interpolated plot'
        }
        s3Upload(jobID, jobInfo, bucketName, db, cur)

        db.close()
        
    except:
        
        print("Error with timestamp: " + timestamp)
        
    return


def wildfirePlot(maskFile, shapeFile):
    """
    This function was used specifically to generate plots of the
    pre-defined California Wildfire masks.  These were processed
    after the masks were ggenerated on the local system.  This
    probably could be used to generate masks for any pre-defined
    phenomena.  You do also need to have the shapefile you want to use.
    In this case, we use one for the Unite States only since we
    were interested in California.  Update in the code as needed, but
    the shapefiles for the US can be found at:
    https://catalog.data.gov/dataset/tiger-line-shapefile-2023-nation-u-s-county-and-equivalent-entities
    :param maskFile: local path and filename of the mask file (*.nc4)
    :type maskFile: str
    :param shapeFile: local path and filename of the shapefile (*.shp)
    :type shapeFile: str
    """
    fileList = glob(maskFile)
    for f in fileList:
        data = nc.Dataset(f, 'r')
        timestamp = [i for i in data['masks'].groups.keys()][0]
        mask_indices = np.asarray(data['masks'][timestamp]['mask_indices'][:])
        lat = np.asarray(data['navigation']['lat'][:])
        lon = np.asarray(data['navigation']['lon'][:])
        empty = np.zeros((len(lat), len(lon)), dtype=int)
        for i, j, val in mask_indices:
            empty[i][j] = 1
        g_lat = []
        g_lon = []
        for i, j, val in mask_indices:
            g_lat.append(lat[i])
            g_lon.append(lon[j])
        g_lat = np.asarray(g_lat)
        g_lon = np.asarray(g_lon)

        cmap = ListedColormap(['white', 'red'])

        # Load the shapefile
        sf = shapefile.Reader(shapeFile)

        # Set up the map
        fig, ax = plt.subplots(figsize=(8, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())

        # Add coastlines and state boundaries
        ax.coastlines(resolution='10m')
        ax.add_feature(cfeature.STATES, edgecolor='gray')

        # Plot only California counties (FIPS state code 06)
        for record, geom in zip(sf.records(), sf.shapes()):
            if record[0] == '06':  # California FIPS code
                county_geom = shape(geom.__geo_interface__)
                
                if isinstance(county_geom, Polygon):
                    polygons = [county_geom]
                elif isinstance(county_geom, MultiPolygon):
                    polygons = county_geom.geoms
                else:
                    continue  # Skip if not polygonal

                # Plot each polygon
                for poly in polygons:
                    xs, ys = poly.exterior.xy
                    ax.plot(xs, ys, transform=ccrs.PlateCarree(), color='black', linewidth=0.5)


        # Turn off axis
        name = data.getncattr('FIRE_NAME')
        objectID = data.getncattr('OBJECTID')
        startDate = dt.strptime(data.getncattr('ALARM_DATE'), '%a, %d %b %Y %H:%M:%S GMT').strftime('%Y-%m-%d')
        features = list()

        try:
            endDate = dt.strptime(data.getncattr('CONT_DATE'), '%a, %d %b %Y %H:%M:%S GMT').strftime('%Y-%m-%d')
        except ValueError:
            endDate = '0000-00-00'
        ax.set_title("%s - California Wildfire (%s - %s)" % (name, startDate, endDate))
        ax.coastlines('10m')
        ax.set_xticks([*range(-115,-125,1)], crs=ccrs.PlateCarree())
        ax.set_yticks([*range(33,45,1)], crs=ccrs.PlateCarree())
        ax.pcolormesh(
                    lon,
                    lat,
                    empty,
                    shading='auto',
                    alpha=0.3,
                    cmap=cmap
                )
        cs = ax.contourf(
            lon,
            lat,
            empty,
            alpha=0
        )
        for collection in cs.collections:
            for path in collection.get_paths():
                # Save geojson -- copied from original code
                if path.to_polygons():
                    for npoly, polypoints in enumerate(path.to_polygons()):
                        # REMINDER: lat and lon positions need to be the reverse of
                        # how they're passed to the plot
                        poly_lats = polypoints[:, 0]
                        poly_lons = polypoints[:, 1]
                        poly_init = Polygon(
                            [coords for coords in zip(poly_lats, poly_lons)]
                        )
                        if poly_init.is_valid:
                            poly_clean = poly_init
                        else:
                            poly_clean = poly_init.buffer(0.)
                        if npoly == 0:
                            poly = poly_clean
                        else:
                            poly = poly.difference(poly_clean)
                    footprint = geojson.Feature(
                        geometry=poly,
                        properties={
                            'anomaly': str(i),
                            'dateTime': dt.strptime(data.getncattr('CONT_DATE'), '%a, %d %b %Y %H:%M:%S GMT').strftime('%Y%m%d%H%M%S')
                        }
                    )
                    features.append(footprint)
        gj = geojson.FeatureCollection(features)
        jobID = maskFile.split('/')[-1].split('-')[0] + '-' + maskFile.split('/')[-1].split('-')[1]
        geoJsonFile = f'/data/tmp/{jobID}-{endDate}.json'
        with open(geoJsonFile, 'w') as f:
            geojson.dump(gj, f)

        circle = Circle((np.median(g_lon), np.median(g_lat)), radius=0.3, edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(circle)
        plt.xlabel(f'longitude ({0.005} resolution)')
        plt.ylabel(f'latitude ({0.005} resolution)')
        
        try:
            plotFile = '/data/tmp/%s-%s.png' % (objectID, dt.strptime(data.getncattr('CONT_DATE'), '%a, %d %b %Y %H:%M:%S GMT').strftime('%Y%m%d%H%M%S'))
        except ValueError:
            plotFile = '/data/tmp/%s-%s.png' % (objectID, '00000000000000')
        plt.savefig(plotFile)
        plt.close()
        print('Saving %s' % plotFile)

    return