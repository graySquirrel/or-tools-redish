import googlemaps
import pandas as pd

gmaps = googlemaps.Client(key='Goog maps API Key') # put your API key here...

custNames = ["Re-Dish", "White Case", "MOMA", "Loreal", "SIS Cooper", "JP Morgan", \
             "Berkeley Carroll", "SIS Wadsworth", "Brearly", "L'Oreal NJ"]
custLats = [40.69998097056833, 40.7594104, 40.7618038, 40.7527315, 40.6865353, 40.6926286, \
            40.67402795, 40.60478, 40.63973545, 40.62745318]
custLons = [-73.94819913666936, -73.9828299, -73.9775202, -74.00152786, -73.9083109, -73.98384265, \
            -73.97790482, -74.0643273, -73.91455675, -74.32094301]
zipped = list(zip(custNames, custLats, custLons))
df = pd.DataFrame(zipped, columns=['ID', 'latitude', 'longitude'])

time_list = []
distance_list = []
origin_id_list = []
origin_id_ord = []
destination_id_list = []
destination_id_ord = []
destinations = list(map(lambda x, y:(x,y), custLats, custLons))
for (i1, row1) in df.iterrows(): # loop through the 'from's and call goog for all dests for each from
    print("origin")
    print(row1['ID'])
    LatOrigin = row1['latitude']
    LongOrigin = row1['longitude']
    origin = (LatOrigin, LongOrigin)
    origin_id = row1['ID']
    results = gmaps.distance_matrix(origin, destinations, mode='driving') # can only do 100 combos at a time.
    for (i2, row2) in df.iterrows():
        print("destination id")
        print(row2['ID'])
        # LatDestination = row2['latitude']
        # LongDestination = row2['longitude']
        destination_id = row2['ID']
        # destination = (LatDestination, LongDestination)
        # result = gmaps.distance_matrix(origin, destination, mode='driving')
        result_distance = results["rows"][0]["elements"][i2]["distance"]["value"]
        result_time = results["rows"][0]["elements"][i2]["duration"]["value"]
        time_list.append(result_time)
        distance_list.append(result_distance)
        origin_id_list.append(origin_id)
        origin_id_ord.append(i1)
        destination_id_list.append(destination_id)
        destination_id_ord.append(i2)

output = pd.DataFrame(origin_id_list, columns = ['origin_id'])
output['destination_id'] = destination_id_list
output['distance in meters'] = distance_list
output['duration in seconds'] = time_list
output['origin_ordinal'] = origin_id_ord
output['destination_ordinal'] = destination_id_ord

output.to_csv("distanceMatrixLong.csv", index=False)

wide = pd.pivot(output, index='origin_ordinal', columns='destination_ordinal', values='distance in meters')
wide.to_csv("distance_matrix.csv", header= None, index=False)

warr = wide.to_numpy()
warr = warr/1000.0
#print(warr)
print("np.array([")
for r in warr:
    print("[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f]," % (r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8],r[9]))
print("])")

widetime = pd.pivot(output, index='origin_ordinal', columns='destination_ordinal', values='duration in seconds')
widetime.to_csv("time_matrix.csv", header= None, index=False)

warrtime = widetime.to_numpy()
warrtime = warrtime/3600.0
#print(warr)
print("np.array([")
for r in warrtime:
    print("[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f]," % (r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8],r[9]))
print("])")
# distance
# np.array([
# [0.000000, 14.502000, 14.531000, 11.288000, 4.228000, 4.035000, 5.548000, 20.082000, 9.519000, 41.720000],
# [13.408000, 0.000000, 0.740000, 3.322000, 27.310000, 12.042000, 14.709000, 27.093000, 31.648000, 44.075000],
# [13.621000, 1.384000, 0.000000, 4.000000, 27.522000, 12.254000, 14.921000, 27.306000, 31.860000, 44.979000],
# [13.960000, 2.460000, 3.202000, 0.000000, 27.862000, 8.627000, 12.501000, 22.557000, 19.920000, 42.864000],
# [4.133000, 28.293000, 28.323000, 28.914000, 0.000000, 7.459000, 8.444000, 30.067000, 5.978000, 54.641000],
# [3.808000, 12.518000, 12.535000, 8.444000, 8.036000, 0.000000, 2.668000, 16.357000, 10.688000, 38.875000],
# [6.142000, 17.822000, 17.839000, 12.540000, 7.593000, 2.996000, 0.000000, 14.237000, 8.962000, 38.820000],
# [21.133000, 28.632000, 28.650000, 23.351000, 31.336000, 19.004000, 17.216000, 0.000000, 27.669000, 26.843000],
# [9.787000, 25.259000, 25.277000, 19.978000, 5.978000, 10.979000, 9.422000, 26.265000, 0.000000, 50.839000],
# [42.156000, 42.328000, 43.371000, 42.164000, 54.658000, 39.239000, 40.537000, 26.756000, 50.990000, 0.000000],
# ])
# #time
# np.array([
# [0.000000, 0.526389, 0.514167, 0.536111, 0.270833, 0.235556, 0.386389, 0.490556, 0.501944, 0.864722],
# [0.463889, 0.000000, 0.074444, 0.232778, 0.610000, 0.450556, 0.600833, 0.604444, 0.849167, 0.731111],
# [0.464167, 0.134167, 0.000000, 0.288056, 0.610556, 0.450833, 0.601389, 0.604722, 0.849444, 0.782222],
# [0.536111, 0.198056, 0.241667, 0.000000, 0.682500, 0.448611, 0.517500, 0.535278, 0.823056, 0.584167],
# [0.281667, 0.692778, 0.680556, 0.722500, 0.000000, 0.498056, 0.466389, 0.699722, 0.385556, 1.132778],
# [0.186667, 0.482500, 0.471389, 0.390833, 0.457778, 0.000000, 0.201667, 0.393889, 0.560278, 0.719444],
# [0.356944, 0.646667, 0.635278, 0.516667, 0.428611, 0.255278, 0.000000, 0.373333, 0.452222, 0.805278],
# [0.538611, 0.678611, 0.667500, 0.548889, 0.712778, 0.481667, 0.416667, 0.000000, 0.525556, 0.562222],
# [0.526667, 0.914722, 0.903611, 0.784722, 0.370556, 0.607500, 0.493056, 0.485833, 0.000000, 0.918889],
# [0.888889, 0.726389, 0.768611, 0.632222, 1.123333, 0.781944, 0.827222, 0.556389, 0.936111, 0.000000],
# ])
