"""
Script to convert line segments (LineStrings) in GeoPackage files into closed polygons
using shapely's polygonize. Designed for processing roof edge data from Gao.

- Input: .gpkg files with line geometries
- Output: .gpkg files with polygon geometries
- Dependencies: geopandas, shapely


"""


# Convert Gao's line segments into polygons

import os
import geopandas as gpd
from shapely.ops import linemerge, polygonize, unary_union
from shapely.geometry import LineString

# Input and output folders
input_folder = r"E:\new_MRF\data\RoofVec\geopackages"
output_folder = r"E:\new_MRF\data\RoofVec\polygon_geopackages"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Iterate over all GeoPackage files
for file_name in os.listdir(input_folder):
    if file_name.endswith(".gpkg"):  # Process only GeoPackage files
        input_file = os.path.join(input_folder, file_name)

        # Remove file extension and append "_polygon.gpkg"
        name_without_ext = os.path.splitext(file_name)[0]
        output_file = os.path.join(output_folder, f"{name_without_ext}_polygon.gpkg")

        # Read the GeoPackage file
        gdf = gpd.read_file(input_file)

        # Merge all line segments into a unified geometry
        merged_lines = unary_union(gdf.geometry)

        # If the result is a single LineString, convert it to a list
        if isinstance(merged_lines, LineString):
            merged_lines = [merged_lines]

        # Attempt to generate polygons from the merged lines
        polygons = list(polygonize(merged_lines))

        # Skip this file if no valid polygons were created
        if not polygons:
            print(f"{file_name} could not generate valid polygons.")
            continue

        # Save polygons to a new GeoDataFrame
        polygon_gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=gdf.crs)
        polygon_gdf.to_file(output_file, layer="polygons", driver="GPKG")

        print(f"Processed file: {file_name} → Output saved to: {output_file}")

print("✅ All files have been successfully processed.")