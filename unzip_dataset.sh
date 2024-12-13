#!/bin/bash

# Path to the ZIP file
zip_file="data_qecnn.zip"

# Path to the extraction folder
output_dir="./dataset"

# Check if the ZIP file exists
if [ ! -f "$zip_file" ]; then
    echo "Error: ZIP file '$zip_file' not found."
    exit 1
fi

# Create the extraction folder if it does not exist
if [ ! -d "$output_dir" ]; then
    echo "Creating output directory '$output_dir'..."
    mkdir -p "$output_dir"
fi

# Unzip the contents of the ZIP file into the output directory
echo "Unzipping '$zip_file' to '$output_dir'..."
unzip -o "$zip_file" -d "$output_dir"

# Check if the unzip command was successful
if [ $? -eq 0 ]; then
    echo "Unzipping completed successfully."
else
    echo "Error: Unzipping failed."
    exit 1
fi
