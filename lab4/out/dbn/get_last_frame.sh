#!/bin/bash


# Parameters
num_images=$1
rows=$2
columns=$3

# Output filename
output_file="generate_last_frame/merged_generate.jpg"

mkdir -p generate_last_frame



#Take the last frames of each mp4 file and store them in generate_last_frame folder with the same filename
for i in {0..$num_images}
do
	ffmpeg -sseof -3 -i generate/rbms.generate$i.mp4 -update 1 -q:v 1 generate_last_frame/rbms.generate$i.jpg
done


# Ensure ImageMagick is installed
if ! command -v convert &> /dev/null
then
    echo "ImageMagick (convert) is required but not installed. Please install it."
    exit 1
fi

# Check for at least 3 arguments: number of images, rows, columns
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <number_of_images> <rows> <columns>"
    exit 1
fi



# Ensure num_images, rows, and columns are integers
if ! [[ "$num_images" =~ ^[0-9]+$ ]] || ! [[ "$rows" =~ ^[0-9]+$ ]] || ! [[ "$columns" =~ ^[0-9]+$ ]]; then
    echo "Error: <number_of_images>, <rows>, and <columns> must be integers."
    exit 1
fi

# Check if rows * columns >= num_images
if [ "$((rows * columns))" -lt "$num_images" ]; then
    echo "Error: Grid size (rows * columns) must be greater than or equal to the number of images."
    exit 1
fi

# Prepare filenames based on the number of images
files=()
for ((i=0; i<num_images; i++)); do
    filename="generate_last_frame/rbms.generate${i}.jpg"
    if [ ! -f "$filename" ]; then
        echo "Error: $filename not found."
        exit 1
    fi
    files+=("$filename")
done

# Ensure we have the right number of images in the array
if [ "${#files[@]}" -ne "$num_images" ]; then
    echo "Error: Number of images does not match expected count."
    exit 1
fi



# Merge the images into the specified grid
montage generate_last_frame/rbms.generate*.jpg -tile ${columns}x${rows} $output_file

# Confirm the output
if [ -f "$output_file" ]; then
    echo "Successfully merged $num_images images into $output_file with a grid of ${rows}x${columns}."
else
    echo "Image merge failed."
fi
