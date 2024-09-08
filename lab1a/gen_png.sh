#!/bin/bash






## declare an array variable
declare -a arr=("scatter_vs_lr"
		"non_linearly_separable"
		"removed_bias"
		"removed_bias_mA_0.4_0_mB_-0.4_0"
		"non_lin_batch_25_from_each_class"
		"non_lin_batch_50_from_A"
		"non_lin_batch_50_from_B"
		"non_lin_batch_point_2_lt_0_and_point_8_gt_0_from_A"
	)


ROOT_OUT="./png_frames_of_gifs"


#If the root folder doesnt exist, then create it
if [ ! -d "$ROOT_OUT" ]; then
	echo "$ROOT_OUT DOES NOT EXIST, CREATING IT NOW"
	mkdir $ROOT_OUT
fi


## now loop through the above array
for i in "${arr[@]}"
do
   echo "NOW VISITING DIRECTORY $i"
   # or do whatever with individual element of the array

   CURR_DIRECTORY=$i
   CURR_DIR_PATH="$ROOT_OUT/$i"

   echo "CURR_DIR_PATH: $CURR_DIR_PATH"
   
   if [ ! -d "$CURR_DIR_PATH" ]; then
       echo "$CURR_DIR_PATH does not exist."
       mkdir $CURR_DIR_PATH
   fi

    


   for file in ./$CURR_DIRECTORY/*
   do


       
       filepath=$ROOT_OUT
       filename=${file%.gif}
       filepath+="${filename:1}"


       if [ ! -d "$filepath" ]; then
           echo "$filepath does not exist."
           mkdir $filepath
       fi


       filepath+="/frame"
       
       echo "$filepath"
       #echo ${filepath}
       #echo "$file" | sed 's:.*/::'
       #cmd [option] "$file" >> results.out
       convert -coalesce "$file" "$filepath".png

   done


done

# You can access them using echo "${arr[0]}", "${arr[1]}" also


