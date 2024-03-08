#!/bin/bash

for file in $*
do
	nb_exp=$(grep "offset_error_mean" $file | wc -l)
	vals_offset=$(grep "offset_error_mean" $file | cut -d= -f2)
	vals_angle=$(grep "angle_error_mean" $file | cut -d= -f2)

	sum_offset=0
	sum_angle=0

	for i in $vals_angle
	do
		sum_angle=$(bc <<< "$sum_angle + $i")
	done
	for i in $vals_offset
	do
		sum_offset=$(bc <<< "$sum_offset + $i")
	done

	mean_offset=$(python -c "print($sum_offset / $nb_exp)")
	mean_angle=$(python -c "print($sum_angle / $nb_exp)")
	echo $file
	echo ----
	echo Amount : $nb_exp
	echo offset $mean_offset
	echo angle $mean_angle
	echo
done
