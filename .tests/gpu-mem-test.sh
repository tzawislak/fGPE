#!/usr/bin/env bash

# test:
#		test potential device memory issues 

set -e

cd .tests

OUTPATH=$(pwd)/../output/.tests
mkdir -p $OUTPATH

echo "output path: ${OUTPATH}"

# check imaginary-time evolution 
sed -i "s#outprefix.*#outprefix       \"${OUTPATH}/im_test_\"#g" im_test_input.txt
compute-sanitizer --tool memcheck --leak-check full  ../gpe im_test_input.txt 

# check real-time evolution
sed -i "s#outprefix.*#outprefix       \"${OUTPATH}/re_test_\"#g" re_test_input.txt
sed -i "s#inprefix.*#inprefix        \"${OUTPATH}/im_test_\"#g" re_test_input.txt
compute-sanitizer --tool memcheck --leak-check full  ../gpe re_test_input.txt

rm -rf $OUTPATH
cd ..
