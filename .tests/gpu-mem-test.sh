#!/usr/bin/env bash

# test:
#		test potential device memory issues 

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTPATH=${DIR}/../output/.tests

IM_FILE=${DIR}/im_test_input.txt
RE_FILE=${DIR}/re_test_input.txt
EXEC=${DIR}/../gpe

mkdir -p $OUTPATH

echo "output path: ${OUTPATH}"

# check imaginary-time evolution 
sed -i "s#outprefix.*#outprefix       \"${OUTPATH}/im_test_\"#g" $IM_FILE
compute-sanitizer --tool memcheck --leak-check full  $EXEC $IM_FILE > /dev/null

# check real-time evolution
sed -i "s#outprefix.*#outprefix       \"${OUTPATH}/re_test_\"#g" $RE_FILE
sed -i "s#inprefix.*#inprefix        \"${OUTPATH}/im_test_\"#g" $RE_FILE
compute-sanitizer --tool memcheck --leak-check full  $EXEC $RE_FILE > /dev/null

rm -rf $OUTPATH
