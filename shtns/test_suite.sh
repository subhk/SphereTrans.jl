#!/bin/bash
# script to test many sht cases

#id=`git branch | sed -e '/^[^*]/d' -e 's/* //'`
id=`git rev-parse HEAD`
log="test_suite.log"

function test1 {
    run="./time_SHT $1 -vector"
    echo $run
    echo "---" >> $log
    echo "*** $run *** " >> $log
    $run > tmp.out
    cat tmp.out | grep ERROR
    cat tmp.out | grep -i nan
    cat tmp.out >> $log
}

echo "beginning test suite for $id" > $log
lscpu -p=MODELNAME | tail -1 > $log

# first, do a huge transform :
test1 "2047 -mres=15 -quickinit -iter=1"

# even bigger :
test1 "7975 -mres=145 -quickinit -iter=1"

# without threads
test1 "2047 -mres=15 -quickinit -iter=1 -nth=1"

# batch transforms
test1 "426 -quickinit -iter=16 -batch -nlorder=2"
test1 "426 -quickinit -iter=15 -batch -reg -transpose"
test1 "426 -quickinit -iter=14 -batch -regpoles -nlorder=2"
test1 "426 -quickinit -iter=13 -batch -schmidt"

for switch in "" "-oop" "-transpose" "-schmidt" "-4pi" "-batch"
do
  for mode in "-quickinit" "-gauss" "-reg" "-regpoles" "-gauss -nth=1"
  do
    for lmax in 2 3 4 11 12 13 14 31 32 33 34 121 122 123 124
    do
      for mmax in 0 1 $lmax
      do
         test1 "$lmax -mmax=$mmax $mode $switch -iter=1"
      done
    done
    for nlat in 32 33 34 36 38 40 42 44 46
    do
         test1 "15 -mmax=10 -nlat=$nlat $mode $switch -iter=10"
    done
  done
done
