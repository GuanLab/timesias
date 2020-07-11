#!/bin/bash
## $1 data base directory
perl split.pl $1 $2
perl prepare_data.pl $1
