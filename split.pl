#!/usr/bin/perl

srand($ARGV[0]);

open FILE, "gs.file" or die;
<FILE>;
while ($line=<FILE>){
	chomp $line;
	@table=split ',', $line;
	@t=split "_", $table[0];
	$sample{$t[0]}=0;
}
close FILE;

@all_pid=keys %sample;
foreach $ppp (@all_pid){
	$r=rand(1);
	if ($r<0.8){
		$train{$ppp}=0;
	}
}

open FILE, "gs.file" or die;
<FILE>;
open TRAIN, ">train_gs.dat.${ARGV[0]}" or die;
open TEST, ">test_gs.dat.${ARGV[0]}" or die;
while ($line=<FILE>){
	@table=split ",", $line;
	if (-f "data/$table[0].psv"){
	@t=split "_", $table[0];
	if (exists $train{$t[0]}){
		print TRAIN "$line";
	}else{
		print TEST "$line";
	}
}
}
close FILE;

